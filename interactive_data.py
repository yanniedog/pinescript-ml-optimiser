"""
Data management and selection utilities for the interactive optimizer.
"""

import re
import runpy
from pathlib import Path
from data_manager import DataManager, VALID_INTERVALS, INTERVAL_NAMES
from pine_parser import parse_pine_script
from optimizer import get_optimizable_params
from interactive_ui import (
    ask_yes_no, 
    split_choice_input, 
    display_data_status, 
    handle_go_back, 
    display_indicator_catalog
)


def select_timeframe(dm: DataManager) -> str:
    """Let user select a timeframe."""
    available = dm.get_available_intervals()
    
    print("\n" + "-"*70)
    print("  Select Timeframe")
    print("-"*70)
    
    print("\n  Available timeframes with existing data:")
    for i, intv in enumerate(available, 1):
        symbols = dm.get_available_symbols(intv)
        name = INTERVAL_NAMES.get(intv, intv)
        print(f"    [{i}] {name} ({intv}) - {len(symbols)} symbols")
    
    print(f"\n  All valid Binance timeframes:")
    print(f"    {', '.join(VALID_INTERVALS)}")
    
    while True:
        choice = input(f"\n  Enter timeframe (e.g., '1h', '4h', '1d') [1h]: ").strip().lower()
        if not choice:
            return "1h"
        if choice in VALID_INTERVALS:
            return choice
        # Try to match by number if they selected from available list
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                return available[idx]
        except ValueError:
            pass
        print(f"  [ERROR] Invalid timeframe. Valid options: {', '.join(VALID_INTERVALS)}")


@handle_go_back("[INFO] Returning to main menu.")
def download_new_data(dm: DataManager):
    """Interactive data download."""
    print("\n  Enter 'B' at any prompt to return to the previous menu.")
    print("\n" + "="*70)
    print("  Download New Data")
    print("="*70)
    
    # Select timeframe
    print("\n  Step 1: Select timeframe")
    print(f"  Valid timeframes: {', '.join(VALID_INTERVALS)}")
    
    while True:
        interval_input = input("  Timeframes [1h] (comma/space separated): ").strip().lower()
        if not interval_input:
            intervals = ["1h"]
            break
        parts = [part for part in re.split(r"[,\s]+", interval_input) if part]
        invalid = [part for part in parts if part not in VALID_INTERVALS]
        if invalid:
            print(
                "  [ERROR] Invalid timeframe(s): "
                f"{', '.join(sorted(set(invalid)))}. Choose from: {', '.join(VALID_INTERVALS)}"
            )
            continue
        intervals = []
        seen = set()
        for part in parts:
            if part not in seen:
                intervals.append(part)
                seen.add(part)
        break
    
    # Get symbols
    print(f"\n  Step 2: Enter symbols to download")
    print("  - Enter comma-separated symbols (e.g., 'BTC, ETH, SOL')")
    print("  - USDT suffix is added automatically")
    print("  - Enter 'list' to see available Binance pairs")
    print("  - Enter 'top20' for top 20 by market cap")
    
    while True:
        symbols_input = input("\n  Symbols: ").strip().upper()
        
        if not symbols_input:
            print("  [ERROR] Please enter at least one symbol")
            continue
        
        if symbols_input == 'LIST':
            print("\n  Fetching available USDT pairs from Binance...")
            all_symbols = dm.fetch_usdt_symbols()
            if all_symbols:
                print(f"\n  Found {len(all_symbols)} USDT pairs. First 50:")
                for i in range(0, min(50, len(all_symbols)), 10):
                    print(f"    {', '.join(all_symbols[i:i+10])}")
                print(f"\n  ... and {len(all_symbols) - 50} more" if len(all_symbols) > 50 else "")
            continue
        
        if symbols_input == 'TOP20':
            symbols = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'DOT', 
                      'MATIC', 'LTC', 'SHIB', 'TRX', 'AVAX', 'LINK', 'ATOM',
                      'UNI', 'XMR', 'ETC', 'XLM', 'PAXG']
            print(f"  Selected top 20: {', '.join(symbols)}")
        else:
            symbols = [s.strip().replace('USDT', '') for s in symbols_input.split(',')]
        
        break
    
    # Confirm
    print(f"\n  Will download {len(symbols)} symbols across {len(intervals)} timeframe(s):")
    print(f"    Timeframes: {', '.join(intervals)}")
    print(f"    {', '.join(s + 'USDT' for s in symbols[:10])}")
    if len(symbols) > 10:
        print(f"    ... and {len(symbols) - 10} more")
    
    if not ask_yes_no("\n  Proceed with download? [Y/n]: ", default=True):
        print("  Download cancelled.")
        return
    
    # Download
    print()
    for interval in intervals:
        print(f"\n  Downloading timeframe: {interval}")
        for symbol in symbols:
            full_symbol = symbol + 'USDT' if not symbol.endswith('USDT') else symbol
            try:
                dm.download_symbol(full_symbol, interval)
            except Exception as e:
                print(f"  [ERROR] Failed to download {full_symbol} @ {interval}: {e}")
    
    print("\n  Download complete!")


def get_optimization_options(dm: DataManager):
    """Get optional optimization parameters from user."""
    options = {}
    
    print("\n" + "-"*70)
    print("  Additional Settings")
    print("-"*70)
    
    # Show available data and let user select
    display_data_status(dm)
    
    # Timeframe selection
    print("\n  Select timeframe for optimization:")
    available_intervals = dm.get_available_intervals()
    if available_intervals:
        print(f"  Available: {', '.join(available_intervals)}")
    
    interval = input("  Timeframe [1h]: ").strip().lower()
    if not interval:
        interval = "1h"
    options['interval'] = interval
    
    # Symbol selection
    available_symbols = dm.get_available_symbols(interval)
    if available_symbols:
        print(f"\n  Available symbols for {interval}: {len(available_symbols)}")
        print(f"    {', '.join(available_symbols[:10])}")
        if len(available_symbols) > 10:
            print(f"    ... and {len(available_symbols) - 10} more")
    
    symbols_input = input("\n  Symbols (comma/space-separated, or Enter for all available) [all]: ").strip()
    if symbols_input:
        normalized_symbols = []
        seen_symbols = set()
        for token in split_choice_input(symbols_input.upper()):
            if not token:
                continue
            candidate = token if token.endswith("USDT") else f"{token}USDT"
            if candidate not in seen_symbols:
                normalized_symbols.append(candidate)
                seen_symbols.add(candidate)
        if normalized_symbols:
            options['symbols'] = ",".join(normalized_symbols)
    
    # Force download
    force_input = input("  Force re-download data? [n]: ").strip().lower()
    if force_input in ['y', 'yes']:
        options['force_download'] = True
    
    # Verbose
    verbose_input = input("  Verbose logging? [n]: ").strip().lower()
    if verbose_input in ['y', 'yes']:
        options['verbose'] = True
    
    return options


def select_datasets_for_matrix(dm: DataManager):
    datasets = dm.get_available_datasets()
    if not datasets:
        print("\n[ERROR] No datasets found. Download data first.")
        return []

    print("\n" + "-" * 70)
    print("  Available datasets")
    print("-" * 70)
    print(f"  Total datasets: {len(datasets)}")
    by_interval = {}
    for symbol, interval in datasets:
        by_interval.setdefault(interval, []).append(symbol)
    for interval in sorted(by_interval.keys()):
        symbols = by_interval[interval]
        interval_name = INTERVAL_NAMES.get(interval, interval)
        print(f"  {interval_name} ({interval}): {len(symbols)} symbols")

    use_all = ask_yes_no("\nUse all available datasets? [y/N]: ", default=False)
    if not use_all:
        intervals_input = input("Intervals (comma/space-separated, Enter for all) [1h]: ").strip()
        if not intervals_input:
            intervals_input = "1h"
        symbols_input = input("Symbols (comma/space-separated, Enter for all): ").strip()

        intervals = None
        if intervals_input:
            intervals = [part.lower() for part in split_choice_input(intervals_input)]

        symbols = None
        if symbols_input:
            symbols = []
            seen = set()
            for part in split_choice_input(symbols_input):
                if not part:
                    continue
                candidate = part.upper()
                if not candidate.endswith("USDT"):
                    candidate = f"{candidate}USDT"
                if candidate not in seen:
                    symbols.append(candidate)
                    seen.add(candidate)

        filtered = []
        for symbol, interval in datasets:
            if intervals and interval not in intervals:
                continue
            if symbols and symbol not in symbols:
                continue
            filtered.append((symbol, interval))
        datasets = filtered

    if not datasets:
        print("\n[ERROR] No datasets match the selection.")
        return []

    print(f"\nUsing {len(datasets)} datasets. Sample:")
    for symbol, interval in datasets[:10]:
        print(f"  - {symbol} @ {interval}")
    if len(datasets) > 10:
        print(f"  ... and {len(datasets) - 10} more")

    return datasets


def maybe_generate_all_indicators():
    """Optionally generate the full set of indicators using the generator script."""
    choice = input("\nGenerate all 100 indicators now? [n]: ").strip().lower()
    if choice not in ['y', 'yes']:
        return False
    
    generator_path = Path(__file__).parent / "create-pinescript-indicators.py"
    if not generator_path.exists():
        print(f"\n[ERROR] Indicator generator script not found: {generator_path}")
        return False
    
    print("\nGenerating indicators...")
    try:
        runpy.run_path(str(generator_path), run_name="__main__")
        print("[OK] Indicator generation complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to generate indicators: {e}")
        return False


def get_eligible_indicators(pine_files):
    eligible_files = []
    ineligible = []
    for pine_file in pine_files:
        try:
            parse_result = parse_pine_script(str(pine_file))
            optimizable_params = get_optimizable_params(parse_result.parameters)
            if len(optimizable_params) >= 2:
                eligible_files.append(pine_file)
            else:
                ineligible.append((pine_file.name, len(optimizable_params)))
        except Exception as exc:
            ineligible.append((pine_file.name, f"parse error: {exc}"))
    return eligible_files, ineligible


def select_indicator_subset(eligible_files):
    if not eligible_files:
        return []

    display_indicator_catalog(eligible_files, limit=30)

    selection_input = input(
        "\nEnter indicator numbers or name fragments (comma/space-separated), "
        "or press Enter to use all: "
    ).strip()
    if not selection_input:
        return eligible_files

    tokens = split_choice_input(selection_input)
    if not tokens:
        return eligible_files

    selected = []
    seen = set()
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(eligible_files):
                if idx not in seen:
                    selected.append(eligible_files[idx - 1])
                    seen.add(idx)
            continue
        lower_token = token.lower()
        for f in eligible_files:
            if f in selected:
                continue
            stem = f.stem.lower()
            name = f.name.lower()
            if lower_token in stem or lower_token in name:
                selected.append(f)
    if not selected:
        print("[ERROR] No indicators matched the provided selection. Keeping full list.")
        return eligible_files

    return selected
