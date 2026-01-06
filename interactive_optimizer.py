#!/usr/bin/env python3
"""
Interactive runner for Pine Script indicator optimization.
Includes data management for downloading any crypto symbol at any timeframe.
"""

import sys
import os
from pathlib import Path

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from optimize_indicator import main as optimize_main
from data_manager import DataManager, VALID_INTERVALS, INTERVAL_NAMES, print_available_data
import argparse


def get_pine_files():
    """Get all .pine files in the current directory, excluding optimized ones."""
    current_dir = Path.cwd()
    pine_files = list(current_dir.glob("*.pine"))
    pine_files = [f for f in pine_files if not f.name.startswith('optimised_')]
    return sorted(pine_files)


def display_pine_menu(pine_files):
    """Display numbered menu of available Pine Script files."""
    if not pine_files:
        print("\n[ERROR] No Pine Script files found in the current directory.")
        print("   Please ensure you have .pine files in:", Path.cwd())
        return None
    
    print("\n" + "="*70)
    print("  Available Pine Script Indicators")
    print("="*70)
    
    for i, file in enumerate(pine_files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"  [{i}] {file.name:<40} ({size_kb:.1f} KB)")
    
    print("="*70)
    return pine_files


def get_user_choice(pine_files):
    """Get user's file selection."""
    while True:
        try:
            choice = input(f"\nSelect indicator to optimize (1-{len(pine_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(pine_files):
                return pine_files[choice_num - 1]
            else:
                print(f"[ERROR] Please enter a number between 1 and {len(pine_files)}")
        except ValueError:
            print("[ERROR] Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n")
            return None


def display_data_status(dm: DataManager, interval: str = None):
    """Display current data status."""
    datasets = dm.get_available_datasets()
    
    print("\n" + "-"*70)
    print("  Current Data Status")
    print("-"*70)
    
    if not datasets:
        print("  No datasets found. Use 'Download new data' to get started.")
        return
    
    # Group by interval
    by_interval = {}
    for symbol, intv in datasets:
        if intv not in by_interval:
            by_interval[intv] = []
        by_interval[intv].append(symbol)
    
    for intv in sorted(by_interval.keys()):
        symbols = by_interval[intv]
        interval_name = INTERVAL_NAMES.get(intv, intv)
        marker = " <-- selected" if intv == interval else ""
        print(f"\n  {interval_name} ({intv}): {len(symbols)} symbols{marker}")
        display_symbols = symbols[:8]
        if len(symbols) > 8:
            print(f"    {', '.join(display_symbols)}, ... (+{len(symbols)-8} more)")
        else:
            print(f"    {', '.join(display_symbols)}")


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


def download_new_data(dm: DataManager):
    """Interactive data download."""
    print("\n" + "="*70)
    print("  Download New Data")
    print("="*70)
    
    # Select timeframe
    print("\n  Step 1: Select timeframe")
    print(f"  Valid timeframes: {', '.join(VALID_INTERVALS)}")
    
    while True:
        interval = input("  Timeframe [1h]: ").strip().lower()
        if not interval:
            interval = "1h"
        if interval in VALID_INTERVALS:
            break
        print(f"  [ERROR] Invalid. Choose from: {', '.join(VALID_INTERVALS)}")
    
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
    print(f"\n  Will download {len(symbols)} symbols at {interval} timeframe:")
    print(f"    {', '.join(s + 'USDT' for s in symbols[:10])}")
    if len(symbols) > 10:
        print(f"    ... and {len(symbols) - 10} more")
    
    confirm = input("\n  Proceed with download? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("  Download cancelled.")
        return
    
    # Download
    print()
    for symbol in symbols:
        full_symbol = symbol + 'USDT' if not symbol.endswith('USDT') else symbol
        try:
            dm.download_symbol(full_symbol, interval)
        except Exception as e:
            print(f"  [ERROR] Failed to download {full_symbol}: {e}")
    
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
    
    symbols_input = input("\n  Symbols (comma-separated, or Enter for all available) [all]: ").strip()
    if symbols_input:
        options['symbols'] = symbols_input.upper()
    
    # Force download
    force_input = input("  Force re-download data? [n]: ").strip().lower()
    if force_input in ['y', 'yes']:
        options['force_download'] = True
    
    # Verbose
    verbose_input = input("  Verbose logging? [n]: ").strip().lower()
    if verbose_input in ['y', 'yes']:
        options['verbose'] = True
    
    return options


def build_args(pine_file, options):
    """Build argument list for optimize_indicator.py main function."""
    args = [str(pine_file)]
    
    if 'max_trials' in options and options['max_trials'] is not None:
        args.extend(['--max-trials', str(options['max_trials'])])
    
    if 'timeout' in options:
        args.extend(['--timeout', str(options['timeout'])])
    
    if 'symbols' in options:
        args.extend(['--symbols', options['symbols']])
    
    if 'interval' in options:
        args.extend(['--interval', options['interval']])
    
    if options.get('force_download'):
        args.append('--force-download')
    
    if options.get('verbose'):
        args.append('--verbose')
    
    return args


def run_optimization(dm: DataManager):
    """Run the optimization workflow."""
    pine_files = get_pine_files()
    
    if not pine_files:
        return
    
    display_pine_menu(pine_files)
    selected_file = get_user_choice(pine_files)
    
    if selected_file is None:
        return
    
    print(f"\n[OK] Selected: {selected_file.name}")
    
    # Ask for timeout
    options = {}
    print()
    while True:
        timeout_input = input("How many minutes should the ML optimization run? [5]: ").strip()
        if not timeout_input:
            timeout_minutes = 5.0
            break
        try:
            timeout_minutes = float(timeout_input)
            if timeout_minutes > 0:
                break
            else:
                print("[ERROR] Please enter a positive number")
        except ValueError:
            print("[ERROR] Please enter a valid number")
    
    options['timeout'] = int(timeout_minutes * 60)
    options['max_trials'] = None
    
    print(f"\nOptimization configured:")
    print(f"  - Time limit: {timeout_minutes:.1f} minute(s)")
    print(f"  - Trials: unlimited (will run as many as possible until time limit)")
    print(f"  - Press Q anytime to stop early and use current best results")
    
    # Ask if user wants to customize
    customize = input("\nCustomize data settings (timeframe, symbols)? [n]: ").strip().lower()
    
    if customize in ['y', 'yes']:
        extra_options = get_optimization_options(dm)
        extra_options.pop('timeout', None)
        extra_options.pop('max_trials', None)
        options.update(extra_options)
    else:
        print("Using defaults: 1h timeframe, all available symbols")
        options['interval'] = '1h'
    
    # Run optimization
    original_argv = sys.argv
    try:
        sys.argv = ['interactive_optimizer.py'] + build_args(selected_file, options)
        optimize_main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


def main_menu():
    """Display main menu and handle selection."""
    dm = DataManager()
    
    while True:
        print("""
======================================================================
         PINE SCRIPT ML INDICATOR OPTIMIZER - Interactive Mode        
         Bayesian Optimization with Walk-Forward Validation           
======================================================================

  Main Menu:
    [1] Optimize an indicator
    [2] Download new data (any crypto symbol, any timeframe)
    [3] View available data
    [Q] Quit
""")
        
        choice = input("  Select option: ").strip().lower()
        
        if choice == '1':
            run_optimization(dm)
        elif choice == '2':
            download_new_data(dm)
        elif choice == '3':
            display_data_status(dm)
            input("\n  Press Enter to continue...")
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            print("\n  [ERROR] Invalid option. Please enter 1, 2, 3, or Q.")


def main():
    """Main entry point."""
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
