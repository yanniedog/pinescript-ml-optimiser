#!/usr/bin/env python3
"""
Interactive runner for Pine Script indicator optimization.
Includes data management for downloading any crypto symbol at any timeframe.
"""

import sys
import os
import runpy
import csv
import shutil
import re
from pathlib import Path
import builtins

# Allow overriding input to support go-back shortcuts.
_ORIGINAL_INPUT = builtins.input


def _input_with_go_back(prompt=""):
    response = _ORIGINAL_INPUT(prompt)
    if response is not None and response.strip().lower() in {"b", "back", "go back"}:
        raise GoBack()
    return response


class GoBack(Exception):
    """Raised when the user requests to go back in a menu."""


builtins.input = _input_with_go_back

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

import optimize_indicator as optimize_module
from data_manager import DataManager, VALID_INTERVALS, INTERVAL_NAMES, print_available_data
from pine_parser import parse_pine_script
from optimizer import get_optimizable_params, optimize_indicator as run_optimizer
from output_generator import generate_outputs
import argparse
import json
import time
from typing import Optional, Callable

from objective import calculate_objective_score as objective_score
from screen_log import enable_screen_log

BACKUP_DONE = False


def handle_go_back(message: str = "[INFO] Returning to previous menu.") -> Callable:
    """Decorator to catch GoBack and show a message."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except GoBack:
                print("\n" + message)
        return wrapper
    return decorator

TRIAL_CONTROL_OPTIONS = {
    "max_trials": None,
    "min_runtime_seconds": None,
    "stall_seconds": None,
    "improvement_rate_floor": None,
}


def _get_trial_overrides():
    """Return the currently configured trial controls."""
    return {k: v for k, v in TRIAL_CONTROL_OPTIONS.items() if v is not None}


def apply_trial_overrides(target: dict):
    """Apply trial control defaults to a dictionary (preserving existing entries)."""
    for key, value in _get_trial_overrides().items():
        if key not in target or target[key] is None:
            target[key] = value


def _prompt_trial_setting(prompt: str, current, cast, validator, invalid_msg: str):
    """Helper to prompt for trial control values (int/float)."""
    while True:
        default = f"[{current}]" if current is not None else "[none]"
        entry = input(f"  {prompt} {default} (enter 'clear' to reset, blank to keep): ").strip()
        if not entry:
            return current
        if entry.lower() in {"clear", "none"}:
            return None
        try:
            value = cast(entry)
        except ValueError:
            print(f"  [ERROR] Invalid number: '{entry}'")
            continue
        if not validator(value):
            print(f"  [ERROR] {invalid_msg}")
            continue
        return value


@handle_go_back("[INFO] Returning to main menu.")
def configure_trial_controls():
    """Interactive menu to configure max/min/stall trial settings."""
    print("\n  Enter 'B' at any prompt to return to the previous menu.")
    print("\n" + "=" * 70)
    print("  Trial Control Settings")
    print("=" * 70)
    print(
        "  These settings are applied to subsequent optimizations "
        "and persist until cleared."
    )
    print("  Set 'max trials' to cap the search, 'min runtime' to prevent early exits,")
    print("  'stall timeout' to delay the no-improvement guard, and 'improvement floor'")
    print("  to tune how sensitive the optimizer is to slowing progress.")

    TRIAL_CONTROL_OPTIONS["max_trials"] = _prompt_trial_setting(
        "Max trials",
        TRIAL_CONTROL_OPTIONS["max_trials"],
        int,
        lambda v: v > 0,
        "Enter a positive integer."
    )
    TRIAL_CONTROL_OPTIONS["min_runtime_seconds"] = _prompt_trial_setting(
        "Minimum runtime (seconds) before early-stop checks",
        TRIAL_CONTROL_OPTIONS["min_runtime_seconds"],
        int,
        lambda v: v >= 0,
        "Enter 0 or a positive number."
    )
    TRIAL_CONTROL_OPTIONS["stall_seconds"] = _prompt_trial_setting(
        "Stall timeout (seconds, 0 disables)",
        TRIAL_CONTROL_OPTIONS["stall_seconds"],
        int,
        lambda v: v >= 0,
        "Enter 0 or a positive number."
    )
    TRIAL_CONTROL_OPTIONS["improvement_rate_floor"] = _prompt_trial_setting(
        "Improvement rate floor (%/s)",
        TRIAL_CONTROL_OPTIONS["improvement_rate_floor"],
        float,
        lambda v: True,
        "Enter a number."
    )

    print("\n  Trial controls updated:")
    for label, key in [
        ("Max trials", "max_trials"),
        ("Min runtime (sec)", "min_runtime_seconds"),
        ("Stall timeout (sec)", "stall_seconds"),
        ("Improvement rate floor", "improvement_rate_floor"),
    ]:
        value = TRIAL_CONTROL_OPTIONS[key]
        print(f"    {label}: {value if value is not None else 'none'}")


def split_choice_input(raw: str):
    """Split user input by commas or whitespace."""
    if not raw:
        return []
    return [token.strip() for token in re.split(r"[,\s]+", raw.strip()) if token.strip()]


def ask_yes_no(prompt: str, default: Optional[bool] = True) -> bool:
    """Prompt for a yes/no answer, insisting on validity before proceeding."""
    while True:
        user_input = input(prompt).strip().lower()
        if not user_input:
            if default is not None:
                return default
            print("  [ERROR] Please enter Y or N.")
            continue
        if user_input in {"y", "yes"}:
            return True
        if user_input in {"n", "no"}:
            return False
        print("  [ERROR] Please enter Y or N.")


def get_pine_files(directory: Path = None):
    """Get all .pine files in the directory, excluding optimized ones."""
    current_dir = directory or Path.cwd()
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


def build_args(pine_file, options):
    """Build argument list for optimize_indicator.py main function."""
    args = [str(pine_file)]
    
    if 'max_trials' in options and options['max_trials'] is not None:
        args.extend(['--max-trials', str(options['max_trials'])])
    
    if 'timeout' in options:
        args.extend(['--timeout', str(options['timeout'])])

    if 'min_runtime_seconds' in options and options['min_runtime_seconds'] is not None:
        args.extend(['--min-runtime-seconds', str(options['min_runtime_seconds'])])

    if 'stall_seconds' in options and options['stall_seconds'] is not None:
        args.extend(['--stall-seconds', str(options['stall_seconds'])])

    if 'improvement_rate_floor' in options and options['improvement_rate_floor'] is not None:
        args.extend(['--improvement-rate-floor', str(options['improvement_rate_floor'])])

    if 'symbols' in options:
        args.extend(['--symbols', options['symbols']])
    
    if 'interval' in options:
        args.extend(['--interval', options['interval']])
    
    if options.get('force_download'):
        args.append('--force-download')
    
    if options.get('verbose'):
        args.append('--verbose')
    
    return args


def calculate_objective_score(metrics) -> float:
    """Calculate overall objective score for ranking."""
    return objective_score(metrics)


def _serialize_metrics(metrics):
    return {
        "total_trades": metrics.total_trades,
        "winning_trades": metrics.winning_trades,
        "losing_trades": metrics.losing_trades,
        "total_return": metrics.total_return,
        "avg_return": metrics.avg_return,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
        "avg_holding_bars": metrics.avg_holding_bars,
        "directional_accuracy": metrics.directional_accuracy,
        "mcc": metrics.mcc,
        "roc_auc": metrics.roc_auc,
        "classification_samples": metrics.classification_samples,
        "forecast_horizon": metrics.forecast_horizon,
        "improvement_over_random": metrics.improvement_over_random,
        "tail_capture_rate": metrics.tail_capture_rate,
        "consistency_score": metrics.consistency_score,
    }


def _safe_tag(tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag.strip())
    return cleaned.strip("_")


def _baseline_objective(result) -> float:
    if result is None:
        return 0.0
    baseline = getattr(result, "baseline_objective", None)
    if baseline is None or baseline == 0.0:
        baseline = calculate_objective_score(result.original_metrics)
    return baseline


def _is_improved_result(result) -> bool:
    if result is None:
        return False
    best_obj = calculate_objective_score(result.best_metrics)
    baseline_obj = _baseline_objective(result)
    return best_obj > baseline_obj


def _serialize_params(params: dict) -> str:
    if not params:
        return ""
    return json.dumps(params, sort_keys=True)


def _json_safe_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _serialize_fold_details(folds):
    if not folds:
        return []
    serialized = []
    for fold in folds:
        serialized.append({k: _json_safe_value(v) for k, v in fold.items()})
    return serialized


def _serialize_per_symbol_metrics(per_symbol_metrics):
    if not per_symbol_metrics:
        return {}
    first_value = next(iter(per_symbol_metrics.values()))
    result = {}
    if isinstance(first_value, dict) and 'original' in first_value:
        for symbol, metrics_pair in per_symbol_metrics.items():
            result[symbol] = {
                "original": _serialize_metrics(metrics_pair.get('original')),
                "optimized": _serialize_metrics(metrics_pair.get('optimized')),
            }
    else:
        for symbol, timeframes in per_symbol_metrics.items():
            result[symbol] = {}
            for timeframe, metrics_pair in timeframes.items():
                result[symbol][timeframe] = {
                    "original": _serialize_metrics(metrics_pair.get('original')),
                    "optimized": _serialize_metrics(metrics_pair.get('optimized')),
                }
    return result


def _serialize_data_usage_info(data_usage_info):
    if not data_usage_info:
        return {}
    serialized = {}
    for symbol, timeframes in data_usage_info.items():
        serialized[symbol] = {}
        for timeframe, info in timeframes.items():
            serialized[symbol][timeframe] = {
                "total_bars": _json_safe_value(info.total_bars),
                "date_range": [_json_safe_value(info.date_range[0]), _json_safe_value(info.date_range[1])],
                "n_folds": _json_safe_value(info.n_folds),
                "train_ratio": _json_safe_value(info.train_ratio),
                "embargo_bars": _json_safe_value(info.embargo_bars),
                "folds": _serialize_fold_details(info.folds),
                "total_train_bars": _json_safe_value(info.total_train_bars),
                "total_test_bars": _json_safe_value(info.total_test_bars),
                "total_embargo_bars": _json_safe_value(info.total_embargo_bars),
                "unused_bars": _json_safe_value(info.unused_bars),
                "potential_bias_issues": info.potential_bias_issues,
            }
    return serialized


def write_matrix_reports(summary_dir: Path, run_info: dict, rows: list):
    summary_dir.mkdir(parents=True, exist_ok=True)

    json_path = summary_dir / "unified_optimization_matrix.json"
    csv_path = summary_dir / "unified_optimization_matrix.csv"
    txt_path = summary_dir / "unified_optimization_matrix.txt"

    payload = {
        "run": run_info,
        "matrix": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_safe_value), encoding="utf-8")

    fieldnames = [
        "indicator_name",
        "file_name",
        "symbol",
        "interval",
        "improved",
        "objective_best",
        "baseline_objective",
        "improvement_pf",
        "forecast_horizon",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "max_drawdown",
        "avg_holding_bars",
        "directional_accuracy",
        "mcc",
        "roc_auc",
        "classification_samples",
        "improvement_over_random",
        "tail_capture_rate",
        "consistency_score",
        "total_return",
        "avg_return",
        "optimization_time",
        "n_trials",
        "best_params",
        "original_params",
        "output_pine",
        "output_report",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("best_metrics", {})
            writer.writerow({
                "indicator_name": row.get("indicator_name"),
                "file_name": row.get("file_name"),
                "symbol": row.get("symbol"),
                "interval": row.get("interval"),
                "improved": row.get("improved"),
                "objective_best": row.get("objective_best"),
                "baseline_objective": row.get("baseline_objective"),
                "improvement_pf": row.get("improvement_pf"),
                "forecast_horizon": metrics.get("forecast_horizon"),
                "total_trades": metrics.get("total_trades"),
                "winning_trades": metrics.get("winning_trades"),
                "losing_trades": metrics.get("losing_trades"),
                "win_rate": metrics.get("win_rate"),
                "profit_factor": metrics.get("profit_factor"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "avg_holding_bars": metrics.get("avg_holding_bars"),
                "directional_accuracy": metrics.get("directional_accuracy"),
                "mcc": metrics.get("mcc"),
                "roc_auc": metrics.get("roc_auc"),
                "classification_samples": metrics.get("classification_samples"),
                "improvement_over_random": metrics.get("improvement_over_random"),
                "tail_capture_rate": metrics.get("tail_capture_rate"),
                "consistency_score": metrics.get("consistency_score"),
                "total_return": metrics.get("total_return"),
                "avg_return": metrics.get("avg_return"),
                "optimization_time": row.get("optimization_time"),
                "n_trials": row.get("n_trials"),
                "best_params": _serialize_params(row.get("best_params")),
                "original_params": _serialize_params(row.get("original_params")),
                "output_pine": row.get("output_pine"),
                "output_report": row.get("output_report"),
            })

    lines = []
    lines.append("=" * 70)
    lines.append("UNIFIED OPTIMIZATION MATRIX REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("RUN CONFIG")
    lines.append("-" * 70)
    for key, value in run_info.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append(f"TOTAL ROWS: {len(rows)}")
    lines.append("")

    for row in rows:
        metrics = row.get("best_metrics", {})
        lines.append("-" * 70)
        lines.append(f"{row.get('indicator_name')} | {row.get('symbol')} @ {row.get('interval')}")
        lines.append(
            f"Objective: {row.get('objective_best', 0):.4f} "
            f"(baseline {row.get('baseline_objective', 0):.4f})"
        )
        lines.append(f"Improved: {'yes' if row.get('improved') else 'no'}")
        lines.append(
            f"PF {metrics.get('profit_factor', 0):.2f} | "
            f"Win {metrics.get('win_rate', 0)*100:.1f}% | "
            f"Sharpe {metrics.get('sharpe_ratio', 0):.2f} | "
            f"Forecast {metrics.get('forecast_horizon', 0)} bars"
        )
        lines.append(
            f"MCC {metrics.get('mcc', 0):.3f} | "
            f"AUC {metrics.get('roc_auc', 0):.3f} | "
            f"Samples {metrics.get('classification_samples', 0)}"
        )
        lines.append(f"Params: {_serialize_params(row.get('best_params'))}")
        lines.append(
            "Metrics: "
            f"trades={metrics.get('total_trades', 0)} "
            f"dir_acc={metrics.get('directional_accuracy', 0)*100:.1f}% "
            f"tail={metrics.get('tail_capture_rate', 0)*100:.1f}% "
            f"consistency={metrics.get('consistency_score', 0):.2f} "
            f"impr_random={metrics.get('improvement_over_random', 0):+.1f}%"
        )

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def write_unified_report(summary_path: Path, json_path: Path, run_info: dict, results: list):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build JSON payload
    payload = {
        "run": run_info,
        "indicators": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_safe_value), encoding="utf-8")
    
    # Build text report
    lines = []
    lines.append("=" * 70)
    lines.append("UNIFIED OPTIMIZATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("RUN CONFIG")
    lines.append("-" * 70)
    for key, value in run_info.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Aggregate stats
    objectives = [r["objective_best"] for r in results if r.get("objective_best") is not None]
    times = [r["optimization_time"] for r in results if r.get("optimization_time") is not None]
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    if objectives:
        lines.append(f"Indicators: {len(results)}")
        lines.append(f"Avg objective: {sum(objectives)/len(objectives):.4f}")
        lines.append(f"Max objective: {max(objectives):.4f}")
        lines.append(f"Min objective: {min(objectives):.4f}")
    if times:
        lines.append(f"Avg optimization time: {sum(times)/len(times):.1f}s")
        lines.append(f"Total optimization time: {sum(times):.1f}s")
    lines.append("")
    
    # Top and bottom by objective
    ranked = sorted(results, key=lambda r: r.get("objective_best", 0), reverse=True)
    lines.append("TOP INDICATORS (BY OBJECTIVE)")
    lines.append("-" * 70)
    for row in ranked[:10]:
        lines.append(
            f"{row['indicator_name']}: objective={row['objective_best']:.4f} "
            f"mcc={row['best_metrics'].get('mcc', 0):.3f} "
            f"auc={row['best_metrics'].get('roc_auc', 0):.3f}"
        )
    lines.append("")
    lines.append("BOTTOM INDICATORS (BY OBJECTIVE)")
    lines.append("-" * 70)
    for row in ranked[-10:]:
        lines.append(
            f"{row['indicator_name']}: objective={row['objective_best']:.4f} "
            f"mcc={row['best_metrics'].get('mcc', 0):.3f} "
            f"auc={row['best_metrics'].get('roc_auc', 0):.3f}"
        )
    lines.append("")
    
    # Per-indicator details
    lines.append("PER-INDICATOR DETAIL")
    lines.append("-" * 70)
    for row in results:
        lines.append("")
        lines.append(f"{row['indicator_name']} ({row['file_name']})")
        lines.append(f"  Objective: {row['objective_best']:.4f} (baseline {row['baseline_objective']:.4f})")
        lines.append(f"  Trials: {row['n_trials']} | Time: {row['optimization_time']:.1f}s")
        lines.append(
            f"  MCC: {row['best_metrics'].get('mcc', 0):.3f} | "
            f"AUC: {row['best_metrics'].get('roc_auc', 0):.3f} | "
            f"Samples: {row['best_metrics'].get('classification_samples', 0)}"
        )
        lines.append(
            f"  Profit Factor: {row['best_metrics']['profit_factor']:.2f} | "
            f"Win Rate: {row['best_metrics']['win_rate']*100:.1f}%"
        )
        lines.append(
            f"  Sharpe: {row['best_metrics']['sharpe_ratio']:.2f} | "
            f"Drawdown: {row['best_metrics']['max_drawdown']:.1f}%"
        )
        lines.append(f"  Output Pine: {row['output_pine']}")
        lines.append(f"  Output Report: {row['output_report']}")
        lines.append(f"  Strategy: {row['config']['strategy']} | Sampler: {row['config']['sampler']}")
        lines.append(f"  Timeout(s): {row['config']['timeout_seconds']} | Max trials: {row['config']['max_trials']}")
        lines.append(f"  Early stop patience: {row['config']['early_stop_patience']}")
        lines.append(f"  Min runtime(s): {row['config']['min_runtime_seconds']} | Stall(s): {row['config']['stall_seconds']}")
        lines.append(f"  Rate floor: {row['config']['improvement_rate_floor']} | Rate window: {row['config']['improvement_rate_window']}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def choose_indicator_directory() -> Path:
    """Prompt for indicator directory, defaulting to ./pinescripts if present."""
    default_dir = Path.cwd() / "pinescripts"
    if default_dir.exists():
        default_value = default_dir
    else:
        default_value = Path.cwd()

    available_dirs = find_pine_directories(Path.cwd())
    counts_map = {path: count for path, count in available_dirs}
    default_count = counts_map.get(default_value)
    if default_count is None:
        if default_value.exists():
            default_count = sum(1 for _ in default_value.rglob("*.pine"))
        else:
            default_count = 0
    directory_choices = [(default_value, default_count)]
    for path, count in available_dirs:
        if path == default_value:
            continue
        directory_choices.append((path, count))

    print("\n  Detected Pine Script directories:")
    for idx, (path, count) in enumerate(directory_choices, 1):
        print(f"    [{idx}] {path} ({count} .pine files)")
    print("    [C] Custom directory path")

    prompt = f"  Select directory (1-{len(directory_choices)}) or enter path [default: {default_value}]: "
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default_value
        if user_input.lower() in {"c", "custom"}:
            custom_input = input("  Enter custom directory path: ").strip()
            candidate = Path(custom_input)
        else:
            candidate = None
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(directory_choices):
                    return directory_choices[idx][0]
            candidate = Path(user_input)
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("  [ERROR] Directory not found. Please enter a valid path.")


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


def find_pine_directories(root: Path):
    """Return directories under root that contain .pine files with counts."""
    counts = {}
    for pine_file in root.rglob("*.pine"):
        if pine_file.is_file():
            parent = pine_file.parent
            counts[parent] = counts.get(parent, 0) + 1
    return sorted(counts.items(), key=lambda item: str(item[0]))


def display_indicator_catalog(pine_files, directory: Path = None, limit: int = 30):
    """Print a numbered list of indicators, optionally tied to a directory."""
    heading = f"Indicators in {directory}" if directory else "Indicators"
    print(f"\n{heading} ({len(pine_files)} total):")
    for idx, pine_file in enumerate(pine_files[:limit], 1):
        print(f"  [{idx}] {pine_file.name}")
    if len(pine_files) > limit:
        print(f"  ... and {len(pine_files) - limit} more")


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

    use_all = ask_yes_no("\nUse all available datasets? [Y/n]: ", default=True)
    if not use_all:
        intervals_input = input("Intervals (comma/space-separated, Enter for all): ").strip()
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


def backup_previous_outputs():
    global BACKUP_DONE
    if BACKUP_DONE:
        return

    sources = [Path("optimized_outputs"), Path("pinescripts")]
    existing_sources = [p for p in sources if p.exists()]
    if not existing_sources:
        BACKUP_DONE = True
        return

    has_content = False
    for path in existing_sources:
        if path.is_dir() and any(path.rglob("*")):
            has_content = True
            break
        if path.is_file():
            has_content = True
            break

    if not has_content:
        BACKUP_DONE = True
        return

    backup_root = Path("backup")
    backup_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing_sources:
        dest = backup_dir / path.name
        if path.is_dir():
            shutil.copytree(path, dest)
        else:
            shutil.copy2(path, dest)

    run_info = {}
    summary_dir = Path("optimized_outputs") / "summary"
    for name in ("unified_optimization_matrix.json", "unified_optimization_results.json"):
        candidate = summary_dir / name
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                run_info = payload.get("run", {}) if isinstance(payload, dict) else {}
                break
            except Exception:
                continue

    info_payload = {
        "backed_up_at": timestamp,
        "source_paths": [str(p) for p in existing_sources],
        "run": run_info,
    }
    (backup_dir / "backup_info.json").write_text(
        json.dumps(info_payload, indent=2, default=_json_safe_value),
        encoding="utf-8"
    )

    info_lines = [
        "Backup info",
        f"Backed up at: {timestamp}",
        f"Sources: {', '.join(str(p) for p in existing_sources)}",
        "",
        "Run config:",
    ]
    if run_info:
        for key, value in run_info.items():
            info_lines.append(f"{key}: {value}")
    else:
        info_lines.append("No prior run config found in summary JSON.")
    (backup_dir / "backup_info.txt").write_text("\n".join(info_lines), encoding="utf-8")

    BACKUP_DONE = True


@handle_go_back("[INFO] Returning to main menu.")
def run_optimization(dm: DataManager):
    """Run the optimization workflow."""
    print("\n  Enter 'B' at any prompt to return to the previous menu.")
    backup_previous_outputs()
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

    apply_trial_overrides(options)
    
    # Run optimization
    original_argv = sys.argv
    try:
        sys.argv = ['interactive_optimizer.py'] + build_args(selected_file, options)
        optimize_module.main()
    except SystemExit:
        print("\n[ERROR] Optimization exited early.")
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


def sort_rankings(rankings: list) -> list:
    """Return a copy of the rankings sorted by score descending."""
    return sorted(rankings, key=lambda r: r.get("score", 0), reverse=True)


@handle_go_back("[INFO] Returning to main menu.")
def run_batch_optimization(dm: DataManager):
    """Run optimization across all indicators in a directory with ranking."""
    print("\n  Enter 'B' at any prompt to return to the previous menu.")
    backup_previous_outputs()
    generated = maybe_generate_all_indicators()
    
    if generated:
        indicator_dir = Path.cwd() / "pinescripts"
    else:
        indicator_dir = choose_indicator_directory()
    
    pine_files = get_pine_files(indicator_dir)
    
    if not pine_files:
        print(f"\n[ERROR] No Pine Script files found in: {indicator_dir}")
        return

    eligible_files, ineligible = get_eligible_indicators(pine_files)

    print(f"\nFound {len(pine_files)} Pine Script files in {indicator_dir}")
    print(f"Eligible for optimization: {len(eligible_files)}")
    if ineligible:
        print(f"Skipped (insufficient params or parse errors): {len(ineligible)}")
        for name, reason in ineligible[:10]:
            print(f"  - {name}: {reason}")
        if len(ineligible) > 10:
            print(f"  ... and {len(ineligible) - 10} more")

    if not eligible_files:
        print("\n[ERROR] No indicators eligible for optimization.")
        return
    
    # Ask for time budget
    options = {}
    min_per_indicator_seconds = 60
    print()
    print("Time budget mode:")
    print("  [1] Total minutes split across all indicators")
    print("  [2] Minutes per indicator (each indicator gets full time)")
    while True:
        budget_input = input("  Choose option [1]: ").strip()
        if not budget_input:
            budget_input = "1"
        if budget_input in ["1", "2"]:
            break
        print("[ERROR] Please enter 1 or 2")

    budget_mode = "total" if budget_input == "1" else "per_indicator"
    timeout_minutes = 0.0
    per_indicator_minutes = 0.0
    per_indicator_seconds = None

    if budget_mode == "total":
        while True:
            timeout_input = input("Total minutes to split across all indicators? [5]: ").strip()
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
        
        total_seconds = int(timeout_minutes * 60)
        max_indicators = max(1, total_seconds // min_per_indicator_seconds)
        if max_indicators < len(eligible_files):
            print(
                f"\n[INFO] Time budget allows {max_indicators} indicator(s) at "
                f"{min_per_indicator_seconds}s each. Limiting run to first "
                f"{max_indicators} indicators."
            )
            eligible_files = eligible_files[:max_indicators]

        base_seconds = total_seconds // len(eligible_files)
        extra_seconds = total_seconds % len(eligible_files)
        per_indicator_budgets = [
            base_seconds + (1 if i < extra_seconds else 0)
            for i in range(len(eligible_files))
        ]
    else:
        while True:
            per_indicator_input = input("Minutes per indicator? [5]: ").strip()
            if not per_indicator_input:
                per_indicator_minutes = 5.0
                break
            try:
                per_indicator_minutes = float(per_indicator_input)
                if per_indicator_minutes > 0:
                    break
                else:
                    print("[ERROR] Please enter a positive number")
            except ValueError:
                print("[ERROR] Please enter a valid number")

        per_indicator_seconds = int(per_indicator_minutes * 60)
        total_seconds = per_indicator_seconds * len(eligible_files)
        per_indicator_budgets = [per_indicator_seconds] * len(eligible_files)

    options['timeout'] = per_indicator_budgets[0] if per_indicator_budgets else total_seconds
    options['max_trials'] = None
    
    print(f"\nBatch optimization configured:")
    print(f"  - Budget mode: {budget_mode.replace('_', ' ')}")
    print(f"  - Indicators: {len(eligible_files)}")
    if budget_mode == "total":
        print(f"  - Total time: {timeout_minutes:.1f} minute(s)")
        print(
            f"  - Time per indicator: {min(per_indicator_budgets)/60:.2f}–"
            f"{max(per_indicator_budgets)/60:.2f} minute(s)"
        )
    else:
        print(f"  - Time per indicator: {per_indicator_minutes:.1f} minute(s)")
        print(f"  - Total time (all indicators): {total_seconds/60:.1f} minute(s)")
    print(f"  - Trials: unlimited (will run as many as possible until time limit)")
    print(f"  - Early stop: disabled (uses full timeout per indicator)")
    print(f"  - Press Q anytime to stop early and use current best results")
    
    customize = input("\nCustomize data settings (timeframe, symbols)? [n]: ").strip().lower()
    if customize in ['y', 'yes']:
        extra_options = get_optimization_options(dm)
        extra_options.pop('timeout', None)
        extra_options.pop('max_trials', None)
        options.update(extra_options)
    else:
        print("Using defaults: 1h timeframe, all available symbols")
        options['interval'] = '1h'

    apply_trial_overrides(options)
    
    rankings = []
    results = []
    skipped_no_improvement = 0
    
    for i, pine_file in enumerate(eligible_files, 1):
        print(f"\n{'='*70}")
        print(f"Processing {i}/{len(eligible_files)}: {pine_file.name}")
        print("="*70)

        indicator_timeout = per_indicator_budgets[i - 1]
        options['timeout'] = indicator_timeout
        options['min_runtime_seconds'] = indicator_timeout
        options['stall_seconds'] = indicator_timeout + 1
        options['improvement_rate_floor'] = 0.0
        
        original_argv = sys.argv
        try:
            sys.argv = ['interactive_optimizer.py'] + build_args(pine_file, options)
            optimize_module.main()
        except SystemExit:
            print(f"[ERROR] Optimization exited early for {pine_file.name}.")
        except KeyboardInterrupt:
            print("\n\nBatch optimization interrupted by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Error during optimization of {pine_file.name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.argv = original_argv
        
        result = optimize_module.LAST_RESULT
        outputs = optimize_module.LAST_OUTPUTS
        if result is None:
            continue
        
        improved = _is_improved_result(result)
        if not improved:
            skipped_no_improvement += 1
            print(f"[INFO] No improvement for {pine_file.name}; keeping baseline outputs.")

        metrics = result.best_metrics
        score = calculate_objective_score(metrics)
        rankings.append({
            "file": pine_file.name,
            "score": score,
            "profit_factor": metrics.profit_factor,
            "win_rate": metrics.win_rate,
            "directional_accuracy": metrics.directional_accuracy,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "mcc": metrics.mcc,
            "roc_auc": metrics.roc_auc,
            "improvement_pf": result.improvement_pf,
        })
        
        if outputs:
            # Safely get LAST_PINE_PATH - getattr returns None if attribute exists but is None
            last_path = getattr(optimize_module, "LAST_PINE_PATH", None)
            indicator_name = (result.best_metrics and last_path and last_path.stem) or pine_file.stem
            results.append({
                "indicator_name": indicator_name,
                "file_name": pine_file.name,
                "output_pine": outputs.get("pine_script"),
                "output_report": outputs.get("report"),
                "optimization_time": result.optimization_time,
                "n_trials": result.n_trials,
                "objective_best": calculate_objective_score(result.best_metrics),
                "baseline_objective": result.baseline_objective,
                "improved": improved,
                "best_metrics": _serialize_metrics(result.best_metrics),
                "original_metrics": _serialize_metrics(result.original_metrics),
                "original_params": result.original_params,
                "best_params": result.best_params,
                "per_symbol_metrics": _serialize_per_symbol_metrics(result.per_symbol_metrics),
                "data_usage_info": _serialize_data_usage_info(result.data_usage_info),
                "datasets_used": result.datasets_used,
                "interval": result.interval,
                "config": {
                    "strategy": getattr(result, "strategy", "tpe"),
                    "sampler": getattr(result, "sampler_name", "tpe"),
                    "timeout_seconds": getattr(result, "timeout_seconds", 0),
                    "max_trials": getattr(result, "max_trials", None),
                    "early_stop_patience": getattr(result, "early_stop_patience", None),
                    "min_runtime_seconds": getattr(result, "min_runtime_seconds", 0),
                    "stall_seconds": getattr(result, "stall_seconds", None),
                    "improvement_rate_floor": getattr(result, "improvement_rate_floor", 0.0),
                    "improvement_rate_window": getattr(result, "improvement_rate_window", 0),
                    "backtester_overrides": getattr(result, "backtester_overrides", {}),
                    "holdout_ratio": getattr(result, "holdout_ratio", 0.0),
                    "holdout_gap_bars": getattr(result, "holdout_gap_bars", 0),
                },
            })
    
    if not rankings:
        print("\nNo successful optimizations to rank.")
        return
    
    rankings = sort_rankings(rankings)
    
    print("\n" + "="*70)
    print("  OPTIMIZATION RANKINGS (Best to Worst)")
    print("="*70)
    print("  Rank  Score  MCC   AUC   PF    Win%  DirAcc  Sharpe  Drawdown  Indicator")
    print("  ----  -----  ----  ----  ----  ----- ------  ------  --------  ---------")
    for idx, row in enumerate(rankings, 1):
        print(
            f"  {idx:>4}  {row['score']:.3f}  {row['mcc']:.3f}  {row['roc_auc']:.3f}  "
            f"{row['profit_factor']:.2f}  {row['win_rate']*100:>5.1f}  {row['directional_accuracy']*100:>6.1f}  "
            f"{row['sharpe_ratio']:>6.2f}  {row['max_drawdown']:>8.2f}  {row['file']}"
        )
    
    # Unified summary outputs
    run_info = {
        "indicator_directory": str(indicator_dir),
        "interval": options.get("interval"),
        "symbols": options.get("symbols", "all available"),
        "budget_mode": budget_mode,
        "total_timeout_seconds": total_seconds,
        "timeout_seconds_per_indicator_min": min(per_indicator_budgets) if per_indicator_budgets else 0,
        "timeout_seconds_per_indicator_max": max(per_indicator_budgets) if per_indicator_budgets else 0,
        "min_per_indicator_seconds": min_per_indicator_seconds,
        "per_indicator_seconds": per_indicator_seconds,
        "generated_all": generated,
        "total_indicators": len(pine_files),
        "eligible_indicators": len(eligible_files),
        "skipped_no_improvement": skipped_no_improvement,
    }
    summary_dir = Path("optimized_outputs") / "summary"
    write_unified_report(
        summary_dir / "unified_optimization_report.txt",
        summary_dir / "unified_optimization_results.json",
        run_info,
        results
    )


@handle_go_back("[INFO] Returning to main menu.")
def run_matrix_optimization(dm: DataManager):
    """Run optimization independently for each indicator-symbol-timeframe combination."""
    print("\n  Enter 'B' at any prompt to return to the previous menu.")
    backup_previous_outputs()
    generated = maybe_generate_all_indicators()

    if generated:
        indicator_dir = Path.cwd() / "pinescripts"
    else:
        indicator_dir = choose_indicator_directory()

    pine_files = get_pine_files(indicator_dir)

    if not pine_files:
        print(f"\n[ERROR] No Pine Script files found in: {indicator_dir}")
        return

    display_indicator_catalog(pine_files, directory=indicator_dir, limit=30)

    eligible_files, ineligible = get_eligible_indicators(pine_files)

    print(f"\nFound {len(pine_files)} Pine Script files in {indicator_dir}")
    print(f"Eligible for optimization: {len(eligible_files)}")
    if ineligible:
        print(f"Skipped (insufficient params or parse errors): {len(ineligible)}")
        for name, reason in ineligible[:10]:
            print(f"  - {name}: {reason}")
        if len(ineligible) > 10:
            print(f"  ... and {len(ineligible) - 10} more")

    if not eligible_files:
        print("\n[ERROR] No indicators eligible for optimization.")
        return

    selected_indicators = select_indicator_subset(eligible_files)
    if not selected_indicators:
        print("\n[ERROR] No indicators selected for optimization.")
        return

    datasets = select_datasets_for_matrix(dm)
    if not datasets:
        return

    def build_combos():
        return [
            (pine_file, symbol, interval)
            for pine_file in selected_indicators
            for symbol, interval in datasets
        ]

    combos = build_combos()

    if not combos:
        print("\n[ERROR] No indicator/dataset combinations to run.")
        return

    print()
    print("Time budget mode:")
    print("  [1] Total minutes split across all combinations")
    print("  [2] Minutes per combination (each combination gets full time)")
    while True:
        budget_input = input("  Choose option [1]: ").strip()
        if not budget_input:
            budget_input = "1"
        if budget_input in ["1", "2"]:
            break
        print("[ERROR] Please enter 1 or 2")

    budget_mode = "total" if budget_input == "1" else "per_combo"
    timeout_minutes = 0.0
    per_combo_minutes = 0.0
    per_combo_seconds = None

    if budget_mode == "total":
        need_time = True
        while True:
            if need_time:
                timeout_input = input("Total minutes to split across all combinations? [10]: ").strip()
                if not timeout_input:
                    timeout_minutes = 10.0
                else:
                    try:
                        timeout_minutes = float(timeout_input)
                        if timeout_minutes <= 0:
                            print("[ERROR] Please enter a positive number")
                            continue
                    except ValueError:
                        print("[ERROR] Please enter a valid number")
                        continue
                need_time = False

            combos = build_combos()
            if not combos:
                print("\n[ERROR] No indicator/dataset combinations to run.")
                return

            total_seconds = float(timeout_minutes * 60)
            per_combo_seconds = total_seconds / len(combos)

            print("\nMatrix scope summary:")
            print(f"  - Indicators: {len(selected_indicators)}")
            print(f"  - Datasets: {len(datasets)}")
            print(f"  - Total combinations: {len(combos)}")
            print(f"  - Total time: {timeout_minutes:.1f} minute(s)")
            print(f"  - Time per combo: {per_combo_seconds/60:.2f} minute(s)")
            if per_combo_seconds < 1.0:
                print("  [WARN] Per-combo time < 1 second; results may be unstable.")

            choice = input("Proceed with this allocation? [P]roceed / [T]ime / [S]cope: ").strip().lower()
            if not choice or choice in ["p", "proceed", "y", "yes"]:
                per_combo_budgets = [per_combo_seconds] * len(combos)
                break
            if choice.startswith("t"):
                need_time = True
                continue
            if choice.startswith("s"):
                datasets = select_datasets_for_matrix(dm)
                if not datasets:
                    return
                selected_indicators = select_indicator_subset(eligible_files)
                continue
            print("[ERROR] Please enter P, T, or S.")
    else:
        while True:
            per_combo_input = input("Minutes per combination? [3]: ").strip()
            if not per_combo_input:
                per_combo_minutes = 3.0
                break
            try:
                per_combo_minutes = float(per_combo_input)
                if per_combo_minutes > 0:
                    break
                else:
                    print("[ERROR] Please enter a positive number")
            except ValueError:
                print("[ERROR] Please enter a valid number")

        combos = build_combos()
        if not combos:
            print("\n[ERROR] No indicator/dataset combinations to run.")
            return

        per_combo_seconds = float(per_combo_minutes * 60)
        total_seconds = per_combo_seconds * len(combos)
        per_combo_budgets = [per_combo_seconds] * len(combos)

    print(f"\nMatrix optimization configured:")
    print(f"  - Budget mode: {budget_mode.replace('_', ' ')}")
    print(f"  - Indicators: {len(selected_indicators)}")
    print(f"  - Datasets: {len(datasets)}")
    print(f"  - Combinations to run: {len(combos)}")
    if budget_mode == "total":
        print(f"  - Total time: {timeout_minutes:.1f} minute(s)")
        print(f"  - Time per combo: {per_combo_seconds/60:.2f} minute(s)")
        if per_combo_seconds < 1.0:
            print("  [WARN] Per-combo time < 1 second; results may be unstable.")
    else:
        print(f"  - Time per combo: {per_combo_minutes:.1f} minute(s)")
        print(f"  - Total time (all combos): {total_seconds/60:.1f} minute(s)")
    print(f"  - Trials: unlimited (runs until time limit per combo)")
    print(f"  - Early stop: disabled (uses full timeout per combo)")
    print(f"  - Press Q anytime to stop early and use current best results")

    results = []
    skipped_no_improvement = 0
    errors = 0
    data_cache = {}
    parse_cache = {}

    for idx, ((pine_file, symbol, interval), combo_timeout) in enumerate(zip(combos, per_combo_budgets), 1):
        print(f"\n{'='*70}")
        print(f"Combo {idx}/{len(combos)}: {pine_file.name} | {symbol} @ {interval}")
        print("="*70)

        if pine_file not in parse_cache:
            try:
                parse_cache[pine_file] = parse_pine_script(str(pine_file))
            except Exception as exc:
                errors += 1
                print(f"[ERROR] Failed to parse {pine_file.name}: {exc}")
                continue

        parse_result = parse_cache[pine_file]

        key = (symbol, interval)
        if key not in data_cache:
            try:
                data_cache[key] = dm.load_symbol(symbol, interval)
            except Exception as exc:
                errors += 1
                print(f"[ERROR] Failed to load data for {symbol} @ {interval}: {exc}")
                continue

        data = {symbol: data_cache[key]}

        combo_label = f"{parse_result.indicator_name or pine_file.stem}:{symbol}@{interval}"

        run_kwargs = {
            "interval": interval,
            "max_trials": None,
            "timeout_seconds": combo_timeout,
            "min_runtime_seconds": combo_timeout,
            "stall_seconds": combo_timeout + 1,
            "improvement_rate_floor": 0.0,
            "indicator_label": combo_label,
        }
        apply_trial_overrides(run_kwargs)

        try:
            result = run_optimizer(
                parse_result,
                data,
                **run_kwargs
            )
        except KeyboardInterrupt:
            print("\n\nMatrix optimization interrupted by user.")
            break
        except Exception as exc:
            errors += 1
            print(f"[ERROR] Optimization failed for {pine_file.name} {symbol} @ {interval}: {exc}")
            import traceback
            traceback.print_exc()
            continue

        improved = _is_improved_result(result)
        if not improved:
            skipped_no_improvement += 1
            print(f"[INFO] No improvement for {pine_file.name} {symbol} @ {interval}; keeping baseline outputs.")

        output_tag = _safe_tag(f"{symbol}_{interval}")
        outputs = generate_outputs(parse_result, result, str(pine_file), output_tag=output_tag)

        row = {
            "indicator_name": parse_result.indicator_name or pine_file.stem,
            "file_name": pine_file.name,
            "symbol": symbol,
            "interval": interval,
            "improved": improved,
            "output_pine": outputs.get("pine_script"),
            "output_report": outputs.get("report"),
            "optimization_time": result.optimization_time,
            "n_trials": result.n_trials,
            "objective_best": calculate_objective_score(result.best_metrics),
            "baseline_objective": _baseline_objective(result),
            "improvement_pf": result.improvement_pf,
            "best_metrics": _serialize_metrics(result.best_metrics),
            "original_metrics": _serialize_metrics(result.original_metrics),
            "original_params": result.original_params,
            "best_params": result.best_params,
        }
        results.append(row)

    if not results:
        print("\nNo successful optimizations to report.")
        return

    intervals = sorted(set(interval for _, interval in datasets))
    symbols = sorted(set(symbol for symbol, _ in datasets))
    run_info = {
        "indicator_directory": str(indicator_dir),
        "intervals": intervals,
        "symbols": symbols,
        "datasets": len(datasets),
        "combo_total": len(combos),
        "combo_run": len(combos),
        "budget_mode": budget_mode,
        "total_timeout_seconds": total_seconds,
        "timeout_seconds_per_combo_min": min(per_combo_budgets) if per_combo_budgets else 0,
        "timeout_seconds_per_combo_max": max(per_combo_budgets) if per_combo_budgets else 0,
        "per_combo_seconds": per_combo_seconds,
        "generated_all": generated,
        "total_indicators": len(pine_files),
        "eligible_indicators": len(eligible_files),
        "selected_indicators": len(selected_indicators),
        "skipped_no_improvement": skipped_no_improvement,
        "errors": errors,
    }

    summary_dir = Path("optimized_outputs") / "summary"
    write_matrix_reports(summary_dir, run_info, results)


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
    [4] Optimize ALL indicators in a directory
    [5] Optimize all indicators per symbol/timeframe (matrix)
    [6] Configure trial controls (max/min/stall)
    [Q] Quit
""")
        print("  Enter 'B' at any prompt to return to the previous menu (main menu only).")
        try:
            choice = input("  Select option: ").strip().lower()
        except GoBack:
            print("  [INFO] Already at the main menu.")
            continue
        
        if choice == '1':
            run_optimization(dm)
        elif choice == '2':
            download_new_data(dm)
        elif choice == '3':
            display_data_status(dm)
            input("\n  Press Enter to continue...")
        elif choice == '4':
            run_batch_optimization(dm)
        elif choice == '5':
            run_matrix_optimization(dm)
        elif choice == '6':
            configure_trial_controls()
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            print("\n  [ERROR] Invalid option. Please enter 1, 2, 3, 4, 5, or Q.")


def main():
    """Main entry point."""
    enable_screen_log()
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
