#!/usr/bin/env python3
"""
Pine Script Indicator ML Optimizer

Optimizes Pine Script indicator parameters using Bayesian optimization
with walk-forward validation to maximize leading indicator profitability.

Usage:
    python optimize_indicator.py <pine_script_file> [options]

Options:
    --max-trials N      Maximum optimization trials (default: 150)
    --timeout N         Maximum time in seconds (default: 300)
    --symbols LIST      Comma-separated symbols to test (default: all)
    --force-download    Force re-download of historical data
    --verbose           Enable verbose logging
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from data_manager import DataManager, SYMBOLS
from pine_parser import parse_pine_script
from optimizer import optimize_indicator, get_optimizable_params
from output_generator import generate_outputs
from screen_log import enable_screen_log

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_OPTIMIZATION_STRATEGY = "tpe"

LAST_RESULT = None
LAST_OUTPUTS = None
LAST_PINE_PATH = None


def print_banner():
    """Print application banner."""
    print("""
+======================================================================+
|              PINE SCRIPT ML INDICATOR OPTIMIZER                      |
|         Bayesian Optimization with Walk-Forward Validation           |
+======================================================================+
    """)


def print_step(step_num: int, total: int, message: str):
    """Print formatted step message."""
    print(f"\n[{step_num}/{total}] {message}")
    print("-" * 60)


def main():
    """Main entry point."""
    global LAST_RESULT, LAST_OUTPUTS, LAST_PINE_PATH
    LAST_RESULT = None
    LAST_OUTPUTS = None
    LAST_PINE_PATH = None

    enable_screen_log()

    parser = argparse.ArgumentParser(
        description='Optimize Pine Script indicator parameters using ML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python optimize_indicator.py EyeX_MFV_v5.pine
    python optimize_indicator.py indicator.pine --max-trials 200 --timeout 600
    python optimize_indicator.py indicator.pine --symbols BTCUSDT,ETHUSDT
        """
    )
    
    parser.add_argument(
        'pine_script',
        help='Path to Pine Script file to optimize'
    )
    parser.add_argument(
        '--max-trials',
        type=int,
        default=None,
        help='Maximum optimization trials (default: unlimited, uses timeout)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Maximum time in seconds (default: 300 = 5 minutes). Also supports --timeout-minutes.'
    )
    parser.add_argument(
        '--timeout-minutes',
        type=float,
        default=None,
        help='Maximum time in minutes (overrides --timeout if specified)'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated list of symbols to use (default: all available)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='Timeframe/interval (e.g., 1h, 4h, 1d) or comma-separated list (e.g., 1h,4h,1d). Default: 1h'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download of historical data'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--min-runtime-seconds',
        type=int,
        default=None,
        help='Minimum runtime before early-stop checks (default: optimizer setting)'
    )
    parser.add_argument(
        '--stall-seconds',
        type=int,
        default=None,
        help='Stop if no improvement for this many seconds (default: optimizer setting)'
    )
    parser.add_argument(
        '--improvement-rate-floor',
        type=float,
        default=None,
        help='Minimum improvement rate to continue (default: optimizer setting)'
    )
    parser.add_argument(
        '--holdout-ratio',
        type=float,
        default=0.2,
        help='Fraction of data reserved for lockbox evaluation (0 disables). Default: 0.2'
    )
    parser.add_argument(
        '--holdout-gap-bars',
        type=int,
        default=None,
        help='Purge gap between optimization and holdout in bars (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Handle timeout-minutes override
    if args.timeout_minutes is not None:
        args.timeout = int(args.timeout_minutes * 60)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.holdout_ratio < 0 or args.holdout_ratio >= 0.9:
        print("Error: --holdout-ratio must be between 0 and 0.9.")
        sys.exit(1)
    
    # Validate input file
    pine_path = Path(args.pine_script)
    if not pine_path.exists():
        print(f"Error: File not found: {pine_path}")
        sys.exit(1)
    
    if not pine_path.suffix == '.pine':
        print(f"Warning: File does not have .pine extension: {pine_path}")
    
    LAST_PINE_PATH = pine_path

    print_banner()
    print(f"Optimizing: {pine_path.name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize data manager
    dm = DataManager()
    
    # Parse intervals (support comma-separated)
    intervals_input = args.interval.split(',')
    intervals = [i.strip() for i in intervals_input]
    
    # Ask user if they want to use multiple timeframes for the same symbol
    if len(intervals) > 1:
        print("\n" + "-" * 70)
        print("  MULTI-TIMEFRAME OPTION DETECTED")
        print("-" * 70)
        print(f"  You specified {len(intervals)} timeframes: {', '.join(intervals)}")
        print("\n  Options:")
        print("    1. Use multiple timeframes for the same symbol (recommended for comparison)")
        print("       Example: BTCUSDT @ 1h, BTCUSDT @ 4h, BTCUSDT @ 1d")
        print("    2. Use single timeframe (use first interval only)")
        print(f"       Example: All symbols @ {intervals[0]} only")
        print()
        
        while True:
            choice = input("  Choose option (1 or 2) [1]: ").strip()
            if not choice:
                choice = "1"
            if choice in ['1', '2']:
                break
            print("  [ERROR] Please enter 1 or 2")
        
        if choice == '2':
            # Use only first interval
            intervals = [intervals[0]]
            print(f"  Using single timeframe: {intervals[0]}")
        else:
            print(f"  Using multiple timeframes: {', '.join(intervals)}")
            print("  Note: Each symbol will be evaluated at each timeframe separately")
    
    # Determine symbols to use
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        # Ensure USDT suffix
        symbols = [s if s.endswith('USDT') else s + 'USDT' for s in symbols]
    else:
        # Use all available symbols for the first interval (or union of all if multiple)
        if len(intervals) == 1:
            symbols = dm.get_available_symbols(intervals[0])
        else:
            # Union of symbols across all intervals
            all_symbols = set()
            for interval in intervals:
                all_symbols.update(dm.get_available_symbols(interval))
            symbols = sorted(list(all_symbols))
        if not symbols:
            # Fall back to defaults if no data exists
            symbols = SYMBOLS
    
    total_steps = 5
    
    try:
        # Step 1: Load/Download Historical Data
        if len(intervals) == 1:
            print_step(1, total_steps, f"Loading historical data ({len(symbols)} symbols @ {intervals[0]})...")
        else:
            print_step(1, total_steps, f"Loading historical data ({len(symbols)} symbols @ {len(intervals)} timeframes: {', '.join(intervals)})...")
        
        # Check which symbols need downloading
        missing = {}
        for symbol in symbols:
            for interval in intervals:
                if not dm.symbol_exists(symbol, interval):
                    if symbol not in missing:
                        missing[symbol] = []
                    missing[symbol].append(interval)
        
        if missing or args.force_download:
            print(f"   Downloading: {len(missing)} symbols" if missing else "   Downloading: all (forced)")
            for symbol in (symbols if args.force_download else list(missing.keys())):
                for interval in (intervals if args.force_download else missing.get(symbol, [])):
                    try:
                        dm.download_symbol(symbol, interval, force=args.force_download)
                    except Exception as e:
                        print(f"   [WARN] Failed to download {symbol} @ {interval}: {e}")
        
        # Load all data - support multi-timeframe structure
        if len(intervals) == 1:
            # Single timeframe: {symbol: DataFrame}
            data = {}
            for symbol in symbols:
                if dm.symbol_exists(symbol, intervals[0]):
                    try:
                        data[symbol] = dm.load_symbol(symbol, intervals[0])
                        print(f"   [OK] {symbol} @ {intervals[0]}: {len(data[symbol]):,} candles")
                    except Exception as e:
                        print(f"   [WARN] Failed to load {symbol}: {e}")
        else:
            # Multi-timeframe: {symbol: {timeframe: DataFrame}}
            data = {}
            for symbol in symbols:
                data[symbol] = {}
                for interval in intervals:
                    if dm.symbol_exists(symbol, interval):
                        try:
                            data[symbol][interval] = dm.load_symbol(symbol, interval)
                            print(f"   [OK] {symbol} @ {interval}: {len(data[symbol][interval]):,} candles")
                        except Exception as e:
                            print(f"   [WARN] Failed to load {symbol} @ {interval}: {e}")
        
        if not data:
            print(f"Error: No historical data available for interval(s) '{args.interval}'.")
            print(f"       Run the interactive mode to download data first.")
            sys.exit(1)
        
        # Step 2: Parse Pine Script
        print_step(2, total_steps, "Parsing Pine Script parameters...")
        
        parse_result = parse_pine_script(str(pine_path))
        
        print(f"   Indicator: {parse_result.indicator_name}")
        print(f"   Version: Pine Script v{parse_result.version}")
        print(f"   Parameters found: {len(parse_result.parameters)}")
        
        for p in parse_result.parameters:
            bounds = f"[{p.min_val}, {p.max_val}]" if p.min_val is not None else "[auto]"
            print(f"      - {p.name}: {p.param_type} = {p.default} {bounds}")
        
        signal_info = parse_result.signal_info
        print(f"   Signal type: {signal_info.signal_type.value}")
        print(f"   Position type: {signal_info.position_type.value}")
        
        if signal_info.buy_conditions:
            print(f"   Buy signals: {signal_info.buy_conditions}")
        if signal_info.sell_conditions:
            print(f"   Sell signals: {signal_info.sell_conditions}")

        optimizable_params = get_optimizable_params(parse_result.parameters)
        if len(optimizable_params) < 2:
            reason = (
                f"Skipping optimization: only {len(optimizable_params)} optimizable "
                f"parameter(s) found. This workflow requires at least 2 to avoid "
                f"trivial or unstable optimizations."
            )
            logger.info(reason)
            print(f"\n[SKIP] {reason}")
            return
        
        # Step 3: Run Optimization
        trials_str = "unlimited" if args.max_trials is None else str(args.max_trials)
        print_step(3, total_steps, f"Running optimization ({trials_str} trials, ~{args.timeout/60:.1f} min)...")
        print(f"   Sampler: TPE (Tree-Parzen Estimator)")
        print(f"   Validation: 5-fold Walk-Forward with 72-bar embargo")
        print(f"   Objective: Profit Factor + Directional Accuracy + Sharpe + Extreme Capture + Consistency")
        print(f"   Lockbox holdout: {args.holdout_ratio:.0%} (gap: {args.holdout_gap_bars if args.holdout_gap_bars is not None else 'auto'} bars)")
        print()
        print(f"   [TIP] Press Q at any time to stop and use current best results")
        print(f"         Watch improvement rate - diminishing returns suggest stopping early")
        print()
        
        optimizer_kwargs = {}
        if args.min_runtime_seconds is not None:
            optimizer_kwargs['min_runtime_seconds'] = args.min_runtime_seconds
        if args.stall_seconds is not None:
            optimizer_kwargs['stall_seconds'] = args.stall_seconds
        if args.improvement_rate_floor is not None:
            optimizer_kwargs['improvement_rate_floor'] = args.improvement_rate_floor

        optimization_result = optimize_indicator(
            parse_result,
            data,
            interval=args.interval,  # Pass full interval string (may be comma-separated)
            max_trials=args.max_trials,
            timeout_seconds=args.timeout,
            strategy=DEFAULT_OPTIMIZATION_STRATEGY,
            holdout_ratio=args.holdout_ratio,
            holdout_gap_bars=args.holdout_gap_bars,
            **optimizer_kwargs
        )
        
        # Step 4: Generate Optimized Pine Script
        print_step(4, total_steps, "Generating optimized indicator...")
        
        outputs = generate_outputs(parse_result, optimization_result, str(pine_path))
        LAST_RESULT = optimization_result
        LAST_OUTPUTS = outputs
        
        print(f"   [OK] Optimized Pine Script: {outputs['pine_script']}")
        print(f"   [OK] Performance Report: {outputs['report']}")
        
        # Step 5: Print Summary
        print_step(5, total_steps, "Optimization Summary")
        
        metrics = optimization_result.best_metrics
        
        print(f"""
+======================================================================+
|  OPTIMIZATION RESULTS                                                |
+======================================================================+
|                                                                      |
|  Peak Forecast Timeframe:  {metrics.forecast_horizon:>3} hours                                |
|                                                                      |
|  Performance Improvement:                                            |
|    * vs Original Config: {optimization_result.improvement_pf:>+6.1f}% profit factor               |
|    * vs Random Baseline: {metrics.improvement_over_random:>+6.1f}%                               |
|                                                                      |
|  Key Metrics:                                                        |
|    * Profit Factor:       {metrics.profit_factor:>6.2f}                                  |
|    * Win Rate:            {metrics.win_rate*100:>5.1f}%                                  |
|    * Directional Accuracy:{metrics.directional_accuracy*100:>5.1f}%                                  |
|    * Extreme Capture:     {metrics.tail_capture_rate*100:>5.1f}%                                  |
|    * Consistency Score:   {metrics.consistency_score:>6.2f}                                  |
|    * Sharpe Ratio:        {metrics.sharpe_ratio:>6.2f}                                  |
|                                                                      |
|  WHEN TO USE:                                                        |
|    Best during trending conditions, optimal forecast                 |
|    horizon of ~{metrics.forecast_horizon} hours for peak profitability.                         |
|                                                                      |
+======================================================================+
        """)

        if optimization_result.holdout_metrics is not None and optimization_result.holdout_original_metrics is not None:
            holdout_best = optimization_result.holdout_metrics
            holdout_orig = optimization_result.holdout_original_metrics
            print("LOCKBOX (OUT-OF-SAMPLE) RESULTS")
            print(f"  Profit Factor:       {holdout_orig.profit_factor:.2f} -> {holdout_best.profit_factor:.2f}")
            print(f"  Win Rate:            {holdout_orig.win_rate*100:>5.1f}% -> {holdout_best.win_rate*100:>5.1f}%")
            print(f"  Directional Accuracy:{holdout_orig.directional_accuracy*100:>5.1f}% -> {holdout_best.directional_accuracy*100:>5.1f}%")
            print()
        
        def format_val(v):
            if isinstance(v, float):
                if abs(v) < 0.0001:
                    return f"{v:.2e}"
                elif abs(v) < 1:
                    return f"{v:.4f}"
                else:
                    return f"{v:.2f}"
            return str(v)
        
        print("Parameter Changes:")
        changes_found = False
        for name, new_val in optimization_result.best_params.items():
            orig_val = optimization_result.original_params.get(name)
            if orig_val != new_val:
                print(f"   {name}: {format_val(orig_val)} -> {format_val(new_val)}")
                changes_found = True
        
        if not changes_found:
            print("   (No parameter changes - original values were optimal)")
        
        print(f"\nOutput files created in: {Path.cwd()}")
        print(f"   - {outputs['pine_script']}")
        print(f"   - {outputs['report']}")
        
        print("\n[SUCCESS] Optimization complete!")
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during optimization: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


def optimize_all_in_directory(directory: str = ".", **kwargs):
    """
    Optimize all .pine files in a directory.
    
    Args:
        directory: Directory to search for .pine files
        **kwargs: Arguments to pass to optimization
    """
    dir_path = Path(directory)
    pine_files = list(dir_path.glob("*.pine"))
    
    # Exclude already optimized files
    pine_files = [f for f in pine_files if not f.name.startswith('optimised_')]
    
    if not pine_files:
        print(f"No Pine Script files found in {directory}")
        return
    
    print(f"Found {len(pine_files)} Pine Script files to optimize:")
    for f in pine_files:
        print(f"   - {f.name}")
    
    print()
    
    for i, pine_file in enumerate(pine_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(pine_files)}: {pine_file.name}")
        print('='*60)
        
        try:
            # Call main with modified sys.argv
            original_argv = sys.argv
            sys.argv = ['optimize_indicator.py', str(pine_file)]
            
            # Add kwargs as command line args
            if kwargs.get('max_trials'):
                sys.argv.extend(['--max-trials', str(kwargs['max_trials'])])
            if kwargs.get('timeout'):
                sys.argv.extend(['--timeout', str(kwargs['timeout'])])
            
            main()
            
            sys.argv = original_argv
            
        except Exception as e:
            print(f"Error optimizing {pine_file.name}: {e}")
            continue


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - optimize all .pine files in current directory
        print("No file specified. Looking for .pine files in current directory...")
        optimize_all_in_directory()
    else:
        main()
