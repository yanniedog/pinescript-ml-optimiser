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
from optimizer import optimize_indicator
from output_generator import generate_outputs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        help='Timeframe/interval (e.g., 1h, 4h, 1d). Default: 1h'
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
    
    args = parser.parse_args()
    
    # Handle timeout-minutes override
    if args.timeout_minutes is not None:
        args.timeout = int(args.timeout_minutes * 60)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    pine_path = Path(args.pine_script)
    if not pine_path.exists():
        print(f"Error: File not found: {pine_path}")
        sys.exit(1)
    
    if not pine_path.suffix == '.pine':
        print(f"Warning: File does not have .pine extension: {pine_path}")
    
    print_banner()
    print(f"Optimizing: {pine_path.name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize data manager
    dm = DataManager()
    interval = args.interval
    
    # Determine symbols to use
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
        # Ensure USDT suffix
        symbols = [s if s.endswith('USDT') else s + 'USDT' for s in symbols]
    else:
        # Use all available symbols for this interval
        symbols = dm.get_available_symbols(interval)
        if not symbols:
            # Fall back to defaults if no data exists
            symbols = SYMBOLS
    
    total_steps = 5
    
    try:
        # Step 1: Load/Download Historical Data
        print_step(1, total_steps, f"Loading historical data ({len(symbols)} symbols @ {interval})...")
        
        # Check which symbols need downloading
        missing = [s for s in symbols if not dm.symbol_exists(s, interval)]
        if missing or args.force_download:
            print(f"   Downloading: {missing if missing else 'all (forced)'}")
            for symbol in (symbols if args.force_download else missing):
                try:
                    dm.download_symbol(symbol, interval, force=args.force_download)
                except Exception as e:
                    print(f"   [WARN] Failed to download {symbol}: {e}")
        
        # Load all data
        data = {}
        for symbol in symbols:
            if dm.symbol_exists(symbol, interval):
                try:
                    data[symbol] = dm.load_symbol(symbol, interval)
                    print(f"   [OK] {symbol}: {len(data[symbol]):,} candles")
                except Exception as e:
                    print(f"   [WARN] Failed to load {symbol}: {e}")
        
        if not data:
            print(f"Error: No historical data available for interval '{interval}'.")
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
        
        # Step 3: Run Optimization
        trials_str = "unlimited" if args.max_trials is None else str(args.max_trials)
        print_step(3, total_steps, f"Running optimization ({trials_str} trials, ~{args.timeout/60:.1f} min)...")
        print(f"   Sampler: TPE (Tree-Parzen Estimator)")
        print(f"   Validation: 5-fold Walk-Forward with 72-bar embargo")
        print(f"   Objective: Profit Factor + Directional Accuracy + Sharpe")
        print()
        print(f"   [TIP] Press Q at any time to stop and use current best results")
        print(f"         Watch improvement rate - diminishing returns suggest stopping early")
        print()
        
        optimization_result = optimize_indicator(
            parse_result,
            data,
            interval=interval,
            max_trials=args.max_trials,
            timeout_seconds=args.timeout
        )
        
        # Step 4: Generate Optimized Pine Script
        print_step(4, total_steps, "Generating optimized indicator...")
        
        outputs = generate_outputs(parse_result, optimization_result, str(pine_path))
        
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
|    * Sharpe Ratio:        {metrics.sharpe_ratio:>6.2f}                                  |
|                                                                      |
|  WHEN TO USE:                                                        |
|    Best during trending conditions, optimal forecast                 |
|    horizon of ~{metrics.forecast_horizon} hours for peak profitability.                         |
|                                                                      |
+======================================================================+
        """)
        
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

