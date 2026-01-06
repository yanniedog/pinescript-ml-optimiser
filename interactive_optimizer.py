#!/usr/bin/env python3
"""
Interactive runner for Pine Script indicator optimization.
Lists available .pine files and lets user select which one to optimize.
"""

import sys
import os
from pathlib import Path

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from optimize_indicator import main as optimize_main
import argparse


def get_pine_files():
    """Get all .pine files in the current directory, excluding optimized ones."""
    current_dir = Path.cwd()
    pine_files = list(current_dir.glob("*.pine"))
    
    # Exclude already optimized files
    pine_files = [f for f in pine_files if not f.name.startswith('optimised_')]
    
    return sorted(pine_files)


def display_menu(pine_files):
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
                print("Exiting...")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(pine_files):
                return pine_files[choice_num - 1]
            else:
                print(f"[ERROR] Please enter a number between 1 and {len(pine_files)}")
        except ValueError:
            print("[ERROR] Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def get_optimization_options():
    """Get optional optimization parameters from user (time and trials already set)."""
    options = {}
    
    print("\n" + "-"*70)
    print("  Additional Settings (press Enter for defaults)")
    print("  Note: Time and trials are already configured based on your input")
    print("-"*70)
    
    # Symbols
    symbols_input = input("  Symbols (comma-separated, or Enter for all) [all]: ").strip()
    if symbols_input:
        options['symbols'] = symbols_input
    
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
    
    if 'max_trials' in options:
        args.extend(['--max-trials', str(options['max_trials'])])
    
    if 'timeout' in options:
        args.extend(['--timeout', str(options['timeout'])])
    
    if 'symbols' in options:
        args.extend(['--symbols', options['symbols']])
    
    if options.get('force_download'):
        args.append('--force-download')
    
    if options.get('verbose'):
        args.append('--verbose')
    
    return args


def main():
    """Main interactive runner."""
    print("""
======================================================================
         PINE SCRIPT ML INDICATOR OPTIMIZER - Interactive Mode        
         Bayesian Optimization with Walk-Forward Validation           
======================================================================
    """)
    
    # Get available Pine Script files
    pine_files = get_pine_files()
    
    if not pine_files:
        return
    
    # Display menu
    display_menu(pine_files)
    
    # Get user selection
    selected_file = get_user_choice(pine_files)
    
    print(f"\n[OK] Selected: {selected_file.name}")
    
    # Always ask for timeout in minutes (most commonly changed setting)
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
    
    # Auto-calculate trials based on time available
    # Estimate: ~30 trials per minute is typical (varies by indicator complexity)
    # We set a higher cap to ensure timeout is the limiting factor
    TRIALS_PER_MINUTE = 30
    auto_trials = max(50, int(timeout_minutes * TRIALS_PER_MINUTE * 1.5))  # 1.5x buffer
    options['max_trials'] = auto_trials
    
    print(f"\nOptimization configured:")
    print(f"  - Time limit: {timeout_minutes:.1f} minute(s)")
    print(f"  - Max trials: {auto_trials} (auto-calculated, time is the limiting factor)")
    print(f"  - Press Ctrl-Q anytime to stop early and use current best results")
    
    # Ask if user wants to customize other options
    customize = input("\nCustomize other settings (symbols, etc.)? [n]: ").strip().lower()
    
    if customize in ['y', 'yes']:
        extra_options = get_optimization_options()
        # Remove timeout and trials from extra since we already set them optimally
        extra_options.pop('timeout', None)
        extra_options.pop('max_trials', None)
        options.update(extra_options)
    else:
        print("Using defaults: all symbols")
    
    # Build arguments and call optimize_indicator
    original_argv = sys.argv
    try:
        sys.argv = ['interactive_optimizer.py'] + build_args(selected_file, options)
        optimize_main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error during optimization: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)

