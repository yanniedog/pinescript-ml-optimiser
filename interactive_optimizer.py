#!/usr/bin/env python3
"""
Interactive runner for Pine Script indicator optimization.
Includes data management for downloading any crypto symbol at any timeframe.

Refactored into modules:
- interactive_types.py
- interactive_ui.py
- interactive_config.py
- interactive_cache.py
- interactive_serialization.py
- interactive_reports.py
- interactive_data.py
- interactive_workflows.py
"""

import sys
import builtins
from typing import Optional, Callable

# Override input globally before imports that might use it
from interactive_ui import _input_with_go_back
builtins.input = _input_with_go_back

from screen_log import enable_screen_log
from data_manager import DataManager, VALID_INTERVALS, INTERVAL_NAMES
from interactive_types import GoBack
from interactive_ui import handle_go_back, ask_yes_no, get_pine_files, display_pine_menu, get_user_choice, split_choice_input, find_pine_directories, display_indicator_catalog, choose_indicator_directory
from interactive_config import configure_trial_controls, TRIAL_CONTROL_OPTIONS, apply_trial_overrides, _get_trial_overrides
from interactive_data import download_new_data, display_data_status, get_optimization_options, select_timeframe, select_datasets_for_matrix, get_eligible_indicators, maybe_generate_all_indicators, select_indicator_subset
from interactive_workflows import run_optimization, run_batch_optimization, run_matrix_optimization, backup_previous_outputs, sort_rankings, build_args, BACKUP_DONE
from interactive_serialization import calculate_objective_score, _is_improved_result, _serialize_metrics, _safe_tag, _baseline_objective, _serialize_params, _json_safe_value, _serialize_fold_details, _serialize_per_symbol_metrics, _serialize_data_usage_info
from interactive_cache import _collect_cached_combos, _restore_cached_outputs, _combo_key_from_row, _load_matrix_rows
from interactive_reports import write_matrix_reports, write_unified_report

# Ensure we can import our modules
# sys.path.insert(0, str(Path(__file__).parent)) # Already in root

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
            user_input = input("  Select option: ").strip()
            # Handle empty input
            if not user_input:
                print("\n  [ERROR] Please enter a valid option (1-6 or Q).")
                continue
            
            choice = user_input.lower()
            
            # Debug: Show what was received (can be removed later)
            # print(f"  [DEBUG] Received input: '{user_input}' -> choice: '{choice}'")
            
        except GoBack:
            print("  [INFO] Already at the main menu.")
            continue
        except (EOFError, KeyboardInterrupt) as e:
            print("\n\nExiting...")
            break
        except Exception as e:
            # If this is actually an EOFError that wasn't caught by the specific handler, handle it
            if isinstance(e, (EOFError, KeyboardInterrupt)):
                print("\n\nExiting...")
                break
            print(f"\n  [ERROR] Unexpected error reading input: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        try:
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
                print("\n  [INFO] Starting matrix optimization...")
                run_matrix_optimization(dm)
            elif choice == '6':
                configure_trial_controls()
            elif choice == 'q':
                print("\nGoodbye!")
                break
            else:
                print(f"\n  [ERROR] Invalid option '{choice}'. Please enter 1, 2, 3, 4, 5, 6, or Q.")
                print(f"  (You entered: '{user_input}' -> processed as: '{choice}')")
        except GoBack:
            # GoBack should be handled by decorators, but catch it here just in case
            print("\n  [INFO] Returning to main menu.")
        except Exception as e:
            print(f"\n  [ERROR] Error executing option '{choice}': {e}")
            import traceback
            traceback.print_exc()


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
