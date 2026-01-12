"""
Optimization workflows for the interactive optimizer.
"""

import sys
import time
import shutil
import json
import zipfile
from pathlib import Path
from data_manager import DataManager
import optimize_indicator as optimize_module
from pine_parser import parse_pine_script
from optimizer import optimize_indicator as run_optimizer
from output_generator import generate_outputs

from interactive_ui import (
    handle_go_back,
    get_pine_files,
    display_pine_menu,
    get_user_choice,
    display_indicator_catalog,
    ask_yes_no,
    display_data_status,
    choose_indicator_directory
)
from interactive_config import apply_trial_overrides
from interactive_data import (
    get_optimization_options,
    maybe_generate_all_indicators,
    get_eligible_indicators,
    select_indicator_subset,
    select_datasets_for_matrix,
    download_new_data
)
from interactive_serialization import (
    _json_safe_value,
    _is_improved_result,
    calculate_objective_score,
    _serialize_metrics,
    _serialize_per_symbol_metrics,
    _serialize_data_usage_info,
    _safe_tag,
    _baseline_objective,
    _serialize_params
)
from interactive_cache import (
    _collect_cached_combos,
    _restore_cached_outputs
)
from interactive_reports import (
    write_unified_report,
    write_matrix_reports
)

BACKUP_DONE = False


def backup_previous_outputs():
    """Backup previous outputs to a timestamped zip file (one per session)."""
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
    try:
        backup_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_zip_path = backup_root / f"{timestamp}.zip"

    # Collect run info before creating zip
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
    info_json_content = json.dumps(info_payload, indent=2, default=_json_safe_value)

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
    info_txt_content = "\n".join(info_lines)

    # Create the zip file
    try:
        with zipfile.ZipFile(backup_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add backup info files
            zf.writestr("backup_info.json", info_json_content)
            zf.writestr("backup_info.txt", info_txt_content)
            
            # Add all source directories/files
            for path in existing_sources:
                if path.is_dir():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            arcname = str(file_path)
                            zf.write(file_path, arcname)
                elif path.is_file():
                    zf.write(path, str(path))
        
        print(f"[BACKUP] Created backup: {backup_zip_path}")
    except Exception as e:
        print(f"[ERROR] Failed to create backup zip: {e}")
        raise

    BACKUP_DONE = True


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

    if 'n_jobs' in options and options['n_jobs'] is not None:
        args.extend(['--n-jobs', str(options['n_jobs'])])

    if 'fast_evaluation' in options and options['fast_evaluation'] is not None:
        if options['fast_evaluation']:
            args.append('--fast-evaluation')

    if 'symbols' in options:
        args.extend(['--symbols', options['symbols']])
    
    if 'interval' in options:
        args.extend(['--interval', options['interval']])
    
    if options.get('force_download'):
        args.append('--force-download')
    
    if options.get('verbose'):
        args.append('--verbose')
    
    return args


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
        if metrics is None:
            print(f"[WARNING] No metrics available for {pine_file.name}; skipping ranking.")
            continue
        
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
            if result.best_metrics is None or result.original_metrics is None:
                print(f"[WARNING] Missing metrics for {pine_file.name}; skipping result serialization.")
                continue
            indicator_name = (last_path and last_path.stem) or pine_file.stem
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
        budget_input = input("  Choose option [2]: ").strip()
        if not budget_input:
            budget_input = "2"
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
            per_combo_input = input("Minutes per combination? [10]: ").strip()
            if not per_combo_input:
                per_combo_minutes = 10.0
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

    total_combo_entries = list(zip(combos, per_combo_budgets))
    total_combo_count = len(total_combo_entries)
    cached_combos = _collect_cached_combos()
    active_entries = []
    reused_rows = []
    for (pine_file, symbol, interval), combo_timeout in total_combo_entries:
        key = (pine_file.name, symbol, interval)
        cached = cached_combos.get(key)
        if cached:
            row, source_root = cached
            if _restore_cached_outputs(row, source_root):
                row["reused"] = True
                reused_rows.append(row)
                print(f"[CACHE] Reusing cached result for {pine_file.name} {symbol} @ {interval}")
                continue
        active_entries.append(((pine_file, symbol, interval), combo_timeout))

    combos = [entry[0] for entry in active_entries]
    per_combo_budgets = [entry[1] for entry in active_entries]

    if reused_rows and not combos:
        print("\n[INFO] All requested combinations already optimized; no new runs will be executed.")
    elif reused_rows:
        print(f"\n[INFO] {len(reused_rows)} cached combination(s) reused; {len(combos)} new combination(s) will be optimized.")

    results = reused_rows.copy()
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

        if result.best_metrics is None or result.original_metrics is None:
            print(f"[WARNING] Missing metrics for {pine_file.name} {symbol} @ {interval}; skipping result.")
            continue

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
        "combo_total": total_combo_count,
        "combo_run": len(combos),
        "combo_cached": len(reused_rows),
        "combo_new": len(combos),
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
