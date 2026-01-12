"""
Report generation utilities for the interactive optimizer.
"""

import csv
import json
import time
from pathlib import Path
from interactive_serialization import _json_safe_value, _serialize_params


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
        trials_str = f"{row['n_trials']}" if row.get('n_trials') is not None else "N/A"
        time_str = f"{row['optimization_time']:.1f}s" if row.get('optimization_time') is not None else "N/A"
        lines.append(f"  Trials: {trials_str} | Time: {time_str}")
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
