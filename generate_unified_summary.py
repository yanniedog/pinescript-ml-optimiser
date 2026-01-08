#!/usr/bin/env python3
"""
Rebuild a unified optimization summary from existing report files.
"""

import argparse
import re
from pathlib import Path

from backtester import BacktestMetrics
from objective import calculate_objective_score
from interactive_optimizer import write_unified_report


METRIC_MAP = {
    "Profit Factor": "profit_factor",
    "Win Rate": "win_rate",
    "Directional Accuracy": "directional_accuracy",
    "Extreme Move Capture": "tail_capture_rate",
    "Consistency Score": "consistency_score",
    "Sharpe Ratio": "sharpe_ratio",
    "Total Trades": "total_trades",
    "Max Drawdown": "max_drawdown",
}

RATE_METRICS = {"Win Rate", "Directional Accuracy", "Extreme Move Capture"}
INT_METRICS = {"Total Trades"}


def _parse_numeric(metric_name: str, raw: str):
    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return None
    raw = raw.replace(",", "")
    if metric_name in INT_METRICS:
        try:
            return int(float(raw))
        except ValueError:
            return None
    if raw.endswith("%"):
        raw = raw[:-1]
    try:
        value = float(raw)
    except ValueError:
        return None
    if metric_name in RATE_METRICS:
        return value / 100.0
    return value


def _build_metrics(values: dict) -> BacktestMetrics:
    metrics = BacktestMetrics()
    if not values:
        return metrics
    metrics.total_trades = int(values.get("total_trades") or 0)
    metrics.profit_factor = float(values.get("profit_factor") or 0.0)
    metrics.win_rate = float(values.get("win_rate") or 0.0)
    metrics.directional_accuracy = float(values.get("directional_accuracy") or 0.0)
    metrics.tail_capture_rate = float(values.get("tail_capture_rate") or 0.0)
    metrics.consistency_score = float(values.get("consistency_score") or 0.0)
    metrics.sharpe_ratio = float(values.get("sharpe_ratio") or 0.0)
    metrics.max_drawdown = float(values.get("max_drawdown") or 0.0)
    return metrics


def _parse_metrics_table(lines):
    metrics = {"original": {}, "optimized": {}}
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped == "PERFORMANCE COMPARISON":
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped:
            continue
        if stripped.startswith("Overall Performance") or stripped.startswith("MULTI-TIMEFRAME"):
            break
        if stripped.startswith("Metric") or set(stripped) == {"-"}:
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 4:
            continue
        name, orig_val, opt_val = parts[0], parts[1], parts[2]
        if name not in METRIC_MAP:
            continue
        key = METRIC_MAP[name]
        metrics["original"][key] = _parse_numeric(name, orig_val)
        metrics["optimized"][key] = _parse_numeric(name, opt_val)
    return metrics


def _parse_config(lines):
    config = {}
    in_config = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Optimization config:":
            in_config = True
            continue
        if not in_config:
            continue
        if not stripped:
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def _parse_opt_data(lines):
    data = {}
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Optimization data:":
            in_section = True
            continue
        if not in_section:
            continue
        if not stripped:
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _parse_summary_values(lines):
    result = {"n_trials": None, "optimization_time": None}
    for line in lines:
        if "Trials completed:" in line:
            try:
                result["n_trials"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                result["n_trials"] = None
        if "Total time taken:" in line:
            match = re.search(r"\(([\d.]+)\s+seconds\)", line)
            if match:
                result["optimization_time"] = float(match.group(1))
    return result


def _serialize_metrics(metrics: BacktestMetrics) -> dict:
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
        "forecast_horizon": metrics.forecast_horizon,
        "improvement_over_random": metrics.improvement_over_random,
        "tail_capture_rate": metrics.tail_capture_rate,
        "consistency_score": metrics.consistency_score,
    }


def parse_report(report_path: Path) -> dict:
    lines = report_path.read_text(encoding="utf-8").splitlines()

    indicator_name = None
    for line in lines:
        if line.startswith("ML OPTIMIZATION REPORT:"):
            indicator_name = line.split(":", 1)[1].strip()
            break
    if not indicator_name:
        indicator_name = report_path.stem

    metrics_values = _parse_metrics_table(lines)
    original_metrics = _build_metrics(metrics_values.get("original"))
    best_metrics = _build_metrics(metrics_values.get("optimized"))

    summary = _parse_summary_values(lines)
    config_raw = _parse_config(lines)
    opt_data = _parse_opt_data(lines)

    report_stem = report_path.stem
    base_name = report_stem
    if base_name.startswith("optimised_"):
        base_name = base_name[len("optimised_"):]
    if base_name.endswith("_report"):
        base_name = base_name[:-len("_report")]

    file_name = f"{base_name}.pine"
    output_pine = str(Path("optimized_outputs") / "pine" / f"optimised_{base_name}.pine")

    config = {
        "strategy": config_raw.get("Strategy", "tpe"),
        "sampler": config_raw.get("Sampler", "tpe"),
        "timeout_seconds": config_raw.get("Timeout (sec)", None),
        "max_trials": config_raw.get("Max trials", None),
        "early_stop_patience": config_raw.get("Early stop patience", None),
        "min_runtime_seconds": config_raw.get("Min runtime (sec)", None),
        "stall_seconds": config_raw.get("Stall timeout (sec)", None),
        "improvement_rate_floor": config_raw.get("Rate floor (%/s)", None),
        "improvement_rate_window": config_raw.get("Rate window", None),
        "backtester_overrides": config_raw.get("Backtester overrides", None),
    }

    objective_best = calculate_objective_score(best_metrics)
    baseline_objective = calculate_objective_score(original_metrics)

    return {
        "indicator_name": indicator_name,
        "file_name": file_name,
        "output_pine": output_pine,
        "output_report": str(report_path),
        "optimization_time": summary.get("optimization_time"),
        "n_trials": summary.get("n_trials"),
        "objective_best": objective_best,
        "baseline_objective": baseline_objective,
        "best_metrics": _serialize_metrics(best_metrics),
        "original_metrics": _serialize_metrics(original_metrics),
        "original_params": {},
        "best_params": {},
        "per_symbol_metrics": {},
        "data_usage_info": {},
        "datasets_used": [],
        "interval": opt_data.get("Timeframe"),
        "config": config,
    }


def build_summary(reports_dir: Path, output_dir: Path):
    report_files = sorted(reports_dir.glob("optimised_*_report.txt"))
    if not report_files:
        raise FileNotFoundError(f"No report files found in {reports_dir}")

    results = []
    intervals = set()
    for report_path in report_files:
        result = parse_report(report_path)
        interval = result.get("interval")
        if interval:
            intervals.add(interval)
        results.append(result)

    interval_display = None
    if len(intervals) == 1:
        interval_display = next(iter(intervals))
    elif intervals:
        interval_display = "mixed"

    run_info = {
        "source": "report_files",
        "report_dir": str(reports_dir),
        "total_indicators": len(results),
    }
    if interval_display:
        run_info["interval"] = interval_display

    write_unified_report(
        output_dir / "unified_optimization_report.txt",
        output_dir / "unified_optimization_results.json",
        run_info,
        results,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate a unified report from existing outputs.")
    parser.add_argument(
        "--reports-dir",
        default="optimized_outputs/reports",
        help="Directory containing optimised_*_report.txt files",
    )
    parser.add_argument(
        "--output-dir",
        default="optimized_outputs/summary",
        help="Directory to write unified_optimization_report.txt and .json",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    build_summary(reports_dir, output_dir)
    print(f"[OK] Unified summary written to {output_dir}")


if __name__ == "__main__":
    main()
