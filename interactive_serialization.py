"""
Serialization utilities for the interactive optimizer.
"""

import json
from pathlib import Path
from objective import calculate_objective_score as objective_score


def calculate_objective_score(metrics) -> float:
    """Calculate overall objective score for ranking."""
    return objective_score(metrics)


def _serialize_metrics(metrics):
    if metrics is None:
        return {}
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
        if result.original_metrics is None:
            return 0.0
        baseline = calculate_objective_score(result.original_metrics)
    return baseline


def _is_improved_result(result) -> bool:
    if result is None:
        return False
    if result.best_metrics is None:
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
