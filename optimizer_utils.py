"""
Utility functions for the optimizer.
"""

import logging
from typing import Dict, Any, List, Optional
from pine_parser import Parameter
from backtester import BacktestMetrics

# Configure logging
logger = logging.getLogger(__name__)

_METRIC_DEFS = {
    "objective_best": {"label": "Objective (avg)"},
    "objective_delta": {"label": "Objective Delta"},
    "objective_overall": {"label": "Objective (overall)"},
    "profit_factor": {"label": "Profit Factor"},
    "win_rate": {"label": "Win Rate"},
    "directional_accuracy": {"label": "Directional Accuracy"},
    "mcc": {"label": "MCC"},
    "roc_auc": {"label": "ROC AUC"},
    "classification_samples": {"label": "Class Samples"},
    "sharpe_ratio": {"label": "Sharpe Ratio"},
    "max_drawdown": {"label": "Max Drawdown"},
    "total_return": {"label": "Total Return"},
    "avg_return": {"label": "Avg Return"},
    "total_trades": {"label": "Total Trades"},
    "winning_trades": {"label": "Winning Trades"},
    "losing_trades": {"label": "Losing Trades"},
    "avg_holding_bars": {"label": "Avg Holding Bars"},
    "forecast_horizon": {"label": "Forecast Horizon"},
    "improvement_over_random": {"label": "Improvement vs Random"},
    "tail_capture_rate": {"label": "Tail Capture Rate"},
    "consistency_score": {"label": "Consistency Score"},
    "total_trials": {"label": "Total Trials"},
    "trials_per_second": {"label": "Trials/sec"},
}
_METRIC_KEYS = list(_METRIC_DEFS.keys())


def _metrics_from_backtest(metrics: Optional[BacktestMetrics]) -> Dict[str, float]:
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


def _metric_label(metric_key: str) -> str:
    return _METRIC_DEFS.get(metric_key, {}).get("label", metric_key)


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 0.0001:
            return f"{value:.2e}"
        return f"{value:.4g}"
    return str(value)


def _format_params(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return "N/A"
    parts = []
    for name in sorted(params.keys()):
        parts.append(f"{name}={_format_param_value(params[name])}")
    return ", ".join(parts)


def _last_non_none(values: List[Optional[float]]) -> Optional[float]:
    for val in reversed(values):
        if val is not None:
            return val
    return None


def _compute_rate_series(
    elapsed_vals: List[float],
    metric_vals: List[Optional[float]],
    baseline_value: Optional[float]
) -> List[Optional[float]]:
    baseline = baseline_value if baseline_value is not None else 0.0
    rates = []
    for elapsed, val in zip(elapsed_vals, metric_vals):
        if val is None or elapsed is None:
            rates.append(None)
            continue
        if elapsed <= 0:
            rates.append(0.0)
            continue
        rates.append((val - baseline) / elapsed)
    return rates


def get_optimizable_params(parameters: List[Parameter]) -> List[Parameter]:
    """Filter parameters to only those worth optimizing."""
    skip_keywords = ['show', 'display', 'color', 'style', 'size', 'line']
    optimizable = []
    for p in parameters:
        name_lower = p.name.lower()
        title_lower = p.title.lower()

        # Skip visual/display parameters
        if any(kw in name_lower or kw in title_lower for kw in skip_keywords):
            continue

        # Skip bool parameters that are likely display toggles
        if p.param_type == 'bool' and any(kw in name_lower or kw in title_lower for kw in skip_keywords):
            continue

        optimizable.append(p)

    logger.info(f"Found {len(optimizable)} optimizable parameters out of {len(parameters)}")
    return optimizable
