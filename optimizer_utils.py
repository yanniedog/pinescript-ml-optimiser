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
    "objective_best": {"label": "MCC (avg across symbols)"},
    "objective_delta": {"label": "MCC Delta vs Original"},
    "objective_overall": {"label": "MCC (aggregated)"},
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


def _lttb_downsample(
    x_vals: List[float],
    y_vals: List[Optional[float]],
    target_points: int = 500
) -> tuple:
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm.
    Preserves visual shape while reducing point count efficiently.
    O(n) complexity, designed for time series visualization.
    
    Returns (downsampled_x, downsampled_y, downsampled_indices)
    """
    n = len(x_vals)
    if n <= target_points or target_points < 3:
        return x_vals, y_vals, list(range(n))
    
    # Replace None values with interpolated or 0 for algorithm
    y_clean = []
    last_valid = 0.0
    for v in y_vals:
        if v is None:
            y_clean.append(last_valid)
        else:
            y_clean.append(v)
            last_valid = v
    
    # Always keep first and last points
    sampled_indices = [0]
    
    # Bucket size (excluding first and last)
    bucket_size = (n - 2) / (target_points - 2)
    
    a = 0  # Previous selected point index
    
    for i in range(target_points - 2):
        # Calculate bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)
        
        # Calculate average point for next bucket (for triangle area)
        next_bucket_start = bucket_end
        next_bucket_end = int((i + 3) * bucket_size) + 1
        next_bucket_end = min(next_bucket_end, n)
        
        if next_bucket_start >= n:
            next_bucket_start = n - 1
        if next_bucket_end > n:
            next_bucket_end = n
            
        # Average of next bucket
        avg_x = 0.0
        avg_y = 0.0
        count = max(1, next_bucket_end - next_bucket_start)
        for j in range(next_bucket_start, next_bucket_end):
            avg_x += x_vals[j]
            avg_y += y_clean[j]
        avg_x /= count
        avg_y /= count
        
        # Find point in current bucket with max triangle area
        max_area = -1.0
        max_idx = bucket_start
        
        point_a_x = x_vals[a]
        point_a_y = y_clean[a]
        
        for j in range(bucket_start, bucket_end):
            # Triangle area (simplified, sign doesn't matter)
            area = abs(
                (point_a_x - avg_x) * (y_clean[j] - point_a_y) -
                (point_a_x - x_vals[j]) * (avg_y - point_a_y)
            )
            if area > max_area:
                max_area = area
                max_idx = j
        
        sampled_indices.append(max_idx)
        a = max_idx
    
    # Always include last point
    sampled_indices.append(n - 1)
    
    # Build output arrays
    out_x = [x_vals[i] for i in sampled_indices]
    out_y = [y_vals[i] for i in sampled_indices]
    
    return out_x, out_y, sampled_indices


def _downsample_series_for_plot(
    series_data: Dict[str, Any],
    metric_keys: List[str],
    target_points: int = 500
) -> Dict[str, Any]:
    """
    Downsample a full series for efficient plotting.
    Preserves visual fidelity using LTTB algorithm.
    Returns a new dict with downsampled arrays.
    """
    x_vals = series_data.get("x", [])
    if len(x_vals) <= target_points:
        return series_data  # No downsampling needed
    
    # Use objective_delta as primary for downsampling decisions
    primary_y = series_data.get("metrics", {}).get("objective_delta", x_vals)
    
    _, _, indices = _lttb_downsample(x_vals, primary_y, target_points)
    
    # Build downsampled result
    result = {
        "x": [x_vals[i] for i in indices],
        "trials": [series_data.get("trials", [])[i] for i in indices if i < len(series_data.get("trials", []))],
        "params": [series_data.get("params", [])[i] for i in indices if i < len(series_data.get("params", []))],
        "metrics": {}
    }
    
    for key in metric_keys:
        metric_vals = series_data.get("metrics", {}).get(key, [])
        if metric_vals:
            result["metrics"][key] = [metric_vals[i] for i in indices if i < len(metric_vals)]
    
    return result


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
