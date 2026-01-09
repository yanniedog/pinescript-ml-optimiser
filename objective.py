"""
Shared objective scoring for optimization and reporting.
"""

from typing import Any


def calculate_objective_score(
    metrics: Any,
    min_trades: int = 10,
    min_trades_penalty: int = 50
) -> float:
    """Calculate the objective score from metrics."""
    if metrics.total_trades < min_trades:
        return 0.0

    pf_score = min(metrics.profit_factor, 5.0) / 5.0
    acc_score = max(0, min(1, (metrics.directional_accuracy - 0.5) * 2))
    sharpe_score = min(max(metrics.sharpe_ratio, 0), 3.0) / 3.0
    win_score = metrics.win_rate
    tail_score = max(0.0, min(1.0, metrics.tail_capture_rate))
    consistency_score = max(0.0, min(1.0, metrics.consistency_score))
    objective = (
        0.263 * pf_score +
        0.211 * acc_score +
        0.158 * sharpe_score +
        0.105 * win_score +
        0.158 * tail_score +
        0.105 * consistency_score
    )

    if metrics.total_trades < min_trades_penalty:
        objective *= metrics.total_trades / min_trades_penalty

    return objective
