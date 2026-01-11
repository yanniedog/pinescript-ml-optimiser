"""
Shared objective scoring for optimization and reporting.
"""

from typing import Any
import math


def calculate_objective_score(
    metrics: Any,
    min_classification_samples: int = 50,
    min_roc_auc: float = 0.52
) -> float:
    """Calculate the objective score from metrics."""
    if metrics.classification_samples < min_classification_samples:
        return 0.0
    if metrics.roc_auc < min_roc_auc:
        return 0.0
    if not math.isfinite(metrics.mcc):
        return 0.0
    return float(metrics.mcc)
