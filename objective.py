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
    """
    Calculate the objective score from metrics using soft constraints.
    
    Instead of hard cutoffs that return 0.0 below thresholds (which distort
    the optimization landscape), this uses soft penalties that smoothly
    reduce the score as metrics fall below desired thresholds.
    
    Args:
        metrics: BacktestMetrics object with mcc, roc_auc, classification_samples
        min_classification_samples: Target minimum samples (soft penalty below)
        min_roc_auc: Target minimum ROC AUC (soft penalty below)
    
    Returns:
        Objective score (MCC adjusted by soft penalties)
    """
    if not math.isfinite(metrics.mcc):
        return 0.0
    
    # Soft penalty for insufficient samples
    # Scales linearly from 0 to 1 as samples approach min_classification_samples
    sample_penalty = min(1.0, metrics.classification_samples / min_classification_samples)
    
    # Soft penalty for ROC AUC below threshold
    # ROC AUC < 0.5 means worse than random, so penalty = 0
    # ROC AUC between 0.5 and min_roc_auc scales linearly
    # ROC AUC >= min_roc_auc means no penalty
    if metrics.roc_auc < 0.5:
        roc_penalty = 0.0
    elif metrics.roc_auc < min_roc_auc:
        roc_penalty = (metrics.roc_auc - 0.5) / (min_roc_auc - 0.5)
    else:
        roc_penalty = 1.0
    
    return float(metrics.mcc * sample_penalty * roc_penalty)
