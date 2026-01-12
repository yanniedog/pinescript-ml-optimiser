"""
Shared objective scoring for optimization and reporting.
"""

from typing import Any
import math


def calculate_mcc_confidence_interval(mcc: float, n_samples: int, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for MCC using asymptotic approximation.
    
    Args:
        mcc: Matthews Correlation Coefficient
        n_samples: Number of classification samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_samples <= 2:
        return (-1.0, 1.0)  # No confidence with too few samples
    
    # Standard error approximation for MCC
    # Based on: SE(MCC) ≈ sqrt((1 - MCC²) / (n - 2))
    mcc_squared = mcc * mcc
    if mcc_squared >= 1.0:
        se_mcc = 0.0
    else:
        se_mcc = math.sqrt((1.0 - mcc_squared) / (n_samples - 2))
    
    # Z-score for confidence level (1.96 for 95%)
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    else:
        # Approximate z-score for other confidence levels
        z = 1.96
    
    lower = mcc - z * se_mcc
    upper = mcc + z * se_mcc
    
    # Clamp to valid MCC range
    return (max(-1.0, lower), min(1.0, upper))


def calculate_objective_score(
    metrics: Any,
    min_classification_samples: int = 100,
    min_roc_auc: float = 0.5,
    min_mcc: float = 0.02,
    require_significance: bool = True,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate the objective score from metrics using statistically robust cutoffs.
    
    Implements multiple layers of filtering:
    1. Hard cutoff on sample count (statistical reliability)
    2. Hard cutoff on ROC AUC (reject worse-than-random)
    3. Hard cutoff on minimum MCC (reject trivially small effects)
    4. Statistical significance test (confidence interval lower bound > 0)
    
    Args:
        metrics: BacktestMetrics object with mcc, roc_auc, classification_samples
        min_classification_samples: Minimum samples required (default 100)
        min_roc_auc: Minimum ROC AUC required (default 0.5 = random)
        min_mcc: Minimum MCC threshold (default 0.02, filters trivial effects)
        require_significance: Require 95% CI lower bound > 0 (default True)
        confidence_level: Confidence level for significance test (default 0.95)
    
    Returns:
        Raw MCC if all thresholds are met, otherwise -infinity (rejected)
    """
    # Reject if MCC is not a valid number
    if not math.isfinite(metrics.mcc):
        return float('-inf')
    
    # Hard cutoff 1: insufficient samples = reject
    # Not enough data to trust the classification result
    if metrics.classification_samples < min_classification_samples:
        return float('-inf')
    
    # Hard cutoff 2: ROC AUC below random = reject
    # ROC AUC < 0.5 means the indicator predicts the opposite direction
    if metrics.roc_auc < min_roc_auc:
        return float('-inf')
    
    # Hard cutoff 3: MCC too small = reject
    # Filter out trivially small effects that won't matter in practice
    if metrics.mcc < min_mcc:
        return float('-inf')
    
    # Hard cutoff 4: Statistical significance test
    # Require that the confidence interval lower bound is > 0
    # This ensures we're confident the MCC is actually positive
    if require_significance:
        lower_bound, _ = calculate_mcc_confidence_interval(
            metrics.mcc, 
            metrics.classification_samples,
            confidence_level
        )
        if lower_bound <= 0:
            return float('-inf')
    
    # All checks passed - return raw MCC for optimization
    return float(metrics.mcc)
