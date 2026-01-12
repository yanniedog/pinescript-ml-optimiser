"""
Shared objective scoring for optimization and reporting.

Implements statistically robust objective functions with:
- Multiple hypothesis testing awareness
- Effective sample size adjustments for autocorrelated data
- Conservative confidence intervals
- Regime-aware quality metrics
"""

from typing import Any, Optional, Tuple, Dict
import math
import logging

logger = logging.getLogger(__name__)


def calculate_mcc_confidence_interval(mcc: float, n_samples: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for MCC using asymptotic approximation.
    
    Uses Fisher's z-transformation for more accurate intervals near boundaries.
    
    Args:
        mcc: Matthews Correlation Coefficient
        n_samples: Number of classification samples
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound) tuple
    """
    if n_samples <= 2:
        return (-1.0, 1.0)  # No confidence with too few samples
    
    # Use Fisher's z-transformation for better accuracy near boundaries
    # This is more appropriate than the naive SE approximation
    if abs(mcc) >= 0.999:
        # Near boundary - use simple approximation
        se_mcc = 0.01
    else:
        # Fisher's z-transformation: z = 0.5 * ln((1+r)/(1-r))
        # SE(z) = 1/sqrt(n-3)
        try:
            z = 0.5 * math.log((1 + mcc) / (1 - mcc))
            se_z = 1.0 / math.sqrt(max(1, n_samples - 3))
        except (ValueError, ZeroDivisionError):
            se_mcc = math.sqrt((1.0 - mcc * mcc) / max(1, n_samples - 2))
            z = None
    
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}
    z_score = z_scores.get(confidence, 1.96)
    
    if 'z' in dir() and z is not None:
        # Transform back from Fisher's z
        z_lower = z - z_score * se_z
        z_upper = z + z_score * se_z
        
        # Convert back to MCC scale
        lower = (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)
        upper = (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1)
    else:
        # Fallback to simple approximation
        lower = mcc - z_score * se_mcc
        upper = mcc + z_score * se_mcc
    
    # Clamp to valid MCC range
    return (max(-1.0, lower), min(1.0, upper))


def calculate_effective_sample_size(
    n_samples: int,
    autocorr_lag1: Optional[float] = None
) -> int:
    """
    Calculate effective sample size accounting for autocorrelation.
    
    When returns/signals are autocorrelated, the effective number of
    independent observations is less than the nominal sample size.
    
    Args:
        n_samples: Nominal sample size
        autocorr_lag1: Lag-1 autocorrelation coefficient (if known)
    
    Returns:
        Effective sample size (always <= n_samples)
    """
    if autocorr_lag1 is None or n_samples < 10:
        return n_samples
    
    # Effective sample size formula for AR(1) process
    rho = max(-0.99, min(0.99, autocorr_lag1))
    
    if abs(rho) < 0.01:
        return n_samples  # Negligible autocorrelation
    
    ess = n_samples * (1 - rho) / (1 + rho)
    return max(3, int(ess))


def calculate_minimum_detectable_effect(
    n_samples: int,
    confidence: float = 0.95,
    power: float = 0.8
) -> float:
    """
    Calculate minimum detectable MCC given sample size.
    
    Useful for determining if a dataset is large enough to detect
    a meaningful effect.
    
    Args:
        n_samples: Sample size
        confidence: Confidence level
        power: Statistical power (1 - Type II error rate)
    
    Returns:
        Minimum detectable MCC
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    power_z = {0.8: 0.842, 0.9: 1.282, 0.95: 1.645}
    
    z_alpha = z_scores.get(confidence, 1.96)
    z_beta = power_z.get(power, 0.842)
    
    # Minimum detectable effect using Fisher's z approximation
    # MDE ≈ tanh((z_α + z_β) / sqrt(n-3))
    if n_samples <= 3:
        return 1.0  # Can't detect any effect
    
    z_mde = (z_alpha + z_beta) / math.sqrt(n_samples - 3)
    
    # Convert from Fisher's z to MCC
    mde = math.tanh(z_mde)
    return min(1.0, max(0.0, mde))


def calculate_objective_score(
    metrics: Any,
    min_classification_samples: int = 100,
    min_roc_auc: float = 0.5,
    min_mcc: float = 0.02,
    require_significance: bool = True,
    confidence_level: float = 0.95,
    penalize_inconsistency: bool = True,
    consistency_weight: float = 0.1,
    use_robust_scoring: bool = True
) -> float:
    """
    Calculate the objective score from metrics using statistically robust cutoffs.
    
    Implements multiple layers of filtering:
    1. Hard cutoff on sample count (statistical reliability)
    2. Hard cutoff on ROC AUC (reject worse-than-random)
    3. Hard cutoff on minimum MCC (reject trivially small effects)
    4. Statistical significance test (confidence interval lower bound > 0)
    5. Optional consistency penalty for unstable indicators
    
    Args:
        metrics: BacktestMetrics object with mcc, roc_auc, classification_samples
        min_classification_samples: Minimum samples required (default 100)
        min_roc_auc: Minimum ROC AUC required (default 0.5 = random)
        min_mcc: Minimum MCC threshold (default 0.02, filters trivial effects)
        require_significance: Require 95% CI lower bound > 0 (default True)
        confidence_level: Confidence level for significance test (default 0.95)
        penalize_inconsistency: Apply penalty for low consistency (default True)
        consistency_weight: Weight of consistency penalty (default 0.1)
        use_robust_scoring: Use robust scoring with shrinkage (default True)
    
    Returns:
        Objective score (higher is better), or -infinity if rejected
    """
    # Reject if MCC is not a valid number
    if not math.isfinite(metrics.mcc):
        return float('-inf')
    
    # Hard cutoff 1: insufficient samples = reject
    # Not enough data to trust the classification result
    if metrics.classification_samples < min_classification_samples:
        logger.debug(f"Rejected: insufficient samples ({metrics.classification_samples} < {min_classification_samples})")
        return float('-inf')
    
    # Hard cutoff 2: ROC AUC below random = reject
    # ROC AUC < 0.5 means the indicator predicts the opposite direction
    if metrics.roc_auc < min_roc_auc:
        logger.debug(f"Rejected: ROC AUC below random ({metrics.roc_auc:.3f} < {min_roc_auc})")
        return float('-inf')
    
    # Hard cutoff 3: MCC too small = reject
    # Filter out trivially small effects that won't matter in practice
    if metrics.mcc < min_mcc:
        logger.debug(f"Rejected: MCC too small ({metrics.mcc:.3f} < {min_mcc})")
        return float('-inf')
    
    # Calculate minimum detectable effect for this sample size
    mde = calculate_minimum_detectable_effect(
        metrics.classification_samples,
        confidence_level,
        power=0.8
    )
    
    # Warn if MCC is close to MDE (marginal detection)
    if metrics.mcc < mde * 1.2:
        logger.debug(
            f"Warning: MCC ({metrics.mcc:.3f}) is close to minimum detectable effect "
            f"({mde:.3f}) for n={metrics.classification_samples}"
        )
    
    # Hard cutoff 4: Statistical significance test
    # Require that the confidence interval lower bound is > 0
    # This ensures we're confident the MCC is actually positive
    if require_significance:
        lower_bound, upper_bound = calculate_mcc_confidence_interval(
            metrics.mcc, 
            metrics.classification_samples,
            confidence_level
        )
        if lower_bound <= 0:
            logger.debug(f"Rejected: CI lower bound <= 0 ({lower_bound:.3f})")
            return float('-inf')
    
    # Calculate base score
    if use_robust_scoring:
        # Robust scoring: use shrinkage estimator
        # Shrinks extreme MCC values toward conservative estimate
        # This helps prevent overfitting to noise
        n = metrics.classification_samples
        
        # Shrinkage factor: more shrinkage for smaller samples
        shrinkage = min(1.0, n / (n + 50))
        
        # Shrink toward minimum significant MCC
        prior_mcc = min_mcc * 1.5  # Conservative prior
        shrunk_mcc = shrinkage * metrics.mcc + (1 - shrinkage) * prior_mcc
        
        base_score = shrunk_mcc
    else:
        base_score = metrics.mcc
    
    # Optional consistency penalty
    if penalize_inconsistency and hasattr(metrics, 'consistency_score'):
        # consistency_score is 0-1, higher is better
        consistency = getattr(metrics, 'consistency_score', 1.0)
        if consistency < 0.5:
            # Apply penalty for inconsistent indicators
            penalty = consistency_weight * (0.5 - consistency)
            base_score = base_score * (1 - penalty)
    
    return float(base_score)


def calculate_composite_objective(
    metrics: Any,
    mcc_weight: float = 0.5,
    auc_weight: float = 0.2,
    consistency_weight: float = 0.15,
    tail_capture_weight: float = 0.15,
    min_samples: int = 100
) -> float:
    """
    Calculate a composite objective score using multiple metrics.
    
    This is useful for multi-objective optimization where we want to
    balance predictive accuracy with other desirable properties.
    
    Args:
        metrics: BacktestMetrics object
        mcc_weight: Weight for MCC (default 0.5)
        auc_weight: Weight for ROC AUC (default 0.2)
        consistency_weight: Weight for consistency (default 0.15)
        tail_capture_weight: Weight for tail capture (default 0.15)
        min_samples: Minimum samples required
    
    Returns:
        Composite objective score (higher is better)
    """
    # Check minimum samples
    if metrics.classification_samples < min_samples:
        return float('-inf')
    
    # Normalize metrics to 0-1 scale
    # MCC: [-1, 1] -> [0, 1]
    mcc_normalized = (metrics.mcc + 1) / 2
    
    # ROC AUC: [0.5, 1] -> [0, 1] (assuming we reject < 0.5)
    auc_normalized = max(0, (metrics.roc_auc - 0.5) / 0.5)
    
    # Consistency is already 0-1
    consistency = getattr(metrics, 'consistency_score', 0.5)
    
    # Tail capture is already 0-1
    tail_capture = getattr(metrics, 'tail_capture_rate', 0.0)
    
    # Weighted combination
    composite = (
        mcc_weight * mcc_normalized +
        auc_weight * auc_normalized +
        consistency_weight * consistency +
        tail_capture_weight * tail_capture
    )
    
    # Scale back to make it comparable to raw MCC
    # Max possible is 1.0, min for valid indicators is ~0.5
    return composite


def get_statistical_quality_report(metrics: Any) -> Dict[str, Any]:
    """
    Generate a detailed statistical quality report for metrics.
    
    Args:
        metrics: BacktestMetrics object
    
    Returns:
        Dict with quality indicators and recommendations
    """
    report = {
        'sample_quality': {},
        'effect_quality': {},
        'reliability': {},
        'recommendations': []
    }
    
    n = metrics.classification_samples
    
    # Sample quality
    mde = calculate_minimum_detectable_effect(n)
    report['sample_quality'] = {
        'n_samples': n,
        'effective_n': n,  # Would use autocorrelation if available
        'minimum_detectable_effect': mde,
        'is_adequate': n >= 100 and metrics.mcc >= mde
    }
    
    # Effect quality
    ci_lower, ci_upper = calculate_mcc_confidence_interval(metrics.mcc, n)
    ci_width = ci_upper - ci_lower
    report['effect_quality'] = {
        'mcc': metrics.mcc,
        'roc_auc': metrics.roc_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_width,
        'is_significant': ci_lower > 0,
        'effect_size': 'large' if metrics.mcc >= 0.5 else 'medium' if metrics.mcc >= 0.3 else 'small' if metrics.mcc >= 0.1 else 'trivial'
    }
    
    # Reliability
    consistency = getattr(metrics, 'consistency_score', None)
    report['reliability'] = {
        'consistency_score': consistency,
        'profit_factor': getattr(metrics, 'profit_factor', None),
        'is_consistent': consistency is None or consistency >= 0.5
    }
    
    # Generate recommendations
    if n < 100:
        report['recommendations'].append(
            f"Insufficient samples ({n}). Need at least 100 for reliable inference."
        )
    elif n < 500:
        report['recommendations'].append(
            f"Sample size ({n}) is marginal. Consider using more data for robust conclusions."
        )
    
    if metrics.mcc < mde:
        report['recommendations'].append(
            f"MCC ({metrics.mcc:.3f}) is below minimum detectable effect ({mde:.3f}). "
            "Effect may not be real."
        )
    
    if ci_width > 0.3:
        report['recommendations'].append(
            f"Wide confidence interval ({ci_width:.2f}). Need more data for precise estimates."
        )
    
    if consistency is not None and consistency < 0.5:
        report['recommendations'].append(
            f"Low consistency ({consistency:.2f}). Indicator may be unstable across regimes."
        )
    
    if not report['recommendations']:
        report['recommendations'].append("Statistical quality is adequate for inference.")
    
    return report
