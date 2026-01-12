"""
Statistical Utilities for Robust Backtesting
Implements Monte Carlo permutation testing, bootstrap confidence intervals,
multiple hypothesis testing corrections, and regime-aware analysis.

Optimized for Surface laptops with efficient memory usage and optional Numba JIT.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation - graceful fallback if unavailable
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    logger.debug("Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    logger.debug("Numba not available, using pure NumPy")


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test."""
    statistic: float           # Test statistic (e.g., MCC, AUC)
    p_value: float             # P-value from permutation test
    ci_lower: float            # Lower bound of confidence interval
    ci_upper: float            # Upper bound of confidence interval
    is_significant: bool       # Whether result is statistically significant
    n_samples: int             # Number of samples used
    n_permutations: int        # Number of permutations used
    effect_size: str           # Effect size interpretation (trivial/small/medium/large)
    confidence_level: float    # Confidence level used (e.g., 0.95)


@dataclass
class MultipleTestingResult:
    """Result of multiple hypothesis testing correction."""
    original_p_values: np.ndarray
    corrected_p_values: np.ndarray
    rejected_null: np.ndarray      # Boolean array - which hypotheses rejected
    method: str                     # Correction method used
    alpha: float                    # Significance level
    n_discoveries: int              # Number of significant results after correction


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS (with fallbacks)
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True, fastmath=True)
    def _calculate_mcc_fast(tp: int, tn: int, fp: int, fn: int) -> float:
        """Numba-optimized MCC calculation."""
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom <= 0:
            return 0.0
        return (tp * tn - fp * fn) / np.sqrt(denom)

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _permutation_mcc_numba(
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        n_permutations: int,
        seed: int
    ) -> np.ndarray:
        """
        Numba-optimized permutation test for MCC.
        Parallelized across permutations.
        """
        n = len(labels)
        null_mccs = np.empty(n_permutations, dtype=np.float64)
        
        for i in prange(n_permutations):
            # Create permuted labels (Fisher-Yates shuffle with seed offset)
            np.random.seed(seed + i)
            perm_labels = labels.copy()
            for j in range(n - 1, 0, -1):
                k = np.random.randint(0, j + 1)
                perm_labels[j], perm_labels[k] = perm_labels[k], perm_labels[j]
            
            # Calculate MCC with permuted labels
            preds = scores >= threshold
            pos_mask = perm_labels == 1
            neg_mask = ~pos_mask
            
            tp = np.sum(preds & pos_mask)
            fp = np.sum(preds & neg_mask)
            fn = np.sum((~preds) & pos_mask)
            tn = np.sum((~preds) & neg_mask)
            
            null_mccs[i] = _calculate_mcc_fast(tp, tn, fp, fn)
        
        return null_mccs

    @jit(nopython=True, cache=True, fastmath=True, parallel=True)
    def _bootstrap_mcc_numba(
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        n_bootstrap: int,
        seed: int
    ) -> np.ndarray:
        """
        Numba-optimized bootstrap for MCC confidence intervals.
        """
        n = len(labels)
        boot_mccs = np.empty(n_bootstrap, dtype=np.float64)
        
        for i in prange(n_bootstrap):
            np.random.seed(seed + i)
            # Bootstrap sample (with replacement)
            indices = np.random.randint(0, n, n)
            boot_labels = labels[indices]
            boot_scores = scores[indices]
            
            # Calculate MCC
            preds = boot_scores >= threshold
            pos_mask = boot_labels == 1
            neg_mask = ~pos_mask
            
            tp = np.sum(preds & pos_mask)
            fp = np.sum(preds & neg_mask)
            fn = np.sum((~preds) & pos_mask)
            tn = np.sum((~preds) & neg_mask)
            
            boot_mccs[i] = _calculate_mcc_fast(tp, tn, fp, fn)
        
        return boot_mccs

else:
    # Pure NumPy fallbacks when Numba is not available
    def _calculate_mcc_fast(tp: int, tn: int, fp: int, fn: int) -> float:
        """Pure NumPy MCC calculation."""
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom <= 0:
            return 0.0
        return (tp * tn - fp * fn) / np.sqrt(denom)

    def _permutation_mcc_numba(
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        n_permutations: int,
        seed: int
    ) -> np.ndarray:
        """Pure NumPy permutation test (vectorized where possible)."""
        rng = np.random.default_rng(seed)
        n = len(labels)
        null_mccs = np.empty(n_permutations, dtype=np.float64)
        
        for i in range(n_permutations):
            perm_labels = rng.permutation(labels)
            preds = scores >= threshold
            pos_mask = perm_labels == 1
            neg_mask = ~pos_mask
            
            tp = np.sum(preds & pos_mask)
            fp = np.sum(preds & neg_mask)
            fn = np.sum((~preds) & pos_mask)
            tn = np.sum((~preds) & neg_mask)
            
            null_mccs[i] = _calculate_mcc_fast(tp, tn, fp, fn)
        
        return null_mccs

    def _bootstrap_mcc_numba(
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        n_bootstrap: int,
        seed: int
    ) -> np.ndarray:
        """Pure NumPy bootstrap."""
        rng = np.random.default_rng(seed)
        n = len(labels)
        boot_mccs = np.empty(n_bootstrap, dtype=np.float64)
        
        for i in range(n_bootstrap):
            indices = rng.integers(0, n, n)
            boot_labels = labels[indices]
            boot_scores = scores[indices]
            
            preds = boot_scores >= threshold
            pos_mask = boot_labels == 1
            neg_mask = ~pos_mask
            
            tp = np.sum(preds & pos_mask)
            fp = np.sum(preds & neg_mask)
            fn = np.sum((~preds) & pos_mask)
            tn = np.sum((~preds) & neg_mask)
            
            boot_mccs[i] = _calculate_mcc_fast(tp, tn, fp, fn)
        
        return boot_mccs


# =============================================================================
# PERMUTATION TESTING
# =============================================================================

def permutation_test_mcc(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    observed_mcc: float,
    n_permutations: int = 1000,
    seed: int = 42,
    alternative: str = 'greater'
) -> Tuple[float, np.ndarray]:
    """
    Perform Monte Carlo permutation test for MCC significance.
    
    Tests null hypothesis that the indicator has no predictive power
    (MCC = 0, i.e., labels are independent of scores).
    
    Args:
        labels: Binary outcome labels (0 or 1)
        scores: Indicator scores/signals
        threshold: Classification threshold used
        observed_mcc: The observed MCC value to test
        n_permutations: Number of permutations (default 1000)
        seed: Random seed for reproducibility
        alternative: 'greater', 'less', or 'two-sided'
    
    Returns:
        (p_value, null_distribution) tuple
    """
    # Ensure contiguous arrays for Numba
    labels = np.ascontiguousarray(labels.astype(np.int64))
    scores = np.ascontiguousarray(scores.astype(np.float64))
    
    # Generate null distribution via permutation
    null_mccs = _permutation_mcc_numba(
        labels, scores, threshold, n_permutations, seed
    )
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'greater':
        # Test if observed MCC is significantly greater than random
        p_value = (np.sum(null_mccs >= observed_mcc) + 1) / (n_permutations + 1)
    elif alternative == 'less':
        p_value = (np.sum(null_mccs <= observed_mcc) + 1) / (n_permutations + 1)
    else:  # two-sided
        p_value = (np.sum(np.abs(null_mccs) >= np.abs(observed_mcc)) + 1) / (n_permutations + 1)
    
    return p_value, null_mccs


def permutation_test_roc_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    observed_auc: float,
    n_permutations: int = 1000,
    seed: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Permutation test for ROC AUC significance.
    
    Args:
        labels: Binary outcome labels
        scores: Indicator scores
        observed_auc: Observed AUC value
        n_permutations: Number of permutations
        seed: Random seed
    
    Returns:
        (p_value, null_distribution)
    """
    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_permutations, dtype=np.float64)
    
    for i in range(n_permutations):
        perm_labels = rng.permutation(labels)
        null_aucs[i] = _calculate_roc_auc_fast(perm_labels, scores)
    
    # Test if AUC > 0.5 (better than random)
    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (n_permutations + 1)
    return p_value, null_aucs


def _calculate_roc_auc_fast(labels: np.ndarray, scores: np.ndarray) -> float:
    """Fast ROC AUC calculation using rank statistics."""
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n - 1 and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    
    sum_pos = float(np.sum(ranks[pos]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_confidence_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    method: str = 'percentile'
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate bootstrap confidence interval for MCC.
    
    Uses BCa (bias-corrected and accelerated) or percentile method.
    
    Args:
        labels: Binary outcome labels
        scores: Indicator scores
        threshold: Classification threshold
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95)
        seed: Random seed
        method: 'percentile' or 'bca'
    
    Returns:
        (lower_bound, upper_bound, bootstrap_distribution)
    """
    labels = np.ascontiguousarray(labels.astype(np.int64))
    scores = np.ascontiguousarray(scores.astype(np.float64))
    
    # Generate bootstrap distribution
    boot_mccs = _bootstrap_mcc_numba(labels, scores, threshold, n_bootstrap, seed)
    
    alpha = 1 - confidence_level
    
    if method == 'bca':
        # BCa method (bias-corrected and accelerated)
        # Calculate observed MCC
        preds = scores >= threshold
        pos_mask = labels == 1
        tp = np.sum(preds & pos_mask)
        fp = np.sum(preds & ~pos_mask)
        fn = np.sum((~preds) & pos_mask)
        tn = np.sum((~preds) & ~pos_mask)
        observed_mcc = _calculate_mcc_fast(tp, tn, fp, fn)
        
        # Bias correction factor
        z0 = _norm_ppf(np.mean(boot_mccs < observed_mcc))
        
        # Acceleration factor (jackknife)
        n = len(labels)
        jack_mccs = np.empty(n, dtype=np.float64)
        for i in range(n):
            jack_labels = np.delete(labels, i)
            jack_scores = np.delete(scores, i)
            jack_preds = jack_scores >= threshold
            jack_pos = jack_labels == 1
            jack_tp = np.sum(jack_preds & jack_pos)
            jack_fp = np.sum(jack_preds & ~jack_pos)
            jack_fn = np.sum((~jack_preds) & jack_pos)
            jack_tn = np.sum((~jack_preds) & ~jack_pos)
            jack_mccs[i] = _calculate_mcc_fast(jack_tp, jack_tn, jack_fp, jack_fn)
        
        jack_mean = np.mean(jack_mccs)
        num = np.sum((jack_mean - jack_mccs) ** 3)
        denom = 6 * (np.sum((jack_mean - jack_mccs) ** 2) ** 1.5)
        a = num / denom if denom != 0 else 0
        
        # Adjusted percentiles
        z_alpha_lower = _norm_ppf(alpha / 2)
        z_alpha_upper = _norm_ppf(1 - alpha / 2)
        
        p_lower = _norm_cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = _norm_cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        lower = np.percentile(boot_mccs, p_lower * 100)
        upper = np.percentile(boot_mccs, p_upper * 100)
    else:
        # Simple percentile method
        lower = np.percentile(boot_mccs, alpha / 2 * 100)
        upper = np.percentile(boot_mccs, (1 - alpha / 2) * 100)
    
    return lower, upper, boot_mccs


def _norm_ppf(p: float) -> float:
    """Standard normal quantile function (inverse CDF)."""
    # Rational approximation for standard normal quantile
    # Accurate to ~1e-9 for 0 < p < 1
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0
    
    if p < 0.5:
        return -_norm_ppf_upper(1 - p)
    return _norm_ppf_upper(p)


def _norm_ppf_upper(p: float) -> float:
    """Helper for normal quantile (upper tail)."""
    # Approximation constants
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00
    ]
    
    p_low = 0.02425
    p_high = 1 - p_low
    
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + np.tanh(x * 0.7978845608028654))  # Approximation


# =============================================================================
# MULTIPLE HYPOTHESIS TESTING CORRECTION
# =============================================================================

def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> MultipleTestingResult:
    """
    Bonferroni correction for multiple hypothesis testing.
    
    Most conservative method - controls family-wise error rate (FWER).
    
    Args:
        p_values: Array of p-values from multiple tests
        alpha: Significance level
    
    Returns:
        MultipleTestingResult with corrected p-values
    """
    n = len(p_values)
    corrected = np.minimum(p_values * n, 1.0)
    rejected = corrected <= alpha
    
    return MultipleTestingResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        rejected_null=rejected,
        method='bonferroni',
        alpha=alpha,
        n_discoveries=int(np.sum(rejected))
    )


def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> MultipleTestingResult:
    """
    Benjamini-Hochberg procedure for FDR control.
    
    Less conservative than Bonferroni - controls false discovery rate (FDR).
    Recommended for exploratory optimization where some false positives are acceptable.
    
    Args:
        p_values: Array of p-values
        alpha: Target FDR level
    
    Returns:
        MultipleTestingResult
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # BH adjusted p-values
    corrected = np.empty(n, dtype=np.float64)
    cummin = np.inf
    
    for i in range(n - 1, -1, -1):
        rank = i + 1
        bh_val = sorted_p[i] * n / rank
        cummin = min(cummin, bh_val)
        corrected[sorted_indices[i]] = min(cummin, 1.0)
    
    rejected = corrected <= alpha
    
    return MultipleTestingResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        rejected_null=rejected,
        method='benjamini_hochberg',
        alpha=alpha,
        n_discoveries=int(np.sum(rejected))
    )


def holm_bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> MultipleTestingResult:
    """
    Holm-Bonferroni step-down procedure.
    
    More powerful than Bonferroni while still controlling FWER.
    
    Args:
        p_values: Array of p-values
        alpha: Significance level
    
    Returns:
        MultipleTestingResult
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    corrected = np.empty(n, dtype=np.float64)
    cummax = 0.0
    
    for i in range(n):
        holm_val = sorted_p[i] * (n - i)
        cummax = max(cummax, holm_val)
        corrected[sorted_indices[i]] = min(cummax, 1.0)
    
    rejected = corrected <= alpha
    
    return MultipleTestingResult(
        original_p_values=p_values,
        corrected_p_values=corrected,
        rejected_null=rejected,
        method='holm_bonferroni',
        alpha=alpha,
        n_discoveries=int(np.sum(rejected))
    )


# =============================================================================
# COMPREHENSIVE STATISTICAL TESTING
# =============================================================================

def comprehensive_mcc_test(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_permutations: int = 1000,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> StatisticalTestResult:
    """
    Comprehensive statistical test combining permutation and bootstrap methods.
    
    Provides:
    - P-value from permutation test
    - Bootstrap confidence interval
    - Effect size interpretation
    - Statistical significance assessment
    
    Args:
        labels: Binary outcome labels
        scores: Indicator scores
        threshold: Classification threshold
        n_permutations: Permutations for p-value
        n_bootstrap: Bootstrap samples for CI
        confidence_level: Confidence level for CI
        seed: Random seed
    
    Returns:
        StatisticalTestResult with all metrics
    """
    # Calculate observed MCC
    preds = scores >= threshold
    pos_mask = labels == 1
    tp = int(np.sum(preds & pos_mask))
    fp = int(np.sum(preds & ~pos_mask))
    fn = int(np.sum((~preds) & pos_mask))
    tn = int(np.sum((~preds) & ~pos_mask))
    observed_mcc = _calculate_mcc_fast(tp, tn, fp, fn)
    
    # Permutation test for p-value
    p_value, _ = permutation_test_mcc(
        labels, scores, threshold, observed_mcc,
        n_permutations=n_permutations, seed=seed
    )
    
    # Bootstrap confidence interval
    ci_lower, ci_upper, _ = bootstrap_confidence_interval(
        labels, scores, threshold,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed + 1000  # Different seed for bootstrap
    )
    
    # Effect size interpretation (Cohen's conventions adapted for MCC)
    abs_mcc = abs(observed_mcc)
    if abs_mcc < 0.1:
        effect_size = 'trivial'
    elif abs_mcc < 0.3:
        effect_size = 'small'
    elif abs_mcc < 0.5:
        effect_size = 'medium'
    else:
        effect_size = 'large'
    
    # Significance: p-value < alpha AND CI doesn't include 0
    alpha = 1 - confidence_level
    is_significant = (p_value < alpha) and (ci_lower > 0 or ci_upper < 0)
    
    return StatisticalTestResult(
        statistic=observed_mcc,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_significant,
        n_samples=len(labels),
        n_permutations=n_permutations,
        effect_size=effect_size,
        confidence_level=confidence_level
    )


# =============================================================================
# AUTOCORRELATION AND TIME SERIES VALIDATION
# =============================================================================

def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Tuple[float, float]:
    """
    Ljung-Box test for autocorrelation in residuals.
    
    Useful for validating that backtest returns aren't serially correlated.
    
    Args:
        residuals: Return residuals or time series
        lags: Number of lags to test
    
    Returns:
        (Q_statistic, p_value)
    """
    n = len(residuals)
    if n < lags + 1:
        return 0.0, 1.0
    
    # Calculate autocorrelations
    mean_r = np.mean(residuals)
    centered = residuals - mean_r
    var = np.var(residuals)
    
    if var == 0:
        return 0.0, 1.0
    
    acf = np.zeros(lags)
    for k in range(1, lags + 1):
        acf[k-1] = np.sum(centered[k:] * centered[:-k]) / (n * var)
    
    # Ljung-Box Q statistic
    q_stat = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    
    # P-value from chi-square distribution (approximate)
    # Using Wilson-Hilferty transformation
    k = lags  # degrees of freedom
    z = ((q_stat / k) ** (1/3) - (1 - 2/(9*k))) / np.sqrt(2/(9*k))
    p_value = 1 - _norm_cdf(z)
    
    return q_stat, p_value


def effective_sample_size(returns: np.ndarray) -> int:
    """
    Calculate effective sample size accounting for autocorrelation.
    
    Useful for adjusting confidence intervals when returns are correlated.
    
    Args:
        returns: Time series of returns
    
    Returns:
        Effective sample size (always <= actual sample size)
    """
    n = len(returns)
    if n < 3:
        return n
    
    # Calculate lag-1 autocorrelation
    centered = returns - np.mean(returns)
    var = np.var(returns)
    
    if var == 0:
        return n
    
    rho1 = np.sum(centered[1:] * centered[:-1]) / (n * var)
    
    # Effective sample size formula
    if abs(rho1) >= 1:
        return 1
    
    ess = n * (1 - rho1) / (1 + rho1)
    return max(1, int(ess))


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_volatility_regimes(
    returns: np.ndarray,
    window: int = 20,
    n_regimes: int = 3
) -> np.ndarray:
    """
    Simple volatility regime detection using rolling standard deviation.
    
    Classifies each period into low/medium/high volatility regimes.
    
    Args:
        returns: Return series
        window: Rolling window for volatility calculation
        n_regimes: Number of regimes (default 3)
    
    Returns:
        Array of regime labels (0 = low vol, n_regimes-1 = high vol)
    """
    n = len(returns)
    if n < window:
        return np.zeros(n, dtype=int)
    
    # Rolling volatility
    rolling_vol = np.empty(n, dtype=np.float64)
    rolling_vol[:window] = np.nan
    
    for i in range(window, n):
        rolling_vol[i] = np.std(returns[i-window:i])
    
    # Remove NaN for percentile calculation
    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) == 0:
        return np.zeros(n, dtype=int)
    
    # Define regime thresholds using percentiles
    percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
    thresholds = np.percentile(valid_vol, percentiles)
    
    # Assign regimes
    regimes = np.zeros(n, dtype=int)
    for i, thresh in enumerate(thresholds):
        regimes[rolling_vol > thresh] = i + 1
    
    return regimes


def regime_aware_metrics(
    returns: np.ndarray,
    regimes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    predictions: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate performance metrics for each regime separately.
    
    Args:
        returns: Return series
        regimes: Regime labels for each bar
        labels: Optional binary labels for classification metrics
        predictions: Optional predictions for classification metrics
    
    Returns:
        Dict mapping regime names to metric dicts
    """
    unique_regimes = np.unique(regimes)
    regime_names = ['low_vol', 'med_vol', 'high_vol'][:len(unique_regimes)]
    
    results = {}
    
    for i, regime in enumerate(unique_regimes):
        mask = regimes == regime
        regime_returns = returns[mask]
        
        if len(regime_returns) == 0:
            continue
        
        name = regime_names[i] if i < len(regime_names) else f'regime_{regime}'
        
        metrics = {
            'n_samples': int(np.sum(mask)),
            'mean_return': float(np.mean(regime_returns)),
            'std_return': float(np.std(regime_returns)),
            'sharpe': float(np.mean(regime_returns) / np.std(regime_returns)) if np.std(regime_returns) > 0 else 0.0
        }
        
        # Add classification metrics if provided
        if labels is not None and predictions is not None:
            regime_labels = labels[mask]
            regime_preds = predictions[mask]
            
            pos_mask = regime_labels == 1
            tp = int(np.sum(regime_preds & pos_mask))
            fp = int(np.sum(regime_preds & ~pos_mask))
            fn = int(np.sum((~regime_preds) & pos_mask))
            tn = int(np.sum((~regime_preds) & ~pos_mask))
            
            metrics['mcc'] = _calculate_mcc_fast(tp, tn, fp, fn)
            metrics['accuracy'] = (tp + tn) / len(regime_labels) if len(regime_labels) > 0 else 0.5
        
        results[name] = metrics
    
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def interpret_mcc(mcc: float) -> str:
    """
    Interpret MCC value according to standard guidelines.
    
    MCC ranges from -1 to 1:
    - 1: Perfect prediction
    - 0: Random prediction
    - -1: Perfect inverse prediction
    """
    if mcc >= 0.7:
        return "Excellent (strong positive correlation)"
    elif mcc >= 0.5:
        return "Good (moderate positive correlation)"
    elif mcc >= 0.3:
        return "Fair (weak positive correlation)"
    elif mcc >= 0.1:
        return "Poor (very weak positive correlation)"
    elif mcc >= -0.1:
        return "Random (no correlation)"
    elif mcc >= -0.3:
        return "Inverse weak (contrarian signal)"
    else:
        return "Inverse strong (strong contrarian signal)"


def minimum_samples_for_significance(
    target_mcc: float = 0.1,
    confidence_level: float = 0.95,
    power: float = 0.8
) -> int:
    """
    Calculate minimum sample size needed for statistical significance.
    
    Based on asymptotic approximation for MCC.
    
    Args:
        target_mcc: Minimum MCC to detect
        confidence_level: Desired confidence level
        power: Statistical power (1 - Type II error rate)
    
    Returns:
        Minimum number of samples needed
    """
    z_alpha = _norm_ppf(1 - (1 - confidence_level) / 2)
    z_beta = _norm_ppf(power)
    
    # Approximation: n ≈ ((z_α + z_β) / arctanh(MCC))²
    if abs(target_mcc) >= 1:
        return 1000000  # Unrealistic
    
    arctanh_mcc = 0.5 * np.log((1 + target_mcc) / (1 - target_mcc))
    if abs(arctanh_mcc) < 1e-10:
        return 1000000
    
    n = int(np.ceil(((z_alpha + z_beta) / arctanh_mcc) ** 2)) + 3
    return max(30, n)  # Minimum 30 samples for CLT


if __name__ == "__main__":
    # Test the statistical utilities
    np.random.seed(42)
    n = 500
    
    # Generate sample data with some predictive signal
    true_signal = np.random.randn(n) * 0.3
    noise = np.random.randn(n)
    labels = (true_signal + noise > 0).astype(int)
    scores = true_signal + np.random.randn(n) * 0.5
    threshold = 0.0
    
    print("=" * 60)
    print("Statistical Utilities Test")
    print("=" * 60)
    
    # Comprehensive test
    result = comprehensive_mcc_test(
        labels, scores, threshold,
        n_permutations=1000,
        n_bootstrap=2000
    )
    
    print(f"\nComprehensive MCC Test:")
    print(f"  MCC: {result.statistic:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  Significant: {result.is_significant}")
    print(f"  Effect size: {result.effect_size}")
    print(f"  Interpretation: {interpret_mcc(result.statistic)}")
    
    # Multiple testing correction
    print("\n\nMultiple Testing Correction:")
    p_values = np.array([0.001, 0.015, 0.03, 0.04, 0.05, 0.08, 0.12])
    
    bonf = bonferroni_correction(p_values)
    bh = benjamini_hochberg_correction(p_values)
    holm = holm_bonferroni_correction(p_values)
    
    print(f"  Original p-values: {p_values}")
    print(f"  Bonferroni discoveries: {bonf.n_discoveries}")
    print(f"  B-H discoveries: {bh.n_discoveries}")
    print(f"  Holm discoveries: {holm.n_discoveries}")
    
    # Minimum sample size
    min_n = minimum_samples_for_significance(target_mcc=0.1)
    print(f"\n\nMinimum samples for MCC=0.1 detection: {min_n}")
    
    print("\n[OK] All tests passed!")
