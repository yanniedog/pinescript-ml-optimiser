"""
Technical Analysis Functions
Pine Script TA functions implemented in vectorized NumPy/Pandas.
All functions process bar-by-bar without look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from functools import lru_cache

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, pd.Series, list]


def ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def ensure_series(data: ArrayLike) -> pd.Series:
    """Convert input to pandas Series."""
    if isinstance(data, pd.Series):
        return data
    return pd.Series(data)


# =============================================================================
# MOVING AVERAGES
# =============================================================================

def sma(src: ArrayLike, length: int) -> np.ndarray:
    """Simple Moving Average - ta.sma(src, length)"""
    series = ensure_series(src)
    return series.rolling(window=length, min_periods=length).mean().values


def ema(src: ArrayLike, length: int) -> np.ndarray:
    """Exponential Moving Average - ta.ema(src, length)"""
    series = ensure_series(src)
    return series.ewm(span=length, adjust=False, min_periods=length).mean().values


def wma(src: ArrayLike, length: int) -> np.ndarray:
    """Weighted Moving Average - ta.wma(src, length)"""
    arr = ensure_array(src)
    weights = np.arange(1, length + 1)
    result = np.full(len(arr), np.nan)
    
    for i in range(length - 1, len(arr)):
        window = arr[i - length + 1:i + 1]
        result[i] = np.sum(window * weights) / np.sum(weights)
    
    return result


def rma(src: ArrayLike, length: int) -> np.ndarray:
    """Running Moving Average (Wilder's smoothing) - ta.rma(src, length)"""
    series = ensure_series(src)
    alpha = 1.0 / length
    return series.ewm(alpha=alpha, adjust=False, min_periods=length).mean().values


def vwma(src: ArrayLike, volume: ArrayLike, length: int) -> np.ndarray:
    """Volume Weighted Moving Average - ta.vwma(src, volume, length)"""
    src_arr = ensure_array(src)
    vol_arr = ensure_array(volume)
    
    src_vol = src_arr * vol_arr
    src_vol_sum = sma(src_vol, length) * length
    vol_sum = sma(vol_arr, length) * length
    
    result = np.where(vol_sum != 0, src_vol_sum / vol_sum, np.nan)
    return result


# =============================================================================
# VOLATILITY
# =============================================================================

def tr(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """True Range - ta.tr"""
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    
    prev_close = np.roll(close_arr, 1)
    prev_close[0] = close_arr[0]
    
    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - prev_close)
    tr3 = np.abs(low_arr - prev_close)
    
    return np.maximum(tr1, np.maximum(tr2, tr3))


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int) -> np.ndarray:
    """Average True Range - ta.atr(length)"""
    true_range = tr(high, low, close)
    return rma(true_range, length)


def stdev(src: ArrayLike, length: int) -> np.ndarray:
    """Standard Deviation - ta.stdev(src, length)"""
    series = ensure_series(src)
    return series.rolling(window=length, min_periods=length).std(ddof=0).values


def variance(src: ArrayLike, length: int) -> np.ndarray:
    """Variance - ta.variance(src, length)"""
    series = ensure_series(src)
    return series.rolling(window=length, min_periods=length).var(ddof=0).values


# =============================================================================
# MOMENTUM
# =============================================================================

def rsi(src: ArrayLike, length: int) -> np.ndarray:
    """Relative Strength Index - ta.rsi(src, length)"""
    arr = ensure_array(src)
    delta = np.diff(arr, prepend=arr[0])
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_val = 100 - (100 / (1 + rs))
    
    return rsi_val


def macd(src: ArrayLike, fast_len: int = 12, slow_len: int = 26, signal_len: int = 9) -> tuple:
    """MACD - returns (macd_line, signal_line, histogram)"""
    fast_ema = ema(src, fast_len)
    slow_ema = ema(src, slow_len)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_len)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def cci(high: ArrayLike, low: ArrayLike, close: ArrayLike, length: int) -> np.ndarray:
    """Commodity Channel Index - ta.cci(length)"""
    tp = (ensure_array(high) + ensure_array(low) + ensure_array(close)) / 3
    tp_sma = sma(tp, length)
    
    # Mean deviation
    series = ensure_series(tp)
    md = series.rolling(window=length, min_periods=length).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    ).values
    
    result = np.where(md != 0, (tp - tp_sma) / (0.015 * md), 0)
    return result


def mfi(high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike, length: int) -> np.ndarray:
    """Money Flow Index - ta.mfi(length)"""
    tp = (ensure_array(high) + ensure_array(low) + ensure_array(close)) / 3
    vol = ensure_array(volume)
    raw_mf = tp * vol
    
    tp_change = np.diff(tp, prepend=tp[0])
    pos_mf = np.where(tp_change > 0, raw_mf, 0)
    neg_mf = np.where(tp_change < 0, raw_mf, 0)
    
    pos_mf_sum = sma(pos_mf, length) * length
    neg_mf_sum = sma(neg_mf, length) * length
    
    mf_ratio = np.where(neg_mf_sum != 0, pos_mf_sum / neg_mf_sum, 100)
    mfi_val = 100 - (100 / (1 + mf_ratio))
    
    return mfi_val


def stoch(high: ArrayLike, low: ArrayLike, close: ArrayLike, k_length: int, d_length: int = 3) -> tuple:
    """Stochastic Oscillator - returns (k, d)"""
    high_arr = ensure_array(high)
    low_arr = ensure_array(low)
    close_arr = ensure_array(close)
    
    highest_high = highest(high_arr, k_length)
    lowest_low = lowest(low_arr, k_length)
    
    k = np.where(
        highest_high != lowest_low,
        100 * (close_arr - lowest_low) / (highest_high - lowest_low),
        50
    )
    d = sma(k, d_length)
    
    return k, d


# =============================================================================
# HIGHEST / LOWEST
# =============================================================================

def highest(src: ArrayLike, length: int) -> np.ndarray:
    """Highest value over period - ta.highest(src, length)"""
    series = ensure_series(src)
    return series.rolling(window=length, min_periods=1).max().values


def lowest(src: ArrayLike, length: int) -> np.ndarray:
    """Lowest value over period - ta.lowest(src, length)"""
    series = ensure_series(src)
    return series.rolling(window=length, min_periods=1).min().values


def highest_bars(src: ArrayLike, length: int) -> np.ndarray:
    """Bars since highest value - ta.highestbars(src, length)"""
    arr = ensure_array(src)
    result = np.zeros(len(arr))
    
    for i in range(len(arr)):
        start = max(0, i - length + 1)
        window = arr[start:i + 1]
        if len(window) > 0:
            max_idx = np.argmax(window)
            result[i] = -(len(window) - 1 - max_idx)
    
    return result


def lowest_bars(src: ArrayLike, length: int) -> np.ndarray:
    """Bars since lowest value - ta.lowestbars(src, length)"""
    arr = ensure_array(src)
    result = np.zeros(len(arr))
    
    for i in range(len(arr)):
        start = max(0, i - length + 1)
        window = arr[start:i + 1]
        if len(window) > 0:
            min_idx = np.argmin(window)
            result[i] = -(len(window) - 1 - min_idx)
    
    return result


# =============================================================================
# PIVOTS
# =============================================================================

def pivothigh(src: ArrayLike, left_bars: int, right_bars: int) -> np.ndarray:
    """Pivot High - ta.pivothigh(src, left, right)
    Returns the pivot value at the pivot bar, NaN elsewhere.
    Note: Pivot is confirmed 'right_bars' bars AFTER it occurs.
    """
    arr = ensure_array(src)
    result = np.full(len(arr), np.nan)
    
    for i in range(left_bars, len(arr) - right_bars):
        is_pivot = True
        center = arr[i]
        
        # Check left bars
        for j in range(1, left_bars + 1):
            if arr[i - j] >= center:
                is_pivot = False
                break
        
        if is_pivot:
            # Check right bars
            for j in range(1, right_bars + 1):
                if arr[i + j] > center:
                    is_pivot = False
                    break
        
        if is_pivot:
            # Return at the confirmation bar (i + right_bars)
            result[i + right_bars] = center
    
    return result


def pivotlow(src: ArrayLike, left_bars: int, right_bars: int) -> np.ndarray:
    """Pivot Low - ta.pivotlow(src, left, right)
    Returns the pivot value at the pivot bar, NaN elsewhere.
    """
    arr = ensure_array(src)
    result = np.full(len(arr), np.nan)
    
    for i in range(left_bars, len(arr) - right_bars):
        is_pivot = True
        center = arr[i]
        
        # Check left bars
        for j in range(1, left_bars + 1):
            if arr[i - j] <= center:
                is_pivot = False
                break
        
        if is_pivot:
            # Check right bars
            for j in range(1, right_bars + 1):
                if arr[i + j] < center:
                    is_pivot = False
                    break
        
        if is_pivot:
            result[i + right_bars] = center
    
    return result


# =============================================================================
# CROSSOVERS
# =============================================================================

def cross(src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
    """Cross in either direction - ta.cross(src1, src2)"""
    arr1 = ensure_array(src1)
    arr2 = ensure_array(src2)
    
    prev1 = np.roll(arr1, 1)
    prev2 = np.roll(arr2, 1)
    prev1[0] = arr1[0]
    prev2[0] = arr2[0]
    
    cross_over = (prev1 <= prev2) & (arr1 > arr2)
    cross_under = (prev1 >= prev2) & (arr1 < arr2)
    
    return cross_over | cross_under


def crossover(src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
    """Cross over - ta.crossover(src1, src2)"""
    arr1 = ensure_array(src1)
    arr2 = ensure_array(src2)
    
    prev1 = np.roll(arr1, 1)
    prev2 = np.roll(arr2, 1)
    prev1[0] = arr1[0]
    prev2[0] = arr2[0]
    
    return (prev1 <= prev2) & (arr1 > arr2)


def crossunder(src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
    """Cross under - ta.crossunder(src1, src2)"""
    arr1 = ensure_array(src1)
    arr2 = ensure_array(src2)
    
    prev1 = np.roll(arr1, 1)
    prev2 = np.roll(arr2, 1)
    prev1[0] = arr1[0]
    prev2[0] = arr2[0]
    
    return (prev1 >= prev2) & (arr1 < arr2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def change(src: ArrayLike, length: int = 1) -> np.ndarray:
    """Change from N bars ago - ta.change(src, length)"""
    arr = ensure_array(src)
    prev = np.roll(arr, length)
    prev[:length] = arr[:length]
    return arr - prev


def roc(src: ArrayLike, length: int) -> np.ndarray:
    """Rate of Change - ta.roc(src, length)"""
    arr = ensure_array(src)
    prev = np.roll(arr, length)
    prev[:length] = np.nan
    return np.where(prev != 0, 100 * (arr - prev) / prev, 0)


def mom(src: ArrayLike, length: int) -> np.ndarray:
    """Momentum - ta.mom(src, length)"""
    return change(src, length)


def nz(src: ArrayLike, replacement: float = 0.0) -> np.ndarray:
    """Replace NaN with value - nz(src, replacement)"""
    arr = ensure_array(src)
    return np.where(np.isnan(arr), replacement, arr)


def na(src: ArrayLike) -> np.ndarray:
    """Check if NaN - na(src)"""
    arr = ensure_array(src)
    return np.isnan(arr)


def barssince(condition: ArrayLike) -> np.ndarray:
    """Bars since condition was true - ta.barssince(condition)"""
    cond = ensure_array(condition).astype(bool)
    result = np.zeros(len(cond))
    
    bars_since = np.nan
    for i in range(len(cond)):
        if cond[i]:
            bars_since = 0
        elif not np.isnan(bars_since):
            bars_since += 1
        result[i] = bars_since
    
    return result


def valuewhen(condition: ArrayLike, src: ArrayLike, occurrence: int = 0) -> np.ndarray:
    """Value when condition was true - ta.valuewhen(condition, src, occurrence)"""
    cond = ensure_array(condition).astype(bool)
    arr = ensure_array(src)
    result = np.full(len(arr), np.nan)
    
    for i in range(len(arr)):
        count = 0
        for j in range(i, -1, -1):
            if cond[j]:
                if count == occurrence:
                    result[i] = arr[j]
                    break
                count += 1
    
    return result


def cum(src: ArrayLike) -> np.ndarray:
    """Cumulative sum - ta.cum(src)"""
    arr = ensure_array(src)
    return np.nancumsum(arr)


def sum_func(src: ArrayLike, length: int) -> np.ndarray:
    """Sum over period - math.sum(src, length)"""
    series = ensure_series(src)
    return series.rolling(window=length, min_periods=1).sum().values


# =============================================================================
# MATH FUNCTIONS (match Pine Script math.* namespace)
# =============================================================================

def math_abs(src: ArrayLike) -> np.ndarray:
    """Absolute value - math.abs(src)"""
    return np.abs(ensure_array(src))


def math_max(src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
    """Maximum - math.max(src1, src2)"""
    return np.maximum(ensure_array(src1), ensure_array(src2))


def math_min(src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
    """Minimum - math.min(src1, src2)"""
    return np.minimum(ensure_array(src1), ensure_array(src2))


def math_log(src: ArrayLike) -> np.ndarray:
    """Natural logarithm - math.log(src)"""
    arr = ensure_array(src)
    return np.where(arr > 0, np.log(arr), np.nan)


def math_exp(src: ArrayLike) -> np.ndarray:
    """Exponential - math.exp(src)"""
    return np.exp(ensure_array(src))


def math_sqrt(src: ArrayLike) -> np.ndarray:
    """Square root - math.sqrt(src)"""
    arr = ensure_array(src)
    return np.where(arr >= 0, np.sqrt(arr), np.nan)


def math_pow(base: ArrayLike, exp: float) -> np.ndarray:
    """Power - math.pow(base, exp)"""
    return np.power(ensure_array(base), exp)


def math_sign(src: ArrayLike) -> np.ndarray:
    """Sign - math.sign(src)"""
    return np.sign(ensure_array(src))


def math_round(src: ArrayLike, precision: int = 0) -> np.ndarray:
    """Round - math.round(src, precision)"""
    return np.round(ensure_array(src), precision)


def math_floor(src: ArrayLike) -> np.ndarray:
    """Floor - math.floor(src)"""
    return np.floor(ensure_array(src))


def math_ceil(src: ArrayLike) -> np.ndarray:
    """Ceiling - math.ceil(src)"""
    return np.ceil(ensure_array(src))


# =============================================================================
# INDICATOR CLASS - Wraps OHLCV data for convenience
# =============================================================================

class TAContext:
    """Context for running technical analysis with OHLCV data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame containing timestamp, open, high, low, close, volume."""
        self.df = df
        self.open = df['open'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.close = df['close'].values
        self.volume = df['volume'].values
        self.timestamp = df['timestamp'].values if 'timestamp' in df.columns else None
        self.length = len(df)
    
    def sma(self, src: ArrayLike, length: int) -> np.ndarray:
        return sma(src, length)
    
    def ema(self, src: ArrayLike, length: int) -> np.ndarray:
        return ema(src, length)
    
    def atr(self, length: int) -> np.ndarray:
        return atr(self.high, self.low, self.close, length)
    
    def rsi(self, src: ArrayLike, length: int) -> np.ndarray:
        return rsi(src, length)
    
    def stdev(self, src: ArrayLike, length: int) -> np.ndarray:
        return stdev(src, length)
    
    def highest(self, src: ArrayLike, length: int) -> np.ndarray:
        return highest(src, length)
    
    def lowest(self, src: ArrayLike, length: int) -> np.ndarray:
        return lowest(src, length)
    
    def pivothigh(self, left: int, right: int) -> np.ndarray:
        return pivothigh(self.high, left, right)
    
    def pivotlow(self, left: int, right: int) -> np.ndarray:
        return pivotlow(self.low, left, right)
    
    def tr(self) -> np.ndarray:
        return tr(self.high, self.low, self.close)
    
    def mfi(self, length: int) -> np.ndarray:
        return mfi(self.high, self.low, self.close, self.volume, length)
    
    def crossover(self, src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
        return crossover(src1, src2)
    
    def crossunder(self, src1: ArrayLike, src2: ArrayLike) -> np.ndarray:
        return crossunder(src1, src2)


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    print("SMA(14):", sma(close, 14)[-5:])
    print("EMA(14):", ema(close, 14)[-5:])
    print("RSI(14):", rsi(close, 14)[-5:])
    print("ATR(14):", atr(high, low, close, 14)[-5:])
    print("STDEV(14):", stdev(close, 14)[-5:])

