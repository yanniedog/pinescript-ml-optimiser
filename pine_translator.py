"""
Pine Script to Python Translator
Translates Pine Script indicator logic to executable Python code.
"""

import re
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass

from pine_parser import ParseResult, Parameter, SignalType
import ta_functions as ta

# Suppress numpy divide warnings - we handle them with np.where
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Result of running an indicator on price data."""
    main_values: np.ndarray          # Main indicator line values
    buy_signals: np.ndarray          # Boolean array for buy signals
    sell_signals: np.ndarray         # Boolean array for sell signals
    combined_signal: np.ndarray      # Combined signal strength (-1 to 1)
    extra_values: Dict[str, np.ndarray] = None  # Additional indicator lines


def safe_divide(a, b, default=0.0):
    """Safe division that returns default when dividing by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result = np.where(np.isfinite(result), result, default)
    return result


class PineTranslator:
    """Translates Pine Script to executable Python indicator functions."""
    
    # Translation mappings for Pine Script to Python
    TA_MAPPINGS = {
        'ta.sma': 'ta.sma',
        'ta.ema': 'ta.ema',
        'ta.wma': 'ta.wma',
        'ta.rma': 'ta.rma',
        'ta.atr': 'self._atr',
        'ta.rsi': 'ta.rsi',
        'ta.stdev': 'ta.stdev',
        'ta.variance': 'ta.variance',
        'ta.highest': 'ta.highest',
        'ta.lowest': 'ta.lowest',
        'ta.highestbars': 'ta.highest_bars',
        'ta.lowestbars': 'ta.lowest_bars',
        'ta.pivothigh': 'self._pivothigh',
        'ta.pivotlow': 'self._pivotlow',
        'ta.cross': 'ta.cross',
        'ta.crossover': 'ta.crossover',
        'ta.crossunder': 'ta.crossunder',
        'ta.change': 'ta.change',
        'ta.roc': 'ta.roc',
        'ta.mom': 'ta.mom',
        'ta.mfi': 'self._mfi',
        'ta.cci': 'self._cci',
        'ta.cum': 'ta.cum',
        'ta.barssince': 'ta.barssince',
        'ta.valuewhen': 'ta.valuewhen',
        'ta.tr': 'self._tr',
    }
    
    MATH_MAPPINGS = {
        'math.abs': 'np.abs',
        'math.max': 'np.maximum',
        'math.min': 'np.minimum',
        'math.log': 'np.log',
        'math.exp': 'np.exp',
        'math.sqrt': 'np.sqrt',
        'math.pow': 'np.power',
        'math.sign': 'np.sign',
        'math.round': 'np.round',
        'math.floor': 'np.floor',
        'math.ceil': 'np.ceil',
    }
    
    VARIABLE_MAPPINGS = {
        'close': 'self.close',
        'open': 'self.open_',
        'high': 'self.high',
        'low': 'self.low',
        'volume': 'self.volume',
        'hl2': '(self.high + self.low) / 2',
        'hlc3': '(self.high + self.low + self.close) / 3',
        'ohlc4': '(self.open_ + self.high + self.low + self.close) / 4',
        'bar_index': 'self.bar_index',
    }
    
    def __init__(self, parse_result: ParseResult, df: pd.DataFrame):
        self.parse_result = parse_result
        self.df = df
        self.params = {p.name: p.default for p in parse_result.parameters}
        
        # OHLCV data
        self.open_ = df['open'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.close = df['close'].values
        self.volume = df['volume'].values
        self.bar_index = np.arange(len(df))
        self.length = len(df)
        
        # State for indicator calculations
        self._cache = {}
    
    def update_params(self, new_params: Dict[str, Any]):
        """Update parameters for a new optimization trial."""
        self.params.update(new_params)
        self._cache.clear()
    
    def _tr(self) -> np.ndarray:
        """True range helper."""
        return ta.tr(self.high, self.low, self.close)
    
    def _atr(self, length: int) -> np.ndarray:
        """ATR helper."""
        return ta.atr(self.high, self.low, self.close, length)
    
    def _pivothigh(self, left: int, right: int) -> np.ndarray:
        """Pivot high helper."""
        return ta.pivothigh(self.high, left, right)
    
    def _pivotlow(self, left: int, right: int) -> np.ndarray:
        """Pivot low helper."""
        return ta.pivotlow(self.low, left, right)
    
    def _mfi(self, length: int) -> np.ndarray:
        """MFI helper."""
        return ta.mfi(self.high, self.low, self.close, self.volume, length)
    
    def _cci(self, length: int) -> np.ndarray:
        """CCI helper."""
        return ta.cci(self.high, self.low, self.close, length)
    
    def run_indicator(self, params: Optional[Dict[str, Any]] = None) -> IndicatorResult:
        """
        Run the indicator with given parameters and return results.
        This is the main entry point for the optimizer.
        """
        if params:
            self.update_params(params)
        
        # Determine which indicator type we're dealing with
        indicator_name = self.parse_result.indicator_name.lower()
        
        # Try to match known indicator patterns
        if 'mfv' in indicator_name or 'money flow' in indicator_name.lower():
            return self._run_mfv_indicator()
        elif 'wisdom' in indicator_name or 'gauge' in indicator_name:
            return self._run_wisdom_gauge_indicator()
        else:
            # Generic indicator - use signal detection
            return self._run_generic_indicator()
    
    def _run_mfv_indicator(self) -> IndicatorResult:
        """
        Run MFV (Money Flow Volume) style indicator.
        Based on EyeX_MFV_v5.pine structure.
        """
        # Get parameters with defaults
        mfv_threshold = self.params.get('mfvThreshold', 0)
        bar_range1 = int(self.params.get('barRange1', 50))
        bar_range2 = int(self.params.get('barRange2', 75))
        bar_range3 = int(self.params.get('barRange3', 100))
        bar_range4 = int(self.params.get('barRange4', 200))
        pivot_lookback = int(self.params.get('pivotLookback', 5))
        price_proximity = self.params.get('priceProximity', 0.00001)
        
        # MFV Calculation
        rng = self.high - self.low
        mf_multiplier = safe_divide(
            (self.close - self.low) - (self.high - self.close),
            rng,
            0.0
        )
        raw_mf_volume = mf_multiplier * self.volume
        
        # Apply threshold
        if mfv_threshold > 0:
            mf_volume = np.where(np.abs(raw_mf_volume) < mfv_threshold, 0.0, raw_mf_volume)
        else:
            mf_volume = raw_mf_volume
        
        # Cumulative MFV for each range
        def calc_cum_mfv(mfv, bar_range):
            result = np.zeros(len(mfv))
            for i in range(len(mfv)):
                if i == 0:
                    result[i] = mfv[i]
                else:
                    subtract = mfv[i - bar_range] if i >= bar_range else 0
                    result[i] = result[i-1] + mfv[i] - subtract
            return result
        
        def normalize_zscore(data, window):
            mean = ta.sma(data, window)
            std = ta.stdev(data, window)
            return safe_divide(data - mean, std, 0.0)
        
        cum_mfv1 = calc_cum_mfv(mf_volume, bar_range1)
        norm_mfv1 = normalize_zscore(cum_mfv1, bar_range1)
        mfv_line1 = np.clip(norm_mfv1 * 10, -100, 100)
        
        cum_mfv2 = calc_cum_mfv(mf_volume, bar_range2)
        norm_mfv2 = normalize_zscore(cum_mfv2, bar_range2)
        mfv_line2 = np.clip(norm_mfv2 * 10, -100, 100)
        
        cum_mfv3 = calc_cum_mfv(mf_volume, bar_range3)
        norm_mfv3 = normalize_zscore(cum_mfv3, bar_range3)
        mfv_line3 = np.clip(norm_mfv3 * 10, -100, 100)
        
        cum_mfv4 = calc_cum_mfv(mf_volume, bar_range4)
        norm_mfv4 = normalize_zscore(cum_mfv4, bar_range4)
        mfv_line4 = np.clip(norm_mfv4 * 10, -100, 100)
        
        # Combined line
        combined_mfv = mfv_line1 + mfv_line2 + mfv_line3 + mfv_line4
        
        # Simple signal generation based on combined MFV
        # Positive = bullish, Negative = bearish
        buy_signals = ta.crossover(combined_mfv, np.zeros_like(combined_mfv))
        sell_signals = ta.crossunder(combined_mfv, np.zeros_like(combined_mfv))
        
        # Normalize combined signal to -1 to 1
        max_val = np.nanmax(np.abs(combined_mfv))
        if max_val > 0:
            combined_signal = combined_mfv / max_val
        else:
            combined_signal = np.zeros_like(combined_mfv)
        
        return IndicatorResult(
            main_values=combined_mfv,
            buy_signals=buy_signals.astype(bool),
            sell_signals=sell_signals.astype(bool),
            combined_signal=combined_signal,
            extra_values={
                'mfv_line1': mfv_line1,
                'mfv_line2': mfv_line2,
                'mfv_line3': mfv_line3,
                'mfv_line4': mfv_line4
            }
        )
    
    def _run_wisdom_gauge_indicator(self) -> IndicatorResult:
        """
        Run Wisdom Gauge style indicator.
        Based on pso_indicator.pine structure.
        """
        # Get parameters
        atr_len = int(self.params.get('atrLen', 23))
        vol_len = int(self.params.get('volLen', 119))
        eff_len = int(self.params.get('effLen', 91))
        smooth_f = int(self.params.get('smoothF', 33))
        smooth_gauge = int(self.params.get('smoothGauge', 12))
        fast_len = int(self.params.get('fastLen', 24))
        slow_len = int(self.params.get('slowLen', 43))
        wick_w = self.params.get('wickW', 1.47)
        tau = self.params.get('tau', 0.64)
        alpha = self.params.get('alpha', 5.3)
        k_scale = self.params.get('kScale', 4.14)
        norm_window = int(self.params.get('normWindow', 243))
        confirm_len = int(self.params.get('confirmLen', 11))
        cooldown_bars = int(self.params.get('cooldownBars', 45))
        buy_level = self.params.get('buyLevel', -23)
        sell_level = self.params.get('sellLevel', 36)
        
        eps = 1e-10
        
        def f_tanh(x):
            xc = np.clip(x, -20.0, 20.0)
            e2 = np.exp(2.0 * xc)
            return (e2 - 1.0) / (e2 + 1.0)
        
        def f_norm_pm1(x, window):
            m = ta.highest(np.abs(x), window)
            return np.clip(np.where(m > 0, x / (m + eps), 0), -1.0, 1.0)
        
        # ATR and volume MA
        atr_vals = self._atr(atr_len)
        vol_ma = ta.ema(self.volume, vol_len)
        rng = self.high - self.low
        
        # LWCP calculation
        clv = safe_divide(2.0 * self.close - self.high - self.low, rng, 0.0)
        uw = self.high - np.maximum(self.open_, self.close)
        lw = np.minimum(self.open_, self.close) - self.low
        wick_bias = safe_divide(lw - uw, rng, 0.0)
        
        raw = clv + wick_w * wick_bias
        lwcp = raw * f_tanh(safe_divide(rng, atr_vals, 0.0)) * f_tanh(safe_divide(self.volume, vol_ma, 0.0))
        
        lwcp_fast = ta.ema(lwcp, fast_len)
        lwcp_slow = ta.ema(lwcp, slow_len)
        lwcp_osc = lwcp_fast - lwcp_slow
        lwcp_n = f_norm_pm1(lwcp_osc, norm_window)
        
        # CPVF calculation
        clv2 = safe_divide((self.close - self.low) - (self.high - self.close), rng, 0.0)
        vol_ratio = safe_divide(self.volume, vol_ma, 1.0)
        vol_ratio = np.maximum(vol_ratio, eps)  # Ensure positive for log
        rv = np.log(vol_ratio)
        P = clv2 * f_tanh(rv)
        
        # Efficiency
        net = np.abs(self.close - np.roll(self.close, eff_len))
        net[:eff_len] = 0
        avg = ta.sma(np.abs(ta.change(self.close, 1)), eff_len)
        E = safe_divide(net, avg * eff_len + eps, 0.0)
        
        w = f_tanh(alpha * (E - tau))
        sigma = safe_divide(atr_vals, self.close, 0.0)
        
        horizon_h = 1
        rhat = k_scale * sigma * np.sqrt(horizon_h) * (2.0 * w - 1.0) * P
        rhat_s = ta.ema(rhat, smooth_f)
        rhat_n = f_norm_pm1(rhat_s, norm_window)
        
        # Confidence
        vol_conf = np.clip(np.abs(rv) / 0.6, 0.0, 1.0)
        reg_conf = np.clip((E - 0.25) / 0.35, 0.0, 1.0)
        
        # Gauge synthesis
        pressure = lwcp_n
        forecast_sig = -rhat_n
        conf_mult = 0.7 + 0.3 * reg_conf
        raw_gauge = (0.6 * pressure + 0.3 * forecast_sig + 0.1 * vol_conf * np.sign(pressure)) * conf_mult
        
        gauge_scaled = 100.0 * raw_gauge
        gauge = ta.ema(gauge_scaled, smooth_gauge)
        gauge_c = np.clip(gauge, -100.0, 100.0)
        
        # Signal generation
        is_local_min = gauge_c == ta.lowest(gauge_c, confirm_len)
        is_local_max = gauge_c == ta.highest(gauge_c, confirm_len)
        
        lwcp_reversing_up = (lwcp_osc < 0) & (lwcp_osc > np.roll(lwcp_osc, 1))
        lwcp_reversing_down = (lwcp_osc > 0) & (lwcp_osc < np.roll(lwcp_osc, 1))
        gauge_recovering = gauge_c > np.roll(gauge_c, 1)
        gauge_rolling_over = gauge_c < np.roll(gauge_c, 1)
        
        buy_setup = (gauge_c <= buy_level) & is_local_min & (lwcp_reversing_up | gauge_recovering)
        sell_setup = (gauge_c >= sell_level) & is_local_max & (lwcp_reversing_down | gauge_rolling_over)
        
        # Apply cooldown
        buy_signals = np.zeros(len(gauge_c), dtype=bool)
        sell_signals = np.zeros(len(gauge_c), dtype=bool)
        
        last_buy = -cooldown_bars - 1
        last_sell = -cooldown_bars - 1
        
        for i in range(1, len(gauge_c)):
            if buy_setup[i] and not buy_setup[i-1] and (i - last_buy >= cooldown_bars):
                buy_signals[i] = True
                last_buy = i
            if sell_setup[i] and not sell_setup[i-1] and (i - last_sell >= cooldown_bars):
                sell_signals[i] = True
                last_sell = i
        
        # Normalize gauge to -1 to 1
        combined_signal = gauge_c / 100.0
        
        return IndicatorResult(
            main_values=gauge_c,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            combined_signal=combined_signal,
            extra_values={
                'lwcp_osc': lwcp_osc,
                'rhat_n': rhat_n,
                'efficiency': E
            }
        )
    
    def _run_generic_indicator(self) -> IndicatorResult:
        """
        Run a generic indicator by attempting to parse and execute its logic.
        Falls back to directional signal mode based on main indicator value.
        """
        signal_info = self.parse_result.signal_info
        
        # Try to compute a simple composite indicator
        # based on detected calculations
        main_values = np.zeros(self.length)
        
        # Check for common patterns in the content
        content = self.parse_result.raw_content.lower()
        
        if 'ta.rsi' in content:
            rsi_len = int(self.params.get('rsiLength', self.params.get('length', 14)))
            main_values = ta.rsi(self.close, rsi_len) - 50  # Center around 0
        elif 'ta.macd' in content:
            fast = int(self.params.get('fastLength', 12))
            slow = int(self.params.get('slowLength', 26))
            signal = int(self.params.get('signalLength', 9))
            macd_line, _, histogram = ta.macd(self.close, fast, slow, signal)
            main_values = histogram
        elif 'ta.cci' in content:
            cci_len = int(self.params.get('cciLength', self.params.get('length', 20)))
            main_values = self._cci(cci_len)
        elif 'ta.mfi' in content:
            mfi_len = int(self.params.get('mfiLength', self.params.get('length', 14)))
            main_values = self._mfi(mfi_len) - 50  # Center around 0
        else:
            # Default: use price momentum
            mom_len = int(self.params.get('length', 14))
            main_values = ta.roc(self.close, mom_len)
        
        # Generate signals based on signal type
        if signal_info.signal_type == SignalType.THRESHOLD:
            buy_level = signal_info.threshold_levels.get('buy', -20)
            sell_level = signal_info.threshold_levels.get('sell', 20)
            buy_signals = ta.crossover(main_values, np.full(self.length, buy_level))
            sell_signals = ta.crossunder(main_values, np.full(self.length, sell_level))
        else:
            # Directional: cross zero
            zero_line = np.zeros(self.length)
            buy_signals = ta.crossover(main_values, zero_line)
            sell_signals = ta.crossunder(main_values, zero_line)
        
        # Normalize
        max_val = np.nanmax(np.abs(main_values))
        if max_val > 0:
            combined_signal = np.clip(main_values / max_val, -1, 1)
        else:
            combined_signal = np.zeros(self.length)
        
        return IndicatorResult(
            main_values=main_values,
            buy_signals=buy_signals.astype(bool),
            sell_signals=sell_signals.astype(bool),
            combined_signal=combined_signal
        )


def create_indicator_runner(parse_result: ParseResult, df: pd.DataFrame) -> Callable:
    """
    Create a callable that runs the indicator with given parameters.
    Returns a function that takes a params dict and returns IndicatorResult.
    """
    translator = PineTranslator(parse_result, df)
    
    def run(params: Dict[str, Any]) -> IndicatorResult:
        return translator.run_indicator(params)
    
    return run


if __name__ == "__main__":
    # Test with sample data
    from pine_parser import parse_pine_script
    import sys
    
    if len(sys.argv) > 1:
        # Parse the Pine Script
        result = parse_pine_script(sys.argv[1])
        
        # Create sample data
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.2
        volume = np.random.randint(1000, 10000, n).astype(float)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n, freq='1H'),
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # Run indicator
        translator = PineTranslator(result, df)
        indicator_result = translator.run_indicator()
        
        print(f"\nIndicator: {result.indicator_name}")
        print(f"Main values (last 5): {indicator_result.main_values[-5:]}")
        print(f"Buy signals: {np.sum(indicator_result.buy_signals)}")
        print(f"Sell signals: {np.sum(indicator_result.sell_signals)}")
    else:
        print("Usage: python pine_translator.py <pine_script_file>")

