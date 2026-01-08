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
    
    def _get_param(self, *names, default=None, param_type=None):
        """
        Look up a parameter by trying multiple possible names.
        NEVER uses hardcoded defaults if a matching parameter exists in the Pine Script.
        
        Args:
            *names: Possible parameter names to try (case-insensitive fuzzy match)
            default: Fallback value ONLY if no parameter found at all
            param_type: 'int', 'float', or None for auto-detect
            
        Returns:
            The parameter value from self.params, or default if not found
        """
        # First, try exact matches
        for name in names:
            if name in self.params:
                val = self.params[name]
                if param_type == 'int':
                    return int(val)
                elif param_type == 'float':
                    return float(val)
                return val
        
        # Second, try case-insensitive matches
        params_lower = {k.lower(): k for k in self.params}
        for name in names:
            if name.lower() in params_lower:
                val = self.params[params_lower[name.lower()]]
                if param_type == 'int':
                    return int(val)
                elif param_type == 'float':
                    return float(val)
                return val
        
        # Third, try fuzzy matches (remove underscores, handle camelCase vs snake_case)
        def normalize(s):
            # Convert to lowercase, remove underscores
            return s.lower().replace('_', '').replace('-', '')
        
        params_normalized = {normalize(k): k for k in self.params}
        for name in names:
            norm_name = normalize(name)
            if norm_name in params_normalized:
                val = self.params[params_normalized[norm_name]]
                if param_type == 'int':
                    return int(val)
                elif param_type == 'float':
                    return float(val)
                return val
            
            # Try partial matches (e.g., 'length' matches 'rsiLength')
            for norm_key, orig_key in params_normalized.items():
                if norm_name in norm_key or norm_key in norm_name:
                    val = self.params[orig_key]
                    if param_type == 'int':
                        return int(val)
                    elif param_type == 'float':
                        return float(val)
                    return val
        
        # No match found - use default but log warning
        if default is not None:
            logger.debug(f"Parameter not found for {names}, using default: {default}")
        
        if param_type == 'int' and default is not None:
            return int(default)
        elif param_type == 'float' and default is not None:
            return float(default)
        return default
    
    def _get_all_int_params(self) -> Dict[str, int]:
        """Get all integer parameters from the parsed Pine Script."""
        result = {}
        for p in self.parse_result.parameters:
            if p.param_type == 'int':
                result[p.name] = int(self.params.get(p.name, p.default))
        return result
    
    def _get_all_float_params(self) -> Dict[str, float]:
        """Get all float parameters from the parsed Pine Script."""
        result = {}
        for p in self.parse_result.parameters:
            if p.param_type == 'float':
                result[p.name] = float(self.params.get(p.name, p.default))
        return result
    
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
        
        # Verify all parsed parameters are available in self.params
        for p in self.parse_result.parameters:
            if p.name not in self.params:
                logger.warning(f"Parameter '{p.name}' from Pine Script not in params dict!")
            else:
                logger.debug(f"Using parameter {p.name}={self.params[p.name]} (default was {p.default})")
        
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

    @staticmethod
    def _normalize_signal_causal(values: np.ndarray) -> np.ndarray:
        """Causally normalize a signal using an expanding max of absolute values."""
        abs_vals = np.abs(values)
        abs_vals = np.nan_to_num(abs_vals, nan=0.0, posinf=0.0, neginf=0.0)
        running_max = np.maximum.accumulate(abs_vals)
        running_max = np.where(running_max > 0, running_max, 1.0)
        normalized = values / running_max
        return np.clip(normalized, -1.0, 1.0)
    
    def _run_mfv_indicator(self) -> IndicatorResult:
        """
        Run MFV (Money Flow Volume) style indicator.
        Based on EyeX_MFV_v5.pine structure.
        """
        # Get parameters - use _get_param to respect Pine Script config
        mfv_threshold = self._get_param('mfvThreshold', 'mfv_threshold', 'threshold', default=0, param_type='float')
        bar_range1 = self._get_param('barRange1', 'bar_range1', 'range1', default=50, param_type='int')
        bar_range2 = self._get_param('barRange2', 'bar_range2', 'range2', default=75, param_type='int')
        bar_range3 = self._get_param('barRange3', 'bar_range3', 'range3', default=100, param_type='int')
        bar_range4 = self._get_param('barRange4', 'bar_range4', 'range4', default=200, param_type='int')
        pivot_lookback = self._get_param('pivotLookback', 'pivot_lookback', 'lookback', default=5, param_type='int')
        price_proximity = self._get_param('priceProximity', 'price_proximity', 'proximity', default=0.00001, param_type='float')
        
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
        
        # Normalize combined signal to -1 to 1 using causal scaling
        combined_signal = self._normalize_signal_causal(combined_mfv)
        
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
        # Get parameters - use _get_param to respect Pine Script config
        atr_len = self._get_param('atrLen', 'atr_len', 'atrLength', default=23, param_type='int')
        vol_len = self._get_param('volLen', 'vol_len', 'volLength', default=119, param_type='int')
        eff_len = self._get_param('effLen', 'eff_len', 'effLength', default=91, param_type='int')
        smooth_f = self._get_param('smoothF', 'smooth_f', 'smoothFactor', default=33, param_type='int')
        smooth_gauge = self._get_param('smoothGauge', 'smooth_gauge', default=12, param_type='int')
        fast_len = self._get_param('fastLen', 'fast_len', 'fastLength', default=24, param_type='int')
        slow_len = self._get_param('slowLen', 'slow_len', 'slowLength', default=43, param_type='int')
        wick_w = self._get_param('wickW', 'wick_w', 'wickWeight', default=1.47, param_type='float')
        tau = self._get_param('tau', 'tauValue', default=0.64, param_type='float')
        alpha = self._get_param('alpha', 'alphaValue', default=5.3, param_type='float')
        k_scale = self._get_param('kScale', 'k_scale', 'kScaleFactor', default=4.14, param_type='float')
        norm_window = self._get_param('normWindow', 'norm_window', 'normLength', default=243, param_type='int')
        confirm_len = self._get_param('confirmLen', 'confirm_len', 'confirmLength', default=11, param_type='int')
        cooldown_bars = self._get_param('cooldownBars', 'cooldown_bars', 'cooldown', default=45, param_type='int')
        buy_level = self._get_param('buyLevel', 'buy_level', 'buyThreshold', default=-23, param_type='float')
        sell_level = self._get_param('sellLevel', 'sell_level', 'sellThreshold', default=36, param_type='float')
        
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
        indicator_name = self.parse_result.indicator_name.lower()
        
        # Check if this looks like a MACD indicator (by name or content)
        is_macd = ('macd' in indicator_name or 
                   ('ta.ema' in content and 'fastema' in content.replace(' ', '').replace('_', '')) or
                   ('ta.ema' in content and 'slowema' in content.replace(' ', '').replace('_', '')) or
                   ('fastlen' in content.replace(' ', '').replace('_', '') and 'slowlen' in content.replace(' ', '').replace('_', '')))
        
        if 'ta.rsi' in content:
            rsi_len = self._get_param('rsiLength', 'rsi_length', 'length', 'len', 'period', default=14, param_type='int')
            main_values = ta.rsi(self.close, rsi_len) - 50  # Center around 0
        elif is_macd or 'ta.macd' in content:
            # Use _get_param to find MACD parameters with various naming conventions
            fast = self._get_param('fastLen', 'fastLength', 'fast', 'fastEmaLen', 'fast_len', default=12, param_type='int')
            slow = self._get_param('slowLen', 'slowLength', 'slow', 'slowEmaLen', 'slow_len', default=26, param_type='int')
            signal = self._get_param('signalLen', 'signalLength', 'signal', 'signalSmaLen', 'signal_len', default=9, param_type='int')
            
            # Ensure fast < slow
            if fast >= slow:
                fast, slow = min(fast, slow - 1), max(fast + 1, slow)
            
            macd_line, signal_line, histogram = ta.macd(self.close, fast, slow, signal)
            main_values = histogram
        elif 'ta.cci' in content:
            cci_len = self._get_param('cciLength', 'cci_length', 'length', 'len', 'period', default=20, param_type='int')
            main_values = self._cci(cci_len)
        elif 'ta.mfi' in content:
            mfi_len = self._get_param('mfiLength', 'mfi_length', 'length', 'len', 'period', default=14, param_type='int')
            main_values = self._mfi(mfi_len) - 50  # Center around 0
        else:
            # Default: use price momentum with any 'length' parameter we can find
            mom_len = self._get_param('length', 'len', 'period', 'lookback', default=14, param_type='int')
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
        
        # Normalize using causal scaling (avoids global lookahead)
        combined_signal = self._normalize_signal_causal(main_values)
        
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
