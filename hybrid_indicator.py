"""
Hybrid Indicator Generator

Creates hybrid/ensemble indicators by combining multiple optimized indicators
and their optimal parameter values.

IMPORTANT CONSTRAINT:
All indicators in a hybrid MUST have been optimized against the SAME symbol
and timeframe combination. Combining indicators optimized on different data
would produce invalid/misleading results since their parameters were tuned
for different market conditions.

For example:
- VALID: Hybrid of RSI, MACD, and Bollinger all optimized on BTCUSDT @ 1h
- INVALID: Hybrid mixing BTCUSDT @ 1h indicators with ETHUSDT @ 4h indicators
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class HybridIndicatorGenerator:
    """
    Generates hybrid Pine Script indicators that combine multiple optimized indicators.
    
    The hybrid indicator uses a voting/ensemble approach where multiple indicator
    signals are combined to produce a unified signal.
    """
    
    def __init__(self):
        self.indicator_results: List[Dict[str, Any]] = []
        self.hybrid_counter = 0
    
    def add_optimized_indicator(
        self,
        indicator_name: str,
        file_path: str,
        best_params: Dict[str, Any],
        original_params: Dict[str, Any],
        metrics: Dict[str, float],
        score: float,
        pine_content: Optional[str] = None
    ):
        """
        Add an optimized indicator to the hybrid pool.
        
        Args:
            indicator_name: Name of the indicator
            file_path: Path to the original Pine Script file
            best_params: Optimized parameter values
            original_params: Original parameter values
            metrics: Performance metrics from optimization
            score: Overall objective score
            pine_content: Optional raw Pine Script content
        """
        self.indicator_results.append({
            "name": indicator_name,
            "file_path": file_path,
            "best_params": best_params.copy(),
            "original_params": original_params.copy(),
            "metrics": metrics.copy() if metrics else {},
            "score": score,
            "pine_content": pine_content,
        })
        logger.info(f"Added indicator '{indicator_name}' to hybrid pool (score: {score:.4f})")
    
    def get_indicator_count(self) -> int:
        """Return number of indicators in the hybrid pool."""
        return len(self.indicator_results)
    
    def clear(self):
        """Clear all indicators from the hybrid pool."""
        self.indicator_results.clear()
        logger.info("Cleared hybrid indicator pool")
    
    def get_top_indicators(self, n: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get top N indicators by score for hybrid generation.
        
        Args:
            n: Number of top indicators to return
            min_score: Minimum score threshold
            
        Returns:
            List of top indicator results
        """
        filtered = [r for r in self.indicator_results if r["score"] >= min_score]
        sorted_results = sorted(filtered, key=lambda x: x["score"], reverse=True)
        return sorted_results[:n]
    
    def generate_hybrid_indicator(
        self,
        n_indicators: int = 5,
        min_score: float = 0.0,
        voting_method: str = "majority",
        hybrid_name: Optional[str] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Generate a hybrid Pine Script indicator combining top performers.
        
        Args:
            n_indicators: Number of top indicators to include
            min_score: Minimum score threshold for inclusion
            voting_method: How to combine signals ("majority", "unanimous", "weighted")
            hybrid_name: Optional custom name for the hybrid
            symbol: Optional symbol context for the hybrid
            interval: Optional interval context for the hybrid
            
        Returns:
            Tuple of (pine_script_content, suggested_filename, hybrid_metadata)
        """
        top_indicators = self.get_top_indicators(n=n_indicators, min_score=min_score)
        
        if len(top_indicators) < 2:
            raise ValueError(f"Need at least 2 indicators for hybrid, got {len(top_indicators)}")
        
        self.hybrid_counter += 1
        
        # Generate unique hybrid name
        context_suffix = ""
        if symbol and interval:
            context_suffix = f"_{symbol}_{interval}"
        elif symbol:
            context_suffix = f"_{symbol}"
        elif interval:
            context_suffix = f"_{interval}"
            
        if hybrid_name:
            final_name = hybrid_name
        else:
            final_name = f"Hybrid_Ensemble_{self.hybrid_counter}{context_suffix}"
        
        # Build the hybrid Pine Script
        pine_content = self._build_hybrid_pine_script(
            top_indicators, 
            final_name, 
            voting_method,
            symbol,
            interval
        )
        
        # Determine filename
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", final_name)
        filename = f"{safe_name}.pine"
        
        # Build metadata
        metadata = {
            "name": final_name,
            "type": "hybrid_ensemble",
            "voting_method": voting_method,
            "source_indicators": [
                {
                    "name": ind["name"],
                    "score": ind["score"],
                    "mcc": ind["metrics"].get("mcc", 0),
                    "roc_auc": ind["metrics"].get("roc_auc", 0),
                    "best_params": ind["best_params"],
                }
                for ind in top_indicators
            ],
            "symbol_context": symbol,
            "interval_context": interval,
            "generated_at": datetime.now().isoformat(),
            "indicator_count": len(top_indicators),
        }
        
        logger.info(f"Generated hybrid indicator '{final_name}' combining {len(top_indicators)} indicators")
        
        return pine_content, filename, metadata
    
    def _build_hybrid_pine_script(
        self,
        indicators: List[Dict[str, Any]],
        name: str,
        voting_method: str,
        symbol: Optional[str] = None,
        interval: Optional[str] = None
    ) -> str:
        """Build the actual Pine Script content for the hybrid indicator."""
        
        # Header
        lines = [
            "//@version=6",
            "// ===========================================================================",
            "// HYBRID ENSEMBLE INDICATOR",
            f"// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"// Voting Method: {voting_method}",
            f"// Combined Indicators: {len(indicators)}",
        ]
        
        if symbol:
            lines.append(f"// Optimized for: {symbol}")
        if interval:
            lines.append(f"// Timeframe: {interval}")
        
        lines.extend([
            "// ---------------------------------------------------------------------------",
            "// SOURCE INDICATORS (by score):",
        ])
        
        for i, ind in enumerate(indicators, 1):
            mcc = ind["metrics"].get("mcc", 0)
            auc = ind["metrics"].get("roc_auc", 0)
            lines.append(f"//   {i}. {ind['name']}: score={ind['score']:.4f}, MCC={mcc:.3f}, AUC={auc:.3f}")
        
        lines.extend([
            "// ===========================================================================",
            "",
            f'indicator("{name}", overlay=false)',
            "",
            "// =============================================================================",
            "// INPUT PARAMETERS",
            "// =============================================================================",
            "",
        ])
        
        # Generate input parameters for each indicator
        indicator_prefixes = []
        all_params = {}
        
        for idx, ind in enumerate(indicators):
            prefix = f"ind{idx + 1}"
            indicator_prefixes.append(prefix)
            indicator_short_name = self._short_name(ind["name"])
            
            # Add enable toggle for each indicator
            lines.append(f'// --- {ind["name"]} ---')
            lines.append(f'{prefix}_enabled = input.bool(true, "{indicator_short_name} Enabled", group="{indicator_short_name}")')
            
            # Add optimized parameters
            for param_name, param_value in ind["best_params"].items():
                # Skip display/visual parameters
                if self._is_display_param(param_name):
                    continue
                    
                full_param_name = f"{prefix}_{param_name}"
                
                if isinstance(param_value, bool):
                    lines.append(
                        f'{full_param_name} = input.bool({str(param_value).lower()}, "{param_name}", group="{indicator_short_name}")'
                    )
                elif isinstance(param_value, int):
                    lines.append(
                        f'{full_param_name} = input.int({param_value}, "{param_name}", group="{indicator_short_name}")'
                    )
                elif isinstance(param_value, float):
                    # Format float with appropriate precision
                    if abs(param_value) < 0.01:
                        val_str = f"{param_value:.6f}"
                    elif abs(param_value) < 1:
                        val_str = f"{param_value:.4f}"
                    else:
                        val_str = f"{param_value:.2f}"
                    lines.append(
                        f'{full_param_name} = input.float({val_str}, "{param_name}", group="{indicator_short_name}")'
                    )
                else:
                    # String or other - skip for now
                    continue
                    
                all_params[full_param_name] = param_value
            
            lines.append("")
        
        # Voting threshold parameter
        lines.extend([
            '// --- Ensemble Settings ---',
            'voting_threshold = input.float(0.5, "Voting Threshold", minval=0.0, maxval=1.0, step=0.1, group="Ensemble")',
            'show_components = input.bool(false, "Show Component Signals", group="Ensemble")',
            "",
        ])
        
        # Generate indicator calculations
        lines.extend([
            "// =============================================================================",
            "// INDICATOR CALCULATIONS",
            "// =============================================================================",
            "",
        ])
        
        # For each indicator, generate a simplified signal calculation
        # Since we don't have full Pine Script parsing, we'll create simplified versions
        # based on common indicator patterns
        
        for idx, (prefix, ind) in enumerate(zip(indicator_prefixes, indicators)):
            ind_name = ind["name"]
            params = ind["best_params"]
            
            lines.append(f"// --- Signal from {ind_name} ---")
            
            # Generate a simplified indicator signal based on common patterns
            signal_code = self._generate_indicator_signal(prefix, ind_name, params)
            lines.extend(signal_code)
            lines.append("")
        
        # Ensemble voting logic
        lines.extend([
            "// =============================================================================",
            "// ENSEMBLE VOTING",
            "// =============================================================================",
            "",
            "// Count active indicators",
            "active_count = 0",
        ])
        
        for prefix in indicator_prefixes:
            lines.append(f"active_count := active_count + ({prefix}_enabled ? 1 : 0)")
        
        lines.extend([
            "",
            "// Calculate buy/sell vote counts",
            "buy_votes = 0.0",
            "sell_votes = 0.0",
            "",
        ])
        
        for prefix in indicator_prefixes:
            lines.extend([
                f"buy_votes := buy_votes + ({prefix}_enabled and {prefix}_buy_signal ? 1.0 : 0.0)",
                f"sell_votes := sell_votes + ({prefix}_enabled and {prefix}_sell_signal ? 1.0 : 0.0)",
            ])
        
        lines.extend([
            "",
            "// Calculate vote ratios",
            "buy_ratio = active_count > 0 ? buy_votes / active_count : 0.0",
            "sell_ratio = active_count > 0 ? sell_votes / active_count : 0.0",
            "",
            "// Generate ensemble signals",
        ])
        
        if voting_method == "unanimous":
            lines.extend([
                "ensemble_buy = buy_ratio >= 1.0",
                "ensemble_sell = sell_ratio >= 1.0",
            ])
        elif voting_method == "weighted":
            lines.extend([
                "// Weighted voting uses voting_threshold as minimum ratio",
                "ensemble_buy = buy_ratio >= voting_threshold",
                "ensemble_sell = sell_ratio >= voting_threshold",
            ])
        else:  # majority
            lines.extend([
                "ensemble_buy = buy_ratio > 0.5",
                "ensemble_sell = sell_ratio > 0.5",
            ])
        
        # Output section
        lines.extend([
            "",
            "// =============================================================================",
            "// OUTPUT",
            "// =============================================================================",
            "",
            "// Plot ensemble signal strength",
            "signal_strength = buy_ratio - sell_ratio",
            'plot(signal_strength, "Ensemble Signal", color=signal_strength > 0 ? color.green : color.red)',
            'hline(0, "Neutral", color=color.gray)',
            'hline(voting_threshold, "Buy Threshold", color=color.green, linestyle=hline.style_dashed)',
            'hline(-voting_threshold, "Sell Threshold", color=color.red, linestyle=hline.style_dashed)',
            "",
            "// Plot vote ratios",
            'plot(buy_ratio, "Buy Votes", color=color.new(color.green, 50), style=plot.style_area)',
            'plot(-sell_ratio, "Sell Votes", color=color.new(color.red, 50), style=plot.style_area)',
            "",
            "// Component signal visualization (optional)",
            "if show_components",
        ])
        
        for i, prefix in enumerate(indicator_prefixes):
            y_offset = i * 0.1
            lines.append(f'    plotchar({prefix}_enabled and {prefix}_buy_signal, "Ind{i+1} Buy", "▲", location.bottom, color.green, size=size.tiny)')
            lines.append(f'    plotchar({prefix}_enabled and {prefix}_sell_signal, "Ind{i+1} Sell", "▼", location.top, color.red, size=size.tiny)')
        
        lines.extend([
            "",
            "// Alert conditions",
            'alertcondition(ensemble_buy, "Ensemble Buy", "Hybrid ensemble generated BUY signal")',
            'alertcondition(ensemble_sell, "Ensemble Sell", "Hybrid ensemble generated SELL signal")',
        ])
        
        return "\n".join(lines)
    
    def _short_name(self, name: str, max_len: int = 20) -> str:
        """Generate a short name for display."""
        # Remove common prefixes
        name = re.sub(r'^(Optimised_?|optimised_?)', '', name)
        # Truncate if needed
        if len(name) > max_len:
            return name[:max_len-3] + "..."
        return name
    
    def _is_display_param(self, param_name: str) -> bool:
        """Check if a parameter is display/visual-only."""
        display_patterns = [
            r'color', r'show', r'display', r'plot', r'line', r'fill',
            r'style', r'width', r'offset', r'transp', r'opacity'
        ]
        name_lower = param_name.lower()
        return any(re.search(p, name_lower) for p in display_patterns)
    
    def _generate_indicator_signal(
        self, 
        prefix: str, 
        indicator_name: str, 
        params: Dict[str, Any]
    ) -> List[str]:
        """
        Generate simplified signal calculation for an indicator.
        
        Since we don't have the full Pine Script code, we create generic
        signal generation based on common indicator patterns.
        """
        lines = []
        name_lower = indicator_name.lower()
        
        # Detect indicator type from name and generate appropriate signal logic
        # Get the first numeric parameter as the primary length
        length_param = None
        for param_name, value in params.items():
            if isinstance(value, int) and 'length' in param_name.lower():
                length_param = f"{prefix}_{param_name}"
                break
            elif isinstance(value, int) and 'period' in param_name.lower():
                length_param = f"{prefix}_{param_name}"
                break
        
        if length_param is None:
            # Use first int param as fallback
            for param_name, value in params.items():
                if isinstance(value, int) and value > 0 and value < 500:
                    length_param = f"{prefix}_{param_name}"
                    break
        
        # Default length if none found
        if length_param is None:
            length_param = "14"
        
        # Generate signal based on indicator type patterns
        if any(x in name_lower for x in ['rsi', 'relative_strength']):
            lines.extend([
                f"{prefix}_value = ta.rsi(close, {length_param})",
                f"{prefix}_buy_signal = {prefix}_value < 30",
                f"{prefix}_sell_signal = {prefix}_value > 70",
            ])
        elif any(x in name_lower for x in ['macd', 'moving_average_conv']):
            lines.extend([
                f"{prefix}_fast_ema = ta.ema(close, math.max(1, {length_param}))",
                f"{prefix}_slow_ema = ta.ema(close, math.max(2, {length_param} * 2))",
                f"{prefix}_macd = {prefix}_fast_ema - {prefix}_slow_ema",
                f"{prefix}_signal_line = ta.ema({prefix}_macd, 9)",
                f"{prefix}_buy_signal = ta.crossover({prefix}_macd, {prefix}_signal_line)",
                f"{prefix}_sell_signal = ta.crossunder({prefix}_macd, {prefix}_signal_line)",
            ])
        elif any(x in name_lower for x in ['stoch', 'stochastic']):
            lines.extend([
                f"{prefix}_k = ta.stoch(close, high, low, {length_param})",
                f"{prefix}_d = ta.sma({prefix}_k, 3)",
                f"{prefix}_buy_signal = {prefix}_k < 20 and ta.crossover({prefix}_k, {prefix}_d)",
                f"{prefix}_sell_signal = {prefix}_k > 80 and ta.crossunder({prefix}_k, {prefix}_d)",
            ])
        elif any(x in name_lower for x in ['cci', 'commodity_channel']):
            lines.extend([
                f"{prefix}_value = ta.cci(high, low, close, {length_param})",
                f"{prefix}_buy_signal = {prefix}_value < -100",
                f"{prefix}_sell_signal = {prefix}_value > 100",
            ])
        elif any(x in name_lower for x in ['mfi', 'money_flow']):
            lines.extend([
                f"{prefix}_value = ta.mfi(high, low, close, volume, {length_param})",
                f"{prefix}_buy_signal = {prefix}_value < 20",
                f"{prefix}_sell_signal = {prefix}_value > 80",
            ])
        elif any(x in name_lower for x in ['adx', 'directional']):
            lines.extend([
                f"[{prefix}_diplus, {prefix}_diminus, {prefix}_adx] = ta.dmi({length_param}, {length_param})",
                f"{prefix}_buy_signal = {prefix}_adx > 25 and {prefix}_diplus > {prefix}_diminus",
                f"{prefix}_sell_signal = {prefix}_adx > 25 and {prefix}_diminus > {prefix}_diplus",
            ])
        elif any(x in name_lower for x in ['bb', 'bollinger']):
            lines.extend([
                f"[{prefix}_middle, {prefix}_upper, {prefix}_lower] = ta.bb(close, {length_param}, 2.0)",
                f"{prefix}_buy_signal = close < {prefix}_lower",
                f"{prefix}_sell_signal = close > {prefix}_upper",
            ])
        elif any(x in name_lower for x in ['atr', 'average_true']):
            lines.extend([
                f"{prefix}_atr = ta.atr({length_param})",
                f"{prefix}_threshold = ta.sma({prefix}_atr, 20)",
                f"{prefix}_buy_signal = {prefix}_atr < {prefix}_threshold * 0.8",
                f"{prefix}_sell_signal = {prefix}_atr > {prefix}_threshold * 1.2",
            ])
        elif any(x in name_lower for x in ['momentum', 'mom']):
            lines.extend([
                f"{prefix}_mom = ta.mom(close, {length_param})",
                f"{prefix}_buy_signal = {prefix}_mom > 0 and {prefix}_mom > {prefix}_mom[1]",
                f"{prefix}_sell_signal = {prefix}_mom < 0 and {prefix}_mom < {prefix}_mom[1]",
            ])
        elif any(x in name_lower for x in ['willr', 'williams']):
            lines.extend([
                f"{prefix}_wr = ta.wpr({length_param})",
                f"{prefix}_buy_signal = {prefix}_wr < -80",
                f"{prefix}_sell_signal = {prefix}_wr > -20",
            ])
        elif any(x in name_lower for x in ['obv', 'on_balance']):
            lines.extend([
                f"{prefix}_obv = ta.obv",
                f"{prefix}_obv_ma = ta.sma({prefix}_obv, {length_param})",
                f"{prefix}_buy_signal = {prefix}_obv > {prefix}_obv_ma",
                f"{prefix}_sell_signal = {prefix}_obv < {prefix}_obv_ma",
            ])
        elif any(x in name_lower for x in ['ema', 'exponential']):
            lines.extend([
                f"{prefix}_fast_ema = ta.ema(close, math.max(1, {length_param}))",
                f"{prefix}_slow_ema = ta.ema(close, math.max(2, {length_param} * 2))",
                f"{prefix}_buy_signal = ta.crossover({prefix}_fast_ema, {prefix}_slow_ema)",
                f"{prefix}_sell_signal = ta.crossunder({prefix}_fast_ema, {prefix}_slow_ema)",
            ])
        elif any(x in name_lower for x in ['sma', 'simple_moving']):
            lines.extend([
                f"{prefix}_fast_sma = ta.sma(close, math.max(1, {length_param}))",
                f"{prefix}_slow_sma = ta.sma(close, math.max(2, {length_param} * 2))",
                f"{prefix}_buy_signal = ta.crossover({prefix}_fast_sma, {prefix}_slow_sma)",
                f"{prefix}_sell_signal = ta.crossunder({prefix}_fast_sma, {prefix}_slow_sma)",
            ])
        elif any(x in name_lower for x in ['vwap']):
            lines.extend([
                f"{prefix}_vwap = ta.vwap",
                f"{prefix}_buy_signal = close > {prefix}_vwap and close[1] <= {prefix}_vwap[1]",
                f"{prefix}_sell_signal = close < {prefix}_vwap and close[1] >= {prefix}_vwap[1]",
            ])
        elif any(x in name_lower for x in ['supertrend']):
            lines.extend([
                f"[{prefix}_supertrend, {prefix}_direction] = ta.supertrend(3.0, {length_param})",
                f"{prefix}_buy_signal = {prefix}_direction < 0 and {prefix}_direction[1] >= 0",
                f"{prefix}_sell_signal = {prefix}_direction > 0 and {prefix}_direction[1] <= 0",
            ])
        elif any(x in name_lower for x in ['ichimoku']):
            lines.extend([
                f"{prefix}_conversion = (ta.highest(high, 9) + ta.lowest(low, 9)) / 2",
                f"{prefix}_base = (ta.highest(high, 26) + ta.lowest(low, 26)) / 2",
                f"{prefix}_buy_signal = ta.crossover({prefix}_conversion, {prefix}_base)",
                f"{prefix}_sell_signal = ta.crossunder({prefix}_conversion, {prefix}_base)",
            ])
        elif any(x in name_lower for x in ['pivot', 'support', 'resistance']):
            lines.extend([
                f"{prefix}_pivot = (high[1] + low[1] + close[1]) / 3",
                f"{prefix}_r1 = 2 * {prefix}_pivot - low[1]",
                f"{prefix}_s1 = 2 * {prefix}_pivot - high[1]",
                f"{prefix}_buy_signal = close > {prefix}_pivot and close[1] <= {prefix}_pivot[1]",
                f"{prefix}_sell_signal = close < {prefix}_pivot and close[1] >= {prefix}_pivot[1]",
            ])
        elif any(x in name_lower for x in ['accelerator', 'ao', 'awesome']):
            lines.extend([
                f"{prefix}_ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)",
                f"{prefix}_ac = {prefix}_ao - ta.sma({prefix}_ao, 5)",
                f"{prefix}_buy_signal = {prefix}_ac > 0 and {prefix}_ac > {prefix}_ac[1]",
                f"{prefix}_sell_signal = {prefix}_ac < 0 and {prefix}_ac < {prefix}_ac[1]",
            ])
        elif any(x in name_lower for x in ['chaikin', 'cmf']):
            lines.extend([
                f"{prefix}_mf_multiplier = ((close - low) - (high - close)) / (high - low)",
                f"{prefix}_mf_volume = {prefix}_mf_multiplier * volume",
                f"{prefix}_cmf = ta.sma({prefix}_mf_volume, {length_param}) / ta.sma(volume, {length_param})",
                f"{prefix}_buy_signal = {prefix}_cmf > 0.05",
                f"{prefix}_sell_signal = {prefix}_cmf < -0.05",
            ])
        else:
            # Generic signal based on price momentum
            lines.extend([
                f"// Generic signal for {indicator_name}",
                f"{prefix}_roc = ta.roc(close, math.max(1, {length_param}))",
                f"{prefix}_roc_ma = ta.sma({prefix}_roc, 5)",
                f"{prefix}_buy_signal = {prefix}_roc > 0 and ta.crossover({prefix}_roc, {prefix}_roc_ma)",
                f"{prefix}_sell_signal = {prefix}_roc < 0 and ta.crossunder({prefix}_roc, {prefix}_roc_ma)",
            ])
        
        return lines


def create_hybrid_from_results(
    results: List[Dict[str, Any]],
    output_dir: str = "pinescripts",
    n_top: int = 5,
    min_score: float = 0.0,
    voting_method: str = "majority",
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """
    Convenience function to create a hybrid indicator from optimization results.
    
    IMPORTANT: All results MUST be from optimizations on the SAME symbol and 
    timeframe combination. The caller is responsible for filtering results
    to ensure this constraint is met before calling this function.
    
    Args:
        results: List of optimization result dictionaries (must be same symbol/timeframe)
        output_dir: Directory to write the hybrid Pine Script
        n_top: Number of top indicators to include
        min_score: Minimum score threshold
        voting_method: Voting method for ensemble
        symbol: Symbol context (for naming and metadata)
        interval: Interval/timeframe context (for naming and metadata)
        
    Returns:
        Tuple of (path to generated file, hybrid metadata) or None if not enough indicators
    """
    generator = HybridIndicatorGenerator()
    
    for result in results:
        if not result.get("best_metrics"):
            continue
            
        metrics = result.get("best_metrics", {})
        if isinstance(metrics, dict):
            score = metrics.get("mcc", 0) + 0.5 * (metrics.get("roc_auc", 0.5) - 0.5)
        else:
            # BacktestMetrics object
            score = getattr(metrics, 'mcc', 0) + 0.5 * (getattr(metrics, 'roc_auc', 0.5) - 0.5)
            metrics = {
                "mcc": getattr(metrics, 'mcc', 0),
                "roc_auc": getattr(metrics, 'roc_auc', 0.5),
                "profit_factor": getattr(metrics, 'profit_factor', 0),
                "win_rate": getattr(metrics, 'win_rate', 0),
            }
        
        generator.add_optimized_indicator(
            indicator_name=result.get("indicator_name", "Unknown"),
            file_path=result.get("file_name", ""),
            best_params=result.get("best_params", {}),
            original_params=result.get("original_params", {}),
            metrics=metrics,
            score=result.get("objective_best", score),
        )
    
    if generator.get_indicator_count() < 2:
        logger.warning(f"Not enough indicators ({generator.get_indicator_count()}) for hybrid generation")
        return None
    
    try:
        pine_content, filename, metadata = generator.generate_hybrid_indicator(
            n_indicators=n_top,
            min_score=min_score,
            voting_method=voting_method,
            symbol=symbol,
            interval=interval,
        )
    except ValueError as e:
        logger.warning(f"Could not generate hybrid: {e}")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hybrid_file = output_path / filename
    hybrid_file.write_text(pine_content, encoding='utf-8')
    
    logger.info(f"Created hybrid indicator: {hybrid_file}")
    
    return hybrid_file, metadata


if __name__ == "__main__":
    # Test hybrid generation
    logging.basicConfig(level=logging.INFO)
    
    # Create test results
    test_results = [
        {
            "indicator_name": "RSI_Indicator",
            "file_name": "rsi.pine",
            "best_params": {"length": 14, "overbought": 70, "oversold": 30},
            "original_params": {"length": 14, "overbought": 70, "oversold": 30},
            "best_metrics": {"mcc": 0.15, "roc_auc": 0.58, "profit_factor": 1.5},
            "objective_best": 0.19,
        },
        {
            "indicator_name": "MACD_Crossover",
            "file_name": "macd.pine",
            "best_params": {"fast": 12, "slow": 26, "signal": 9},
            "original_params": {"fast": 12, "slow": 26, "signal": 9},
            "best_metrics": {"mcc": 0.12, "roc_auc": 0.55, "profit_factor": 1.3},
            "objective_best": 0.145,
        },
        {
            "indicator_name": "Bollinger_Bands",
            "file_name": "bb.pine",
            "best_params": {"length": 20, "mult": 2.0},
            "original_params": {"length": 20, "mult": 2.0},
            "best_metrics": {"mcc": 0.10, "roc_auc": 0.53, "profit_factor": 1.2},
            "objective_best": 0.115,
        },
    ]
    
    result = create_hybrid_from_results(
        test_results,
        output_dir="test_hybrids",
        n_top=3,
        voting_method="majority"
    )
    
    if result:
        path, metadata = result
        print(f"Generated: {path}")
        print(f"Metadata: {metadata}")
