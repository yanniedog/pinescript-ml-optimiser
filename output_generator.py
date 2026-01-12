"""
Output Generator
Generates optimized Pine Script files and performance reports.
"""

import re
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from pine_parser import ParseResult, Parameter
from optimizer import OptimizationResult, DataUsageInfo
from backtester import BacktestMetrics
from objective import calculate_objective_score
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generates optimized Pine Script and reports."""
    
    def __init__(self, parse_result: ParseResult, optimization_result: OptimizationResult):
        self.parse_result = parse_result
        self.optimization_result = optimization_result
    
    def _format_value(self, value) -> str:
        """Format a value for display with proper precision."""
        if isinstance(value, bool):
            return str(value)
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            if abs(value) < 0.0001:
                return f"{value:.2e}"
            elif abs(value) < 1:
                return f"{value:.4f}"
            else:
                return f"{value:.2f}"
        return str(value)
    
    def _calculate_objective_score(self, metrics: BacktestMetrics) -> float:
        """Calculate objective score using the same formula as optimizer."""
        return calculate_objective_score(metrics)
    
    def _interval_to_hours(self, interval: str) -> float:
        """Convert interval string to hours."""
        interval_map = {
            "1m": 1/60, "3m": 3/60, "5m": 5/60, "15m": 15/60, "30m": 30/60,
            "1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12,
            "1d": 24, "3d": 72,
            "1w": 168,
            "1M": 720  # Approximate
        }
        return interval_map.get(interval.lower(), 1.0)  # Default to 1 hour
    
    def _format_timeframe_duration(self, horizon_bars: int, interval: str) -> str:
        """Convert horizon in bars to time duration."""
        hours = horizon_bars * self._interval_to_hours(interval)
        
        if hours < 1:
            minutes = int(hours * 60)
            return f"{horizon_bars} bars ({minutes} minutes)"
        elif hours < 24:
            return f"{horizon_bars} bars ({hours:.1f} hours)"
        else:
            days = hours / 24
            if days < 7:
                return f"{horizon_bars} bars ({days:.1f} days)"
            else:
                weeks = days / 7
                return f"{horizon_bars} bars ({weeks:.1f} weeks)"
    
    def _format_date_range(self, start: datetime, end: datetime) -> str:
        """Format date range for display."""
        return f"{start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _calculate_bar_percentage(self, used: int, total: int) -> float:
        """Calculate percentage of bars used."""
        return (used / total * 100) if total > 0 else 0.0
    
    def _get_symbol_rankings(self, opt: OptimizationResult, metric_type: str = 'objective') -> List[Tuple[str, float, float, float]]:
        """Calculate rankings for symbols based on optimized metric."""
        rankings = []
        
        # Check if multi-timeframe structure
        is_multi_tf = False
        if opt.per_symbol_metrics:
            first_value = next(iter(opt.per_symbol_metrics.values()))
            if isinstance(first_value, dict) and first_value:
                first_nested = next(iter(first_value.values()))
                if isinstance(first_nested, dict) and 'original' in first_nested:
                    is_multi_tf = True
        
        if is_multi_tf:
            # Multi-timeframe: aggregate per symbol
            for symbol, timeframes_dict in opt.per_symbol_metrics.items():
                # Aggregate metrics across timeframes for this symbol
                all_metrics = []
                for timeframe, metrics_dict in timeframes_dict.items():
                    all_metrics.append(metrics_dict.get('optimized', BacktestMetrics()))
                
                if all_metrics:
                    # Aggregate original metrics too
                    all_orig_metrics = []
                    for timeframe, metrics_dict in timeframes_dict.items():
                        all_orig_metrics.append(metrics_dict.get('original', BacktestMetrics()))

                    def _aggregate_classification(metrics_list):
                        total_tp = sum(m.tp for m in metrics_list)
                        total_tn = sum(m.tn for m in metrics_list)
                        total_fp = sum(m.fp for m in metrics_list)
                        total_fn = sum(m.fn for m in metrics_list)
                        total_samples = sum(m.classification_samples for m in metrics_list)
                        if total_samples > 0:
                            roc_auc = sum(m.roc_auc * m.classification_samples for m in metrics_list) / total_samples
                        else:
                            roc_auc = 0.5
                        denom = (total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn)
                        mcc = (total_tp * total_tn - total_fp * total_fn) / (denom ** 0.5) if denom > 0 else 0.0
                        return mcc, roc_auc, total_samples, total_tp, total_tn, total_fp, total_fn

                    opt_mcc, opt_auc, opt_samples, opt_tp, opt_tn, opt_fp, opt_fn = _aggregate_classification(all_metrics)
                    orig_mcc, orig_auc, orig_samples, orig_tp, orig_tn, orig_fp, orig_fn = _aggregate_classification(all_orig_metrics)

                    # Simple average for now (plus classification aggregates)
                    avg_opt_metrics = BacktestMetrics(
                        profit_factor=sum(m.profit_factor for m in all_metrics) / len(all_metrics),
                        win_rate=sum(m.win_rate for m in all_metrics) / len(all_metrics),
                        directional_accuracy=sum(m.directional_accuracy for m in all_metrics) / len(all_metrics),
                        sharpe_ratio=sum(m.sharpe_ratio for m in all_metrics) / len(all_metrics),
                        total_trades=sum(m.total_trades for m in all_metrics),
                        mcc=opt_mcc,
                        roc_auc=opt_auc,
                        classification_samples=opt_samples,
                        tp=opt_tp,
                        tn=opt_tn,
                        fp=opt_fp,
                        fn=opt_fn
                    )
                    
                    avg_orig_metrics = BacktestMetrics(
                        profit_factor=sum(m.profit_factor for m in all_orig_metrics) / len(all_orig_metrics) if all_orig_metrics else 0,
                        win_rate=sum(m.win_rate for m in all_orig_metrics) / len(all_orig_metrics) if all_orig_metrics else 0,
                        directional_accuracy=sum(m.directional_accuracy for m in all_orig_metrics) / len(all_orig_metrics) if all_orig_metrics else 0,
                        sharpe_ratio=sum(m.sharpe_ratio for m in all_orig_metrics) / len(all_orig_metrics) if all_orig_metrics else 0,
                        total_trades=sum(m.total_trades for m in all_orig_metrics) if all_orig_metrics else 0,
                        mcc=orig_mcc,
                        roc_auc=orig_auc,
                        classification_samples=orig_samples,
                        tp=orig_tp,
                        tn=orig_tn,
                        fp=orig_fp,
                        fn=orig_fn
                    )
                    
                    if metric_type == 'objective':
                        orig_score = self._calculate_objective_score(avg_orig_metrics)
                        opt_score = self._calculate_objective_score(avg_opt_metrics)
                    elif metric_type == 'profit_factor':
                        orig_score = avg_orig_metrics.profit_factor
                        opt_score = avg_opt_metrics.profit_factor
                    elif metric_type == 'win_rate':
                        orig_score = avg_orig_metrics.win_rate * 100
                        opt_score = avg_opt_metrics.win_rate * 100
                    elif metric_type == 'directional_accuracy':
                        orig_score = avg_orig_metrics.directional_accuracy * 100
                        opt_score = avg_opt_metrics.directional_accuracy * 100
                    else:
                        continue
                    
                    change = ((opt_score - orig_score) / orig_score * 100) if orig_score > 0 else 0
                    rankings.append((symbol, orig_score, opt_score, change))
        else:
            # Single-timeframe structure
            for symbol, metrics_dict in opt.per_symbol_metrics.items():
                orig_metrics = metrics_dict.get('original', BacktestMetrics())
                opt_metrics = metrics_dict.get('optimized', BacktestMetrics())
                
                if metric_type == 'objective':
                    orig_score = self._calculate_objective_score(orig_metrics)
                    opt_score = self._calculate_objective_score(opt_metrics)
                elif metric_type == 'profit_factor':
                    orig_score = orig_metrics.profit_factor
                    opt_score = opt_metrics.profit_factor
                elif metric_type == 'win_rate':
                    orig_score = orig_metrics.win_rate * 100
                    opt_score = opt_metrics.win_rate * 100
                elif metric_type == 'directional_accuracy':
                    orig_score = orig_metrics.directional_accuracy * 100
                    opt_score = opt_metrics.directional_accuracy * 100
                else:
                    continue
                
                change = ((opt_score - orig_score) / orig_score * 100) if orig_score > 0 else 0
                rankings.append((symbol, orig_score, opt_score, change))
        
        # Sort by optimized score (descending)
        rankings.sort(key=lambda x: x[2], reverse=True)
        return rankings
    
    def generate_optimized_pine(self, output_path: str = None) -> str:
        """
        Generate optimized Pine Script with ML-derived parameters.
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Path to generated file
        """
        original_content = self.parse_result.raw_content
        optimized_content = original_content
        
        # Replace each parameter's default value
        for param in self.parse_result.parameters:
            if param.name in self.optimization_result.best_params:
                new_value = self.optimization_result.best_params[param.name]
                old_value = param.default
                
                # Skip if no change
                if new_value == old_value:
                    continue
                
                # Build replacement pattern for input.* call
                optimized_content = self._replace_param_default(
                    optimized_content,
                    param,
                    new_value
                )
        
        # Add optimization metadata header
        header = self._generate_header()
        
        # Insert header after version line
        version_pattern = re.compile(r'(//@version=\d+)')
        match = version_pattern.search(optimized_content)
        if match:
            insert_pos = match.end()
            optimized_content = optimized_content[:insert_pos] + '\n' + header + optimized_content[insert_pos:]
        else:
            optimized_content = header + '\n' + optimized_content
        
        # Update indicator name to include "Optimised"
        optimized_content = self._update_indicator_name(optimized_content)
        
        # Determine output path
        if output_path is None:
            original_path = Path(self.parse_result.indicator_name.replace(' ', '_').replace('/', '_'))
            # Find source file name if available
            output_path = f"optimised_{original_path.stem}.pine"
        
        # Write file
        Path(output_path).write_text(optimized_content, encoding='utf-8')
        logger.info(f"Generated optimized Pine Script: {output_path}")
        
        return output_path
    
    def _replace_param_default(self, content: str, param: Parameter, new_value: Any) -> str:
        """Replace a parameter's default value in the Pine Script."""
        # Format value based on type
        if param.param_type == 'bool':
            value_str = 'true' if new_value else 'false'
        elif param.param_type == 'int':
            value_str = str(int(new_value))
        else:
            # Float - use appropriate precision
            if abs(new_value) < 0.01:
                value_str = f"{new_value:.6f}"
            elif abs(new_value) < 1:
                value_str = f"{new_value:.4f}"
            else:
                value_str = f"{new_value:.2f}"
        
        # Pattern to match the input.* call for this parameter
        # Handle various formats: input.int(23, ...), input.int(defval=23, ...)
        patterns = [
            # Standard: varname = input.type(value, ...)
            rf'({re.escape(param.name)}\s*=\s*input\.{param.param_type}\s*\(\s*)({re.escape(str(param.default))})',
            # With defval: input.type(defval=value, ...)
            rf'({re.escape(param.name)}\s*=\s*input\.{param.param_type}\s*\([^)]*defval\s*=\s*)({re.escape(str(param.default))})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                content = content[:match.start(2)] + value_str + content[match.end(2):]
                return content
        
        # Fallback: try simpler replacement
        old_line = param.original_line
        if old_line:
            old_default = str(param.default)
            new_line = old_line.replace(f'({old_default}', f'({value_str}', 1)
            if new_line != old_line:
                content = content.replace(old_line, new_line)
        
        return content
    
    def _generate_header(self) -> str:
        """Generate optimization metadata header comment."""
        opt = self.optimization_result
        metrics = opt.best_metrics
        data_summary = self._get_optimization_data_summary()
        
        header_lines = [
            "// ===========================================================================",
            "// ML-OPTIMISED PARAMETERS",
            f"// Optimized: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"// Trials: {opt.n_trials} | Time: {opt.optimization_time:.1f}s",
            "// ---------------------------------------------------------------------------",
            f"// Strategy: {getattr(opt, 'strategy', 'tpe')} | Sampler: {getattr(opt, 'sampler_name', 'tpe')}",
            f"// Timeframe: {data_summary['interval'] or 'Not specified'}",
            f"// Symbols: {', '.join(data_summary['symbols']) if data_summary['symbols'] else 'Not specified'}",
            f"// Date range: {data_summary['date_range']}",
            "// ---------------------------------------------------------------------------",
            f"// MCC (primary): {metrics.mcc:.3f}",
            f"// ROC AUC: {metrics.roc_auc:.3f}",
            f"// Profit Factor: {metrics.profit_factor:.2f} ({opt.improvement_pf:+.1f}% vs original)",
            f"// Win Rate: {metrics.win_rate:.1%}",
            f"// Directional Accuracy: {metrics.directional_accuracy:.1%}",
            f"// Extreme Move Capture: {metrics.tail_capture_rate:.1%}",
            f"// Consistency Score: {metrics.consistency_score:.2f}",
            f"// Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            f"// Max Drawdown: {metrics.max_drawdown:.1f}%",
            f"// Improvement over Random: {metrics.improvement_over_random:+.1f}%",
        ]

        if opt.holdout_metrics is not None and opt.holdout_original_metrics is not None:
            header_lines.extend([
                "// ---------------------------------------------------------------------------",
                "// LOCKBOX (OUT-OF-SAMPLE) RESULTS:",
                f"//   MCC (primary): {opt.holdout_original_metrics.mcc:.3f} -> {opt.holdout_metrics.mcc:.3f}",
                f"//   ROC AUC: {opt.holdout_original_metrics.roc_auc:.3f} -> {opt.holdout_metrics.roc_auc:.3f}",
                f"//   Profit Factor: {opt.holdout_original_metrics.profit_factor:.2f} -> {opt.holdout_metrics.profit_factor:.2f}",
                f"//   Win Rate: {opt.holdout_original_metrics.win_rate:.1%} -> {opt.holdout_metrics.win_rate:.1%}",
                f"//   Directional Accuracy: {opt.holdout_original_metrics.directional_accuracy:.1%} -> {opt.holdout_metrics.directional_accuracy:.1%}",
            ])

        header_lines.extend([
            "// ---------------------------------------------------------------------------",
            f"// OPTIMAL FORECAST HORIZON: {metrics.forecast_horizon} hours",
            f"// ",
            f"// WHEN TO USE: This indicator is optimized for {self._get_optimal_conditions()}",
            f"// EXPECTED PROFITABILITY: Best results with ~{metrics.forecast_horizon}h forecast horizon",
            "// ---------------------------------------------------------------------------",
            "// PARAMETER CHANGES:",
        ])
        
        # Add parameter changes
        for name, new_val in opt.best_params.items():
            old_val = opt.original_params.get(name)
            if old_val != new_val:
                header_lines.append(f"//   {name}: {self._format_value(old_val)} -> {self._format_value(new_val)}")
        
        header_lines.append("// ===========================================================================")
        
        return '\n'.join(header_lines)
    
    def _get_optimal_conditions(self) -> str:
        """Determine optimal market conditions based on metrics."""
        metrics = self.optimization_result.best_metrics
        
        conditions = []
        
        # Based on win rate
        if metrics.win_rate > 0.6:
            conditions.append("high probability setups")
        
        # Based on profit factor
        if metrics.profit_factor > 2.0:
            conditions.append("trending markets")
        elif metrics.profit_factor < 1.5:
            conditions.append("ranging/consolidating markets")
        
        # Based on forecast horizon - fluid continuum
        if metrics.forecast_horizon <= 4:
            conditions.append("scalping/ultra-short-term")
        elif metrics.forecast_horizon <= 12:
            conditions.append("short-term intraday moves")
        elif metrics.forecast_horizon <= 24:
            conditions.append("intraday holds")
        elif metrics.forecast_horizon <= 48:
            conditions.append("overnight to multi-day holds")
        elif metrics.forecast_horizon <= 96:
            conditions.append("swing trades")
        else:
            conditions.append("position trades/longer holds")
        
        return ", ".join(conditions) if conditions else "general market conditions"

    def _get_optimization_data_summary(self) -> Dict[str, Any]:
        """Summarize symbols/timeframes/date ranges used in optimization."""
        opt = self.optimization_result
        symbols = sorted(opt.datasets_used) if opt.datasets_used else []
        interval = opt.interval or ""
        min_date = None
        max_date = None
        
        for symbol, timeframes in opt.data_usage_info.items():
            for timeframe, info in timeframes.items():
                if not info or not info.date_range:
                    continue
                start, end = info.date_range
                if min_date is None or start < min_date:
                    min_date = start
                if max_date is None or end > max_date:
                    max_date = end
        
        date_range = None
        if min_date and max_date:
            date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        
        return {
            "symbols": symbols,
            "interval": interval,
            "date_range": date_range or "Not specified",
        }
    
    def _update_indicator_name(self, content: str) -> str:
        """Update indicator name to include 'Optimised' prefix."""
        # Match indicator() call and update the name
        pattern = r'(indicator\s*\(\s*["\'])([^"\']+)(["\'])'
        
        def replace_name(match):
            prefix = match.group(1)
            name = match.group(2)
            suffix = match.group(3)
            
            # Add "Optimised" prefix if not already there
            if not name.startswith('Optimised'):
                name = f"Optimised {name}"
            
            return f"{prefix}{name}{suffix}"
        
        return re.sub(pattern, replace_name, content)
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate detailed performance report.
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Report content as string
        """
        opt = self.optimization_result
        orig_metrics = opt.original_metrics
        best_metrics = opt.best_metrics
        
        # Format time nicely
        total_seconds = opt.optimization_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds:.1f}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
        
        report_lines = [
            "=" * 70,
            f"ML OPTIMIZATION REPORT: {self.parse_result.indicator_name}",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "OPTIMIZATION SUMMARY",
            "-" * 70,
            f"  Trials completed:     {opt.n_trials}",
            f"  Total time taken:     {time_str} ({opt.optimization_time:.1f} seconds)",
            f"  Parameters optimized: {len([p for p in opt.best_params if opt.best_params[p] != opt.original_params.get(p)])}",
        ]
        
        # Add historical datasets used
        if opt.datasets_used:
            interval_str = f" @ {opt.interval}" if opt.interval else ""
            report_lines.append(f"  Historical datasets:   {len(opt.datasets_used)} symbols{interval_str}")
            report_lines.append("")
            report_lines.append("  Datasets referenced:")
            for dataset in sorted(opt.datasets_used):
                dataset_str = f"{dataset}{interval_str}" if opt.interval else dataset
                report_lines.append(f"    - {dataset_str}")
        else:
            report_lines.append(f"  Historical datasets:   Not specified")
        
        # Optimization data summary
        data_summary = self._get_optimization_data_summary()
        report_lines.append("")
        report_lines.append("  Optimization data:")
        report_lines.append(f"    Timeframe:          {data_summary['interval'] or 'Not specified'}")
        report_lines.append(f"    Symbols:            {', '.join(data_summary['symbols']) if data_summary['symbols'] else 'Not specified'}")
        report_lines.append(f"    Date range:         {data_summary['date_range']}")
        
        # Add optimization config details
        report_lines.append("")
        report_lines.append("  Optimization config:")
        report_lines.append(f"    Strategy:            {getattr(opt, 'strategy', 'tpe')}")
        report_lines.append(f"    Sampler:             {getattr(opt, 'sampler_name', 'tpe')}")
        report_lines.append(f"    Timeout (sec):       {getattr(opt, 'timeout_seconds', 0)}")
        report_lines.append(f"    Max trials:          {getattr(opt, 'max_trials', None)}")
        report_lines.append(f"    Early stop patience: {getattr(opt, 'early_stop_patience', None)}")
        report_lines.append(f"    Min runtime (sec):   {getattr(opt, 'min_runtime_seconds', 0)}")
        report_lines.append(f"    Stall timeout (sec): {getattr(opt, 'stall_seconds', None)}")
        report_lines.append(f"    Rate floor (%/s):    {getattr(opt, 'improvement_rate_floor', 0.0)}")
        report_lines.append(f"    Rate window:         {getattr(opt, 'improvement_rate_window', 0)}")
        report_lines.append(f"    Backtester overrides:{getattr(opt, 'backtester_overrides', {})}")
        if getattr(opt, "holdout_ratio", 0.0) > 0:
            report_lines.append(f"    Holdout ratio:       {opt.holdout_ratio:.0%}")
            report_lines.append(f"    Holdout gap bars:    {getattr(opt, 'holdout_gap_bars', 0)}")
        
        report_lines.extend([
            "",
            "-" * 70,
            "PERFORMANCE COMPARISON",
            "-" * 70,
            "",
            f"  {'Metric':<25} {'Original':>12} {'Optimized':>12} {'Change':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}",
        ])
        
        # Add metric comparisons
        metrics_to_compare = [
            ('MCC (primary)', orig_metrics.mcc, best_metrics.mcc, '.3f'),
            ('ROC AUC', orig_metrics.roc_auc, best_metrics.roc_auc, '.3f'),
            ('Classification Samples', orig_metrics.classification_samples, best_metrics.classification_samples, 'd'),
            ('Profit Factor', orig_metrics.profit_factor, best_metrics.profit_factor, '.2f'),
            ('Win Rate', orig_metrics.win_rate * 100, best_metrics.win_rate * 100, '.1f%'),
            ('Directional Accuracy', orig_metrics.directional_accuracy * 100, best_metrics.directional_accuracy * 100, '.1f%'),
            ('Extreme Move Capture', orig_metrics.tail_capture_rate * 100, best_metrics.tail_capture_rate * 100, '.1f%'),
            ('Consistency Score', orig_metrics.consistency_score, best_metrics.consistency_score, '.2f'),
            ('Sharpe Ratio', orig_metrics.sharpe_ratio, best_metrics.sharpe_ratio, '.2f'),
            ('Total Trades', orig_metrics.total_trades, best_metrics.total_trades, 'd'),
            ('Avg Return/Trade', orig_metrics.avg_return, best_metrics.avg_return, '.2f%'),
            ('Max Drawdown', orig_metrics.max_drawdown, best_metrics.max_drawdown, '.1f%'),
        ]
        
        for name, orig, best, fmt in metrics_to_compare:
            if 'd' in fmt:
                orig_str = f"{int(orig)}"
                best_str = f"{int(best)}"
                if orig > 0:
                    change = (best - orig) / orig * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "N/A"
            elif '%' in fmt:
                orig_str = f"{orig:.1f}%"
                best_str = f"{best:.1f}%"
                change = best - orig
                change_str = f"{change:+.1f}pp"
            else:
                orig_str = f"{orig:{fmt}}"
                best_str = f"{best:{fmt}}"
                if orig > 0:
                    change = (best - orig) / orig * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "N/A"
            
            report_lines.append(f"  {name:<25} {orig_str:>12} {best_str:>12} {change_str:>12}")

        if opt.holdout_metrics is not None and opt.holdout_original_metrics is not None:
            report_lines.extend([
                "",
                "-" * 70,
                "LOCKBOX (OUT-OF-SAMPLE) PERFORMANCE",
                "-" * 70,
                "",
                f"  {'Metric':<25} {'Original':>12} {'Optimized':>12} {'Change':>12}",
                f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}",
            ])

            holdout_metrics_to_compare = [
                ('MCC (primary)', opt.holdout_original_metrics.mcc, opt.holdout_metrics.mcc, '.3f'),
                ('ROC AUC', opt.holdout_original_metrics.roc_auc, opt.holdout_metrics.roc_auc, '.3f'),
                ('Classification Samples', opt.holdout_original_metrics.classification_samples, opt.holdout_metrics.classification_samples, 'd'),
                ('Profit Factor', opt.holdout_original_metrics.profit_factor, opt.holdout_metrics.profit_factor, '.2f'),
                ('Win Rate', opt.holdout_original_metrics.win_rate * 100, opt.holdout_metrics.win_rate * 100, '.1f%'),
                ('Directional Accuracy', opt.holdout_original_metrics.directional_accuracy * 100, opt.holdout_metrics.directional_accuracy * 100, '.1f%'),
                ('Extreme Move Capture', opt.holdout_original_metrics.tail_capture_rate * 100, opt.holdout_metrics.tail_capture_rate * 100, '.1f%'),
                ('Consistency Score', opt.holdout_original_metrics.consistency_score, opt.holdout_metrics.consistency_score, '.2f'),
                ('Sharpe Ratio', opt.holdout_original_metrics.sharpe_ratio, opt.holdout_metrics.sharpe_ratio, '.2f'),
                ('Total Trades', opt.holdout_original_metrics.total_trades, opt.holdout_metrics.total_trades, 'd'),
                ('Avg Return/Trade', opt.holdout_original_metrics.avg_return, opt.holdout_metrics.avg_return, '.2f%'),
                ('Max Drawdown', opt.holdout_original_metrics.max_drawdown, opt.holdout_metrics.max_drawdown, '.1f%'),
            ]

            for name, orig, best, fmt in holdout_metrics_to_compare:
                if 'd' in fmt:
                    orig_str = f"{int(orig)}"
                    best_str = f"{int(best)}"
                    if orig > 0:
                        change = (best - orig) / orig * 100
                        change_str = f"{change:+.1f}%"
                    else:
                        change_str = "N/A"
                elif '%' in fmt:
                    orig_str = f"{orig:.1f}%"
                    best_str = f"{best:.1f}%"
                    change = best - orig
                    change_str = f"{change:+.1f}pp"
                else:
                    orig_str = f"{orig:{fmt}}"
                    best_str = f"{best:{fmt}}"
                    if orig > 0:
                        change = (best - orig) / orig * 100
                        change_str = f"{change:+.1f}%"
                    else:
                        change_str = "N/A"

                report_lines.append(f"  {name:<25} {orig_str:>12} {best_str:>12} {change_str:>12}")
        
        # Add Overall Performance by Symbol table (ranked)
        if opt.per_symbol_metrics:
            rankings = self._get_symbol_rankings(opt, 'objective')
            if rankings:
                report_lines.extend([
                    "",
                    "  Overall Performance by Symbol (Ranked):",
                    f"  {'Rank':<6} {'Symbol':<12} {'Original':>12} {'Optimized':>12} {'Change':>12}",
                    f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}",
                ])
                for rank, (symbol, orig_score, opt_score, change) in enumerate(rankings, 1):
                    report_lines.append(f"  {rank:<6} {symbol:<12} {orig_score:>12.4f} {opt_score:>12.4f} {change:>+11.1f}%")
        
        # Add per-symbol performance comparison table
        if opt.per_symbol_metrics:
            report_lines.extend([
                "",
                "-" * 70,
                "PER-SYMBOL PERFORMANCE",
                "-" * 70,
                "",
                "  Profit Factor by Symbol (Ranked):",
                f"  {'Rank':<6} {'Symbol':<12} {'Original':>10} {'Optimized':>10} {'Change':>10}",
                f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}",
            ])
            
            rankings = self._get_symbol_rankings(opt, 'profit_factor')
            for rank, (symbol, orig_pf, opt_pf, change) in enumerate(rankings, 1):
                change_str = f"{change:+.1f}%" if orig_pf > 0 else "N/A"
                report_lines.append(f"  {rank:<6} {symbol:<12} {orig_pf:>10.2f} {opt_pf:>10.2f} {change_str:>10}")
            
            report_lines.extend([
                "",
                "  Win Rate by Symbol (Ranked):",
                f"  {'Rank':<6} {'Symbol':<12} {'Original':>10} {'Optimized':>10} {'Change':>10}",
                f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}",
            ])
            
            rankings = self._get_symbol_rankings(opt, 'win_rate')
            for rank, (symbol, orig_wr, opt_wr, change) in enumerate(rankings, 1):
                change_pp = opt_wr - orig_wr
                report_lines.append(f"  {rank:<6} {symbol:<12} {orig_wr:>9.1f}% {opt_wr:>9.1f}% {change_pp:>+9.1f}pp")
            
            report_lines.extend([
                "",
                "  Directional Accuracy by Symbol (Ranked):",
                f"  {'Rank':<6} {'Symbol':<12} {'Original':>10} {'Optimized':>10} {'Change':>10}",
                f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}",
            ])
            
            rankings = self._get_symbol_rankings(opt, 'directional_accuracy')
            for rank, (symbol, orig_da, opt_da, change) in enumerate(rankings, 1):
                change_pp = opt_da - orig_da
                report_lines.append(f"  {rank:<6} {symbol:<12} {orig_da:>9.1f}% {opt_da:>9.1f}% {change_pp:>+9.1f}pp")
            
            report_lines.extend([
                "",
                "  Trades by Symbol:",
                f"  {'Symbol':<12} {'Original':>10} {'Optimized':>10} {'Change':>10}",
                f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}",
            ])
            
            # For trades, we'll sort by optimized trades (descending)
            trades_data = []
            
            # Check if multi-timeframe structure
            is_multi_tf = False
            if opt.per_symbol_metrics:
                first_value = next(iter(opt.per_symbol_metrics.values()))
                if isinstance(first_value, dict) and first_value:
                    first_nested = next(iter(first_value.values()))
                    if isinstance(first_nested, dict) and 'original' in first_nested:
                        is_multi_tf = True
            
            if is_multi_tf:
                # Multi-timeframe: aggregate trades across timeframes
                for symbol in opt.per_symbol_metrics:
                    total_orig = 0
                    total_opt = 0
                    for timeframe, metrics_dict in opt.per_symbol_metrics[symbol].items():
                        total_orig += metrics_dict['original'].total_trades
                        total_opt += metrics_dict['optimized'].total_trades
                    trades_data.append((symbol, total_orig, total_opt))
            else:
                # Single-timeframe structure
                for symbol in opt.per_symbol_metrics:
                    sym_data = opt.per_symbol_metrics[symbol]
                    orig_trades = sym_data['original'].total_trades
                    opt_trades = sym_data['optimized'].total_trades
                    trades_data.append((symbol, orig_trades, opt_trades))
            
            trades_data.sort(key=lambda x: x[2], reverse=True)
            for symbol, orig_trades, opt_trades in trades_data:
                if orig_trades > 0:
                    change = (opt_trades - orig_trades) / orig_trades * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "N/A"
                report_lines.append(f"  {symbol:<12} {orig_trades:>10} {opt_trades:>10} {change_str:>10}")
        
        # Add Optimal Forecast Horizon by Symbol section
        if opt.per_symbol_metrics:
            report_lines.extend([
                "",
                "-" * 70,
                "OPTIMAL FORECAST HORIZON BY SYMBOL",
                "-" * 70,
                "",
            ])
            
            # Check if multi-timeframe structure
            is_multi_tf = False
            if opt.per_symbol_metrics:
                first_value = next(iter(opt.per_symbol_metrics.values()))
                if isinstance(first_value, dict) and first_value:
                    first_nested = next(iter(first_value.values()))
                    if isinstance(first_nested, dict) and 'original' in first_nested:
                        is_multi_tf = True
            
            if is_multi_tf:
                # Multi-timeframe structure
                report_lines.extend([
                    f"  {'Symbol':<12} {'Timeframe':<12} {'Horizon (bars)':>16} {'Horizon (time)':>20}",
                    f"  {'-'*12} {'-'*12} {'-'*16} {'-'*20}",
                ])
                for symbol in sorted(opt.per_symbol_metrics.keys()):
                    for timeframe in sorted(opt.per_symbol_metrics[symbol].keys()):
                        metrics = opt.per_symbol_metrics[symbol][timeframe]['optimized']
                        horizon_bars = metrics.forecast_horizon
                        horizon_time = self._format_timeframe_duration(horizon_bars, timeframe)
                        report_lines.append(f"  {symbol:<12} {timeframe:<12} {horizon_bars:>16} {horizon_time:>20}")
            else:
                # Single-timeframe structure
                report_lines.extend([
                    f"  {'Symbol':<12} {'Horizon (bars)':>16} {'Horizon (time)':>20}",
                    f"  {'-'*12} {'-'*16} {'-'*20}",
                ])
                interval = opt.interval if opt.interval else "1h"
                for symbol in sorted(opt.per_symbol_metrics.keys()):
                    metrics = opt.per_symbol_metrics[symbol]['optimized']
                    horizon_bars = metrics.forecast_horizon
                    horizon_time = self._format_timeframe_duration(horizon_bars, interval)
                    report_lines.append(f"  {symbol:<12} {horizon_bars:>16} {horizon_time:>20}")
        
        # Add Multi-Timeframe Performance Comparison section (if applicable)
        if opt.per_symbol_metrics and opt.timeframes_used:
            has_multi_tf = any(len(tfs) > 1 for tfs in opt.timeframes_used.values())
            if has_multi_tf:
                report_lines.extend([
                    "",
                    "-" * 70,
                    "MULTI-TIMEFRAME PERFORMANCE COMPARISON",
                    "-" * 70,
                    "",
                    f"  {'Symbol':<12} {'Timeframe':<12} {'Original':>12} {'Optimized':>12} {'Change':>12} {'Optimal Horizon':>20}",
                    f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*20}",
                ])
                for symbol in sorted(opt.per_symbol_metrics.keys()):
                    if symbol in opt.timeframes_used and len(opt.timeframes_used[symbol]) > 1:
                        for timeframe in sorted(opt.timeframes_used[symbol]):
                            if timeframe in opt.per_symbol_metrics[symbol]:
                                metrics_dict = opt.per_symbol_metrics[symbol][timeframe]
                                orig_metrics = metrics_dict['original']
                                opt_metrics = metrics_dict['optimized']
                                orig_obj = self._calculate_objective_score(orig_metrics)
                                opt_obj = self._calculate_objective_score(opt_metrics)
                                change = ((opt_obj - orig_obj) / orig_obj * 100) if orig_obj > 0 else 0
                                horizon_time = self._format_timeframe_duration(opt_metrics.forecast_horizon, timeframe)
                                report_lines.append(f"  {symbol:<12} {timeframe:<12} {orig_obj:>12.4f} {opt_obj:>12.4f} {change:>+11.1f}% {horizon_time:>20}")
        
        # Add Historical Data Usage Assessment section
        if opt.data_usage_info:
            report_lines.extend([
                "",
                "-" * 70,
                "HISTORICAL DATA USAGE ASSESSMENT",
                "-" * 70,
                "",
                "  Summary by Dataset:",
                f"  {'Symbol':<12} {'Timeframe':<12} {'Total Bars':>12} {'Date Range':<25} {'Train %':>10} {'Test %':>10} {'Embargo %':>12} {'Unused %':>10}",
                f"  {'-'*12} {'-'*12} {'-'*12} {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*10}",
            ])
            
            for symbol in sorted(opt.data_usage_info.keys()):
                for timeframe in sorted(opt.data_usage_info[symbol].keys()):
                    usage = opt.data_usage_info[symbol][timeframe]
                    date_range_str = f"{usage.date_range[0].strftime('%Y-%m-%d')} to {usage.date_range[1].strftime('%Y-%m-%d')}"
                    train_pct = self._calculate_bar_percentage(usage.total_train_bars, usage.total_bars)
                    test_pct = self._calculate_bar_percentage(usage.total_test_bars, usage.total_bars)
                    embargo_pct = self._calculate_bar_percentage(usage.total_embargo_bars, usage.total_bars)
                    unused_pct = self._calculate_bar_percentage(usage.unused_bars, usage.total_bars)
                    tf_display = timeframe if timeframe else opt.interval
                    report_lines.append(f"  {symbol:<12} {tf_display:<12} {usage.total_bars:>12,} {date_range_str:<25} {train_pct:>9.1f}% {test_pct:>9.1f}% {embargo_pct:>11.1f}% {unused_pct:>9.1f}%")
            
            # Detailed breakdown per dataset
            for symbol in sorted(opt.data_usage_info.keys()):
                for timeframe in sorted(opt.data_usage_info[symbol].keys()):
                    usage = opt.data_usage_info[symbol][timeframe]
                    tf_display = timeframe if timeframe else opt.interval
                    report_lines.extend([
                        "",
                        f"  Detailed Breakdown: {symbol} @ {tf_display}",
                        f"    Total bars: {usage.total_bars:,}",
                        f"    Date range: {self._format_date_range(usage.date_range[0], usage.date_range[1])}",
                        f"    Walk-forward folds: {usage.n_folds}",
                        f"    Train ratio: {usage.train_ratio:.1%}",
                        f"    Embargo period: {usage.embargo_bars} bars",
                        "",
                        "    Fold Details:",
                        f"    {'Fold':<6} {'Train Range (bars)':<20} {'Train Range (dates)':<35} {'Test Range (bars)':<20} {'Test Range (dates)':<35} {'Embargo':<10} {'Train %':<10} {'Test %':<10}",
                        f"    {'-'*6} {'-'*20} {'-'*35} {'-'*20} {'-'*35} {'-'*10} {'-'*10} {'-'*10}",
                    ])
                    
                    for fold in usage.folds:
                        train_range = f"{fold['train_start']}-{fold['train_end']-1}"
                        test_range = f"{fold['test_start']}-{fold['test_end']-1}"
                        train_date_range = f"{fold['train_start_date'].strftime('%Y-%m-%d')} to {fold['train_end_date'].strftime('%Y-%m-%d')}" if fold['train_start_date'] and fold['train_end_date'] else "N/A"
                        test_date_range = f"{fold['test_start_date'].strftime('%Y-%m-%d')} to {fold['test_end_date'].strftime('%Y-%m-%d')}" if fold['test_start_date'] and fold['test_end_date'] else "N/A"
                        train_pct = self._calculate_bar_percentage(fold['train_bars'], usage.total_bars)
                        test_pct = self._calculate_bar_percentage(fold['test_bars'], usage.total_bars)
                        report_lines.append(f"    {fold['fold_num']:<6} {train_range:<20} {train_date_range:<35} {test_range:<20} {test_date_range:<35} {fold['embargo_bars']:<10} {train_pct:<9.1f}% {test_pct:<9.1f}%")
                    
                    # Baseline parameter configuration
                    report_lines.extend([
                        "",
                        "    Baseline (Default) Parameter Configuration:",
                        "    (Parameters used for original/unchanged indicator evaluation)",
                    ])
                    
                    baseline_params = []
                    for param_name in sorted(opt.original_params.keys()):
                        param_value = opt.original_params[param_name]
                        baseline_params.append(f"      {param_name}: {self._format_value(param_value)}")
                    
                    if baseline_params:
                        report_lines.extend(baseline_params)
                    else:
                        report_lines.append("      (No parameters found)")
                    
                    # Bias analysis
                    report_lines.extend([
                        "",
                        "    Bias Analysis:",
                    ])
                    
                    if not usage.potential_bias_issues:
                        report_lines.append("    ✓ No overlapping test sets detected")
                        report_lines.append("    ✓ Embargo period is adequate")
                        report_lines.append("    ✓ No data leakage detected between folds")
                    else:
                        for issue in usage.potential_bias_issues:
                            if issue.startswith("Warning:"):
                                report_lines.append(f"    ⚠ {issue}")
                            elif issue.startswith("Overlapping"):
                                report_lines.append(f"    ✗ {issue}")
                            else:
                                report_lines.append(f"    ⚠ {issue}")
        
        # Add improvement history section
        report_lines.extend([
            "",
            "-" * 70,
            "OPTIMIZATION PROGRESS (Improvements Over Baseline)",
            "-" * 70,
            "",
            f"  Baseline (original config) objective: {opt.baseline_objective:.4f}",
            "",
        ])
        
        # Filter to only show actual improvements over baseline
        actual_improvements = []
        if opt.improvement_history:
            for entry in opt.improvement_history:
                if entry['objective'] >= opt.baseline_objective:
                    actual_improvements.append(entry)
            
            if actual_improvements:
                for i, entry in enumerate(actual_improvements, 1):
                    elapsed = entry['elapsed_seconds']
                    obj = entry['objective']
                    pct = entry['pct_vs_original']
                    avg_rate = entry['avg_rate_pct_per_sec']
                    marginal_rate = entry['marginal_rate_pct_per_sec']
                    params = entry['params']
                    
                    sign = "+" if pct >= 0 else ""
                    report_lines.append(f"  [{i}] @ {elapsed:.1f}s: objective={obj:.4f} ({sign}{pct:.2f}% vs original)")
                    report_lines.append(f"      Avg rate: {avg_rate:+.3f}%/sec | Marginal rate: {marginal_rate:.3f}%/sec")
                    
                    # Show changed params (vs original)
                    changed_params = []
                    for pname, pval in params.items():
                        orig_val = opt.original_params.get(pname)
                        if orig_val != pval:
                            changed_params.append(f"{pname}={self._format_value(pval)}")
                    
                    if changed_params:
                        # Split into multiple lines if too long
                        param_str = ", ".join(changed_params)
                        if len(param_str) > 55:
                            report_lines.append(f"      Config changes:")
                            for cp in changed_params:
                                report_lines.append(f"        - {cp}")
                        else:
                            report_lines.append(f"      Config: {param_str}")
                    report_lines.append("")
                
            # Summary of improvement trajectory
            if len(actual_improvements) >= 2:
                first_entry = actual_improvements[0]
                last_entry = actual_improvements[-1]
                
                first_rate = first_entry['avg_rate_pct_per_sec']
                last_rate = last_entry['avg_rate_pct_per_sec']
                
                report_lines.append(f"  Improvement Trajectory:")
                report_lines.append(f"    First improvement rate: {first_rate:+.3f}%/sec")
                report_lines.append(f"    Final improvement rate: {last_rate:+.3f}%/sec")
                
                if first_rate > 0 and last_rate > 0:
                    slowdown = (1 - last_rate / first_rate) * 100
                    if slowdown > 0:
                        report_lines.append(f"    Rate slowdown: {slowdown:.1f}% (diminishing returns observed)")
                    else:
                        report_lines.append(f"    Rate acceleration: {-slowdown:.1f}% (improving efficiency)")
                # Report average and moving-average improvement rates
                avg_rates = [e['avg_rate_pct_per_sec'] for e in actual_improvements]
                if avg_rates:
                    avg_rate = sum(avg_rates) / len(avg_rates)
                    window = 5 if len(avg_rates) >= 5 else len(avg_rates)
                    moving_avg = sum(avg_rates[-window:]) / window
                    report_lines.append(f"    Avg improvement rate: {avg_rate:+.3f}%/sec (vs baseline)")
                    report_lines.append(f"    Moving avg ({window}): {moving_avg:+.3f}%/sec")
                report_lines.append("")
            else:
                report_lines.append("  No improvements over baseline were found during optimization.")
                report_lines.append("  All trials performed worse than the original configuration.")
                report_lines.append("")
        else:
            report_lines.append("  No improvements found during optimization.")
            report_lines.append("")
        
        report_lines.extend([
            "-" * 70,
            "UNDERSTANDING THE METRICS",
            "-" * 70,
            "",
            "  MCC (PRIMARY OBJECTIVE)",
            "    What it measures: Correlation between predicted and actual direction",
            "    Good value: > 0.10 (0.0 = random, 1.0 = perfect, -1.0 = inverse)",
            "    Why it matters: Robust to class imbalance and penalizes all error types.",
            "",
            "  ROC AUC (CONSTRAINT)",
            "    What it measures: Threshold-independent ranking quality",
            "    Good value: >= 0.52 (0.50 = random)",
            "    Why it matters: Ensures genuine predictive signal beyond threshold tuning.",
            "",
            "  REPORTING METRICS (NOT OPTIMIZED)",
            "    Profit Factor, Win Rate, Sharpe, Directional Accuracy, Tail Capture,",
            "    Consistency, and Drawdown are provided for context only.",
            "",
            "-" * 70,
            "OPTIMAL FORECAST HORIZON",
            "-" * 70,
            f"  Peak performance at: {best_metrics.forecast_horizon} hours ahead",
            f"  Optimal forecast horizon: ~{best_metrics.forecast_horizon} hours",
            "",
            "-" * 70,
            "EXPECTED SIGNAL QUALITY",
            "-" * 70,
            f"  MCC (OOS): {best_metrics.mcc:.3f}",
            f"  ROC AUC (OOS): {best_metrics.roc_auc:.3f}",
            f"  Classification samples: {best_metrics.classification_samples}",
            "",
            "  WHEN TO TRADE:",
            f"    * Best during: {self._get_optimal_conditions()}",
            f"    * Optimal timeframe: {best_metrics.forecast_horizon}h forecast horizon",
            f"    * Expected win rate: {best_metrics.win_rate:.1%}",
            "",
            "-" * 70,
            "OPTIMIZED PARAMETERS",
            "-" * 70,
        ])
        
        # List parameter changes
        for param in self.parse_result.parameters:
            name = param.name
            orig_val = opt.original_params.get(name)
            best_val = opt.best_params.get(name)
            
            if orig_val != best_val:
                report_lines.append(f"  {name}:")
                report_lines.append(f"    Original: {orig_val}")
                report_lines.append(f"    Optimized: {best_val}")
                report_lines.append("")
        
        # Add interpretation guide
        report_lines.extend([
            "-" * 70,
            "HOW TO INTERPRET THESE RESULTS",
            "-" * 70,
            "",
        ])
        
        # Add contextual interpretation based on actual metrics
        pf = best_metrics.profit_factor
        wr = best_metrics.win_rate
        sr = best_metrics.sharpe_ratio
        da = best_metrics.directional_accuracy
        
        # Profit Factor assessment
        if pf >= 2.0:
            pf_assessment = "EXCELLENT - Strong edge, likely profitable in live trading"
        elif pf >= 1.5:
            pf_assessment = "GOOD - Solid edge, should be profitable with discipline"
        elif pf >= 1.2:
            pf_assessment = "MODERATE - Small edge, requires strict risk management"
        elif pf >= 1.0:
            pf_assessment = "MARGINAL - Barely profitable, high risk of losses with fees/slippage"
        else:
            pf_assessment = "POOR - Currently unprofitable, needs more optimization or different approach"
        
        # Win Rate assessment
        if wr >= 0.65:
            wr_assessment = "HIGH - Psychologically easy to trade, watch for small wins/big losses"
        elif wr >= 0.55:
            wr_assessment = "GOOD - Balanced, sustainable for most traders"
        elif wr >= 0.45:
            wr_assessment = "MODERATE - Requires discipline, ensure winners > losers in size"
        else:
            wr_assessment = "LOW - Trend-following style, needs big winners to compensate"
        
        # Sharpe assessment
        if sr >= 2.0:
            sr_assessment = "EXCELLENT - Very consistent, low-stress trading"
        elif sr >= 1.0:
            sr_assessment = "GOOD - Reasonable risk-adjusted returns"
        elif sr >= 0.5:
            sr_assessment = "MODERATE - Some volatility, expect drawdowns"
        else:
            sr_assessment = "LOW - High volatility, prepare for significant swings"
        
        report_lines.extend([
            f"  PROFIT FACTOR ({pf:.2f}): {pf_assessment}",
            "",
            f"  WIN RATE ({wr:.1%}): {wr_assessment}",
            "",
            f"  SHARPE RATIO ({sr:.2f}): {sr_assessment}",
            "",
            "-" * 70,
            "PRACTICAL RECOMMENDATIONS",
            "-" * 70,
            "",
            f"  1. POSITION SIZING: With Sharpe of {sr:.2f}, consider {'conservative' if sr < 1.0 else 'moderate' if sr < 2.0 else 'normal'} position sizes",
            "",
            f"  2. HOLD TIME: Optimal results with ~{best_metrics.forecast_horizon}h hold periods",
            "",
            f"  3. MARKET CONDITIONS: Best during {self._get_optimal_conditions()}",
            "",
            "  4. RISK MANAGEMENT:",
            f"     - Max drawdown seen: {best_metrics.max_drawdown:.1f}%",
            f"     - Set stop losses accounting for this volatility",
            f"     - Never risk more than 1-2% per trade",
            "",
            "  5. VALIDATION: Before live trading:",
            "     - Paper trade for at least 20-30 signals",
            "     - Verify results match backtest expectations",
            "     - Account for fees, slippage, and execution delays",
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        report_content = '\n'.join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report_content, encoding='utf-8')
            logger.info(f"Generated report: {output_path}")
        
        return report_content
    
    def generate_html_report(self, output_path: str = None) -> str:
        """
        Generate interactive HTML report with Plotly charts.
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Path to generated file
        """
        opt = self.optimization_result
        orig_metrics = opt.original_metrics
        best_metrics = opt.best_metrics
        
        # Prepare chart data from improvement history
        chart_data = self._prepare_chart_data(opt)
        
        # Get JavaScript file path (relative to HTML output)
        # HTML files are in optimized_outputs/reports/, assets is at project root
        # So relative path is ../../assets/plotly_rightclick_hide.js
        # Use forward slashes for cross-platform web compatibility
        js_path_str = "../../assets/plotly_rightclick_hide.js"
        
        # Build HTML content
        html_content = self._build_html_content(
            chart_data, 
            orig_metrics, 
            best_metrics, 
            opt,
            js_path_str
        )
        
        # Determine output path
        if output_path is None:
            source_path = Path(self.parse_result.indicator_name.replace(' ', '_').replace('/', '_'))
            output_path = f"optimized_outputs/reports/optimised_{source_path.stem}_report.html"
        
        # Write file
        Path(output_path).write_text(html_content, encoding='utf-8')
        logger.info(f"Generated HTML report: {output_path}")
        
        return output_path
    
    def _prepare_chart_data(self, opt: OptimizationResult) -> Dict[str, Any]:
        """Prepare data for Plotly charts from improvement history."""
        if not opt.improvement_history:
            return {"traces": [], "has_data": False}
        
        # Extract data from improvement history
        elapsed_times = []
        objectives = []
        labels = []
        
        for i, entry in enumerate(opt.improvement_history):
            elapsed = entry.get('elapsed_seconds', i * 10)  # Fallback if missing
            objective = entry.get('objective', 0)
            elapsed_times.append(elapsed)
            objectives.append(objective)
            
            # Create label from params
            params = entry.get('params', {})
            param_strs = [f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" 
                         for k, v in sorted(params.items())]
            label = ", ".join(param_strs[:3])  # Limit to first 3 params for readability
            if len(param_strs) > 3:
                label += "..."
            labels.append(label or f"Trial {i+1}")
        
        # Add baseline point at time 0
        baseline_objective = opt.baseline_objective
        
        return {
            "traces": [
                {
                    "x": [0] + elapsed_times,
                    "y": [baseline_objective] + objectives,
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "Objective Score",
                    "customdata": ["Baseline"] + labels,
                    "line": {"color": "rgb(31, 119, 180)"},
                    "marker": {"size": 6}
                }
            ],
            "has_data": True,
            "x_title": "Elapsed Time (seconds)",
            "y_title": "Objective Score"
        }
    
    def _build_html_content(self, chart_data: Dict[str, Any], orig_metrics: BacktestMetrics, 
                           best_metrics: BacktestMetrics, opt: OptimizationResult, js_path: str) -> str:
        """Build the complete HTML content."""
        
        # Calculate objective scores
        from objective import calculate_objective_score
        orig_objective = calculate_objective_score(orig_metrics)
        best_objective = calculate_objective_score(best_metrics)
        
        # Prepare metrics comparison data
        metrics_comparison = [
            ("MCC (primary)", orig_metrics.mcc, best_metrics.mcc, ".3f"),
            ("ROC AUC", orig_metrics.roc_auc, best_metrics.roc_auc, ".3f"),
            ("Profit Factor", orig_metrics.profit_factor, best_metrics.profit_factor, ".2f"),
            ("Win Rate", orig_metrics.win_rate * 100, best_metrics.win_rate * 100, ".1f"),
            ("Directional Accuracy", orig_metrics.directional_accuracy * 100, best_metrics.directional_accuracy * 100, ".1f"),
            ("Sharpe Ratio", orig_metrics.sharpe_ratio, best_metrics.sharpe_ratio, ".2f"),
        ]
        
        # Format chart JSON
        chart_json = json.dumps(chart_data["traces"]) if chart_data["has_data"] else "[]"
        layout_json = json.dumps({
            "title": "Optimization Progress",
            "xaxis": {"title": chart_data.get("x_title", "Trial Number")},
            "yaxis": {"title": chart_data.get("y_title", "Objective Score")},
            "hovermode": "closest",
            "showlegend": True,
            "margin": {"l": 60, "r": 20, "t": 40, "b": 60}
        })
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Report: {self.parse_result.indicator_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .metrics-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .improvement {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .decline {{
            color: #f44336;
            font-weight: bold;
        }}
        #objective-graph {{
            width: 100%;
            height: 500px;
            margin: 20px 0;
        }}
        #legend-box {{
            margin: 20px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .legend-button {{
            display: inline-block;
            margin: 5px;
            padding: 8px 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .legend-button:hover {{
            background-color: #45a049;
        }}
        .legend-button.hidden {{
            background-color: #ccc;
            opacity: 0.5;
        }}
        .summary {{
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ML Optimization Report: {self.parse_result.indicator_name}</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Trials:</strong> {opt.n_trials} | <strong>Time:</strong> {opt.optimization_time:.1f}s</p>
            <p><strong>Objective Score:</strong> {orig_objective:.4f} → <strong class="improvement">{best_objective:.4f}</strong></p>
            <p><strong>Improvement:</strong> {((best_objective - orig_objective) / orig_objective * 100) if orig_objective > 0 else 0:+.1f}%</p>
        </div>
        
        <h2>Optimization Progress</h2>
        <div id="objective-graph"></div>
        <div id="legend-box"></div>
        
        <h2>Performance Metrics Comparison</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Original</th>
                    <th>Optimized</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for metric_name, orig_val, best_val, fmt in metrics_comparison:
            change = best_val - orig_val
            change_pct = (change / orig_val * 100) if orig_val != 0 else 0
            change_class = "improvement" if change >= 0 else "decline"
            change_sign = "+" if change >= 0 else ""
            
            if "%" in fmt:
                change_str = f"{change_sign}{change:.1f}%"
            else:
                change_str = f"{change_sign}{change:.{fmt.split('.')[1].split('f')[0] if '.' in fmt else '2'}f}"
            
            html += f"""                <tr>
                    <td>{metric_name}</td>
                    <td>{orig_val:{fmt}}</td>
                    <td>{best_val:{fmt}}</td>
                    <td class="{change_class}">{change_str}</td>
                </tr>
"""
        
        html += """            </tbody>
        </table>
        
        <h2>Optimized Parameters</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Original</th>
                    <th>Optimized</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for param_name in sorted(set(list(opt.original_params.keys()) + list(opt.best_params.keys()))):
            orig_val = opt.original_params.get(param_name, "N/A")
            best_val = opt.best_params.get(param_name, "N/A")
            if orig_val != best_val:
                html += f"""                <tr>
                    <td>{param_name}</td>
                    <td>{self._format_value(orig_val)}</td>
                    <td class="improvement">{self._format_value(best_val)}</td>
                    <td>{self._format_value(best_val) if orig_val == "N/A" else f"{self._format_value(orig_val)} → {self._format_value(best_val)}"}</td>
                </tr>
"""
        
        html += """            </tbody>
        </table>
    </div>
    
    <script>
        // Initialize Plotly chart
        var chartData = """ + chart_json + """;
        var layout = """ + layout_json + """;
        
        if (chartData.length > 0) {
            Plotly.newPlot('objective-graph', chartData, layout, {responsive: true});
            
            // Create legend buttons
            var legendBox = document.getElementById('legend-box');
            chartData.forEach(function(trace, index) {
                var button = document.createElement('button');
                button.className = 'legend-button';
                button.setAttribute('data-label', trace.name || 'Trace ' + index);
                button.textContent = trace.name || 'Trace ' + index;
                button.onclick = function() {
                    var visibility = chartData[index].visible === false ? true : false;
                    var update = {visible: visibility};
                    Plotly.restyle('objective-graph', update, index);
                    this.classList.toggle('hidden');
                };
                legendBox.appendChild(button);
            });
        } else {
            document.getElementById('objective-graph').innerHTML = 
                '<p style="text-align: center; color: #999; padding: 50px;">No optimization history data available.</p>';
        }
    </script>
    <script src=\"""" + js_path + """\"></script>
</body>
</html>"""
        
        return html
    
    def print_summary(self):
        """Print a concise summary to console."""
        opt = self.optimization_result
        metrics = opt.best_metrics
        
        print("\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETE: {self.parse_result.indicator_name}")
        print("=" * 60)
        print(f"Peak Forecast Timeframe: {metrics.forecast_horizon} hours")
        print(f"MCC (primary):          {metrics.mcc:.3f} | AUC: {metrics.roc_auc:.3f}")
        print(f"Performance vs Original: {opt.improvement_pf:+.1f}% profit factor improvement")
        print(f"Extreme Move Capture:    {metrics.tail_capture_rate:.1%}")
        print(f"Consistency Score:       {metrics.consistency_score:.2f}")
        print(f"Expected Profitability:  Best during {self._get_optimal_conditions()}")
        print(f"                         Optimal forecast horizon: ~{metrics.forecast_horizon} hours")
        print()
        print("Optimized Parameters:")
        
        for param in self.parse_result.parameters:
            name = param.name
            orig_val = opt.original_params.get(name)
            best_val = opt.best_params.get(name)
            if orig_val != best_val:
                print(f"  {name}: {self._format_value(orig_val)} -> {self._format_value(best_val)}")
        
        print("=" * 60)


def generate_outputs(
    parse_result: ParseResult,
    optimization_result: OptimizationResult,
    source_filename: str,
    output_tag: str = None
) -> Dict[str, str]:
    """
    Generate all output files.
    
    Args:
        parse_result: Parsed Pine Script
        optimization_result: Optimization results
        source_filename: Original Pine Script filename
        
    Returns:
        Dict with paths to generated files
    """
    generator = OutputGenerator(parse_result, optimization_result)
    
    # Generate output paths
    source_path = Path(source_filename)
    base_dir = Path("optimized_outputs")
    pine_dir = base_dir / "pine"
    report_dir = base_dir / "reports"
    pine_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    if output_tag:
        safe_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", output_tag.strip()).strip("_")
    else:
        safe_tag = ""

    suffix = f"_{safe_tag}" if safe_tag else ""
    pine_output = str(pine_dir / f"optimised_{source_path.stem}{suffix}.pine")
    report_output = str(report_dir / f"optimised_{source_path.stem}{suffix}_report.txt")
    html_output = str(report_dir / f"optimised_{source_path.stem}{suffix}_report.html")
    
    # Generate files
    pine_path = generator.generate_optimized_pine(pine_output)
    report_content = generator.generate_report(report_output)
    html_path = generator.generate_html_report(html_output)
    
    # Print summary
    generator.print_summary()
    
    return {
        'pine_script': pine_path,
        'report': report_output,
        'report_content': report_content,
        'html_report': html_path
    }


if __name__ == "__main__":
    # Test with mock data
    from pine_parser import parse_pine_script, Parameter, SignalInfo, SignalType, PositionType
    from backtester import BacktestMetrics
    
    # Create mock parse result
    parse_result = ParseResult(
        parameters=[
            Parameter(name="length", param_type="int", default=14, min_val=1, max_val=100),
            Parameter(name="threshold", param_type="float", default=0.5, min_val=0, max_val=1),
        ],
        signal_info=SignalInfo(
            signal_type=SignalType.THRESHOLD,
            position_type=PositionType.BOTH
        ),
        indicator_name="Test Indicator",
        version=6,
        raw_content='//@version=6\nindicator("Test Indicator")\nlength = input.int(14, "Length")\n'
    )
    
    # Create mock optimization result
    opt_result = OptimizationResult(
        best_params={"length": 21, "threshold": 0.65},
        original_params={"length": 14, "threshold": 0.5},
        best_metrics=BacktestMetrics(
            total_trades=150,
            winning_trades=90,
            losing_trades=60,
            win_rate=0.6,
            profit_factor=1.85,
            sharpe_ratio=1.2,
            directional_accuracy=0.62,
            forecast_horizon=48,
            improvement_over_random=24.0
        ),
        original_metrics=BacktestMetrics(
            total_trades=120,
            winning_trades=60,
            losing_trades=60,
            win_rate=0.5,
            profit_factor=1.2,
            sharpe_ratio=0.8,
            directional_accuracy=0.52,
            forecast_horizon=24
        ),
        n_trials=100,
        optimization_time=120.5,
        improvement_pf=54.2,
        improvement_accuracy=19.2,
        optimal_horizon=48
    )
    
    # Generate outputs
    generator = OutputGenerator(parse_result, opt_result)
    report = generator.generate_report()
    print(report)
