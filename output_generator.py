"""
Output Generator
Generates optimized Pine Script files and performance reports.
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from pine_parser import ParseResult, Parameter
from optimizer import OptimizationResult

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
        
        header_lines = [
            "// ===========================================================================",
            "// ML-OPTIMISED PARAMETERS",
            f"// Optimized: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"// Trials: {opt.n_trials} | Time: {opt.optimization_time:.1f}s",
            "// ---------------------------------------------------------------------------",
            f"// Profit Factor: {metrics.profit_factor:.2f} ({opt.improvement_pf:+.1f}% vs original)",
            f"// Win Rate: {metrics.win_rate:.1%}",
            f"// Directional Accuracy: {metrics.directional_accuracy:.1%}",
            f"// Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            f"// Improvement over Random: {metrics.improvement_over_random:+.1f}%",
            "// ---------------------------------------------------------------------------",
            f"// OPTIMAL FORECAST HORIZON: {metrics.forecast_horizon} hours",
            f"// ",
            f"// WHEN TO USE: This indicator is optimized for {self._get_optimal_conditions()}",
            f"// EXPECTED PROFITABILITY: Best results holding positions for {metrics.forecast_horizon}-{metrics.forecast_horizon*2}h",
            "// ---------------------------------------------------------------------------",
            "// PARAMETER CHANGES:",
        ]
        
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
        
        # Based on forecast horizon
        if metrics.forecast_horizon <= 12:
            conditions.append("short-term moves")
        elif metrics.forecast_horizon >= 72:
            conditions.append("swing trades")
        else:
            conditions.append("intraday to multi-day holds")
        
        return ", ".join(conditions) if conditions else "general market conditions"
    
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
            f"  Optimization time:    {opt.optimization_time:.1f} seconds",
            f"  Parameters optimized: {len([p for p in opt.best_params if opt.best_params[p] != opt.original_params.get(p)])}",
            "",
            "-" * 70,
            "PERFORMANCE COMPARISON",
            "-" * 70,
            "",
            f"  {'Metric':<25} {'Original':>12} {'Optimized':>12} {'Change':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}",
        ]
        
        # Add metric comparisons
        metrics_to_compare = [
            ('Profit Factor', orig_metrics.profit_factor, best_metrics.profit_factor, '.2f'),
            ('Win Rate', orig_metrics.win_rate * 100, best_metrics.win_rate * 100, '.1f%'),
            ('Directional Accuracy', orig_metrics.directional_accuracy * 100, best_metrics.directional_accuracy * 100, '.1f%'),
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
                orig_str = f"{orig:.2f}"
                best_str = f"{best:.2f}"
                if orig > 0:
                    change = (best - orig) / orig * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "N/A"
            
            report_lines.append(f"  {name:<25} {orig_str:>12} {best_str:>12} {change_str:>12}")
        
        report_lines.extend([
            "",
            "-" * 70,
            "OPTIMAL FORECAST HORIZON",
            "-" * 70,
            f"  Peak performance at: {best_metrics.forecast_horizon} hours ahead",
            f"  Recommended holding period: {best_metrics.forecast_horizon} - {best_metrics.forecast_horizon * 2} hours",
            "",
            "-" * 70,
            "EXPECTED PROFITABILITY",
            "-" * 70,
            f"  Improvement over random baseline: {best_metrics.improvement_over_random:+.1f}%",
            f"  Improvement over original config: {opt.improvement_pf:+.1f}% (profit factor)",
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
        
        report_lines.extend([
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        report_content = '\n'.join(report_lines)
        
        if output_path:
            Path(output_path).write_text(report_content, encoding='utf-8')
            logger.info(f"Generated report: {output_path}")
        
        return report_content
    
    def print_summary(self):
        """Print a concise summary to console."""
        opt = self.optimization_result
        metrics = opt.best_metrics
        
        print("\n" + "=" * 60)
        print(f"OPTIMIZATION COMPLETE: {self.parse_result.indicator_name}")
        print("=" * 60)
        print(f"Peak Forecast Timeframe: {metrics.forecast_horizon} hours")
        print(f"Performance vs Original: {opt.improvement_pf:+.1f}% profit factor improvement")
        print(f"Performance vs Random:   {metrics.improvement_over_random:+.1f}% improvement")
        print(f"Expected Profitability:  Best during {self._get_optimal_conditions()}")
        print(f"                         Optimal holding period: {metrics.forecast_horizon}-{metrics.forecast_horizon*2} hours")
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
    source_filename: str
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
    pine_output = f"optimised_{source_path.stem}.pine"
    report_output = f"optimised_{source_path.stem}_report.txt"
    
    # Generate files
    pine_path = generator.generate_optimized_pine(pine_output)
    report_content = generator.generate_report(report_output)
    
    # Print summary
    generator.print_summary()
    
    return {
        'pine_script': pine_path,
        'report': report_output,
        'report_content': report_content
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

