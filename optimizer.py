"""
Optuna Optimizer for Pine Script Indicators
Uses TPE (Tree-Parzen Estimator) with early pruning for efficient optimization.
Optimized for Surface Pro with limited resources.

Refactored into modules:
- optimizer_types.py
- optimizer_utils.py
- optimizer_ui.py
- optimizer_tracking.py
- optimizer_core.py
- optimizer_strategies.py
"""

import sys
import logging
import pandas as pd
import numpy as np
import optuna
from typing import Dict, Any, Optional

# Re-export from modules for backward compatibility
from optimizer_types import DataUsageInfo, OptimizationResult
from optimizer_utils import (
    _METRIC_DEFS,
    _METRIC_KEYS,
    _metrics_from_backtest,
    _metric_label,
    _format_param_value,
    _format_params,
    _last_non_none,
    _compute_rate_series,
    get_optimizable_params
)
from optimizer_ui import (
    KeyboardMonitor,
    RealtimeBestPlotter,
    PlotlyRealtimePlotter,
    get_realtime_plotter
)
from optimizer_tracking import OptimizationProgressTracker
from optimizer_core import PineOptimizer
from optimizer_strategies import _optimize_multi_fidelity

from pine_parser import ParseResult

# Configure logging
# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optimize_indicator(
    parse_result: ParseResult,
    data: Dict[str, pd.DataFrame],
    interval: str = "",
    **kwargs
) -> OptimizationResult:
    """
    Convenience function to run optimization.
    
    Args:
        parse_result: Parsed Pine Script
        data: Dict of symbol -> DataFrame
        interval: Timeframe/interval used (e.g., "1h", "4h", "1d")
        **kwargs: Additional arguments for PineOptimizer (including n_jobs, fast_evaluation)
        
    Returns:
        OptimizationResult
    """
    strategy = kwargs.pop("strategy", "tpe").lower()
    if strategy == "multi_fidelity":
        result = _optimize_multi_fidelity(parse_result, data, interval, **kwargs)
    else:
        optimizer = PineOptimizer(parse_result, data, interval=interval, **kwargs)
        result = optimizer.optimize()
    
    # Set interval in result
    result.interval = interval
    result.strategy = strategy
    return result


if __name__ == "__main__":
    # Test with sample data
    from pine_parser import parse_pine_script
    import sys
    
    if len(sys.argv) > 1:
        # Parse Pine Script
        parse_result = parse_pine_script(sys.argv[1])
        
        # Create sample data
        np.random.seed(42)
        n = 3000
        
        data = {}
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            trend = np.cumsum(np.random.randn(n) * 0.1)
            noise = np.random.randn(n) * 0.5
            close = 100 + trend + noise
            
            data[symbol] = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=n, freq='h'),
                'open': close + np.random.randn(n) * 0.2,
                'high': close + np.abs(np.random.randn(n) * 0.3),
                'low': close - np.abs(np.random.randn(n) * 0.3),
                'close': close,
                'volume': np.random.randint(1000, 10000, n).astype(float)
            })
        
        # Run optimization
        result = optimize_indicator(
            parse_result,
            data,
            max_trials=50,
            timeout_seconds=60
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        print(result.get_summary())
        print("\nOptimized Parameters:")
        for name, value in result.best_params.items():
            orig = result.original_params.get(name)
            if orig != value:
                orig_str = f"{orig:.4f}" if isinstance(orig, float) and abs(orig) < 1 else (f"{orig:.2f}" if isinstance(orig, float) else str(orig))
                val_str = f"{value:.4f}" if isinstance(value, float) and abs(value) < 1 else (f"{value:.2f}" if isinstance(value, float) else str(value))
                print(f"  {name}: {orig_str} -> {val_str}")
    else:
        print("Usage: python optimizer.py <pine_script_file>")
