"""
Type definitions and dataclasses for the optimizer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import optuna
from backtester import BacktestMetrics

@dataclass
class DataUsageInfo:
    """Information about how historical data was used in walk-forward validation."""
    total_bars: int
    date_range: Tuple[datetime, datetime]
    n_folds: int
    train_ratio: float
    embargo_bars: int
    folds: List[Dict[str, Any]]  # List of fold details
    total_train_bars: int
    total_test_bars: int
    total_embargo_bars: int
    unused_bars: int
    potential_bias_issues: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict[str, Any]
    original_params: Dict[str, Any]
    best_metrics: BacktestMetrics
    original_metrics: BacktestMetrics
    n_trials: int
    optimization_time: float
    improvement_pf: float  # Profit factor improvement %
    improvement_accuracy: float  # Directional accuracy improvement %
    optimal_horizon: int  # Best forecast horizon
    study: optuna.Study = None
    improvement_history: List[dict] = field(default_factory=list)  # Detailed history with params
    baseline_objective: float = 0.0  # Original config's objective score
    per_symbol_metrics: Dict[str, Dict[str, BacktestMetrics]] = field(default_factory=dict)  # {symbol: {'original': metrics, 'optimized': metrics}} OR {symbol: {timeframe: {'original': metrics, 'optimized': metrics}}}
    timeframes_used: Dict[str, List[str]] = field(default_factory=dict)  # {symbol: [timeframe1, timeframe2, ...]}
    data_usage_info: Dict[str, Dict[str, DataUsageInfo]] = field(default_factory=dict)  # {symbol: {timeframe: DataUsageInfo}}
    datasets_used: List[str] = field(default_factory=list)  # List of datasets used (symbol names, e.g., ["BTCUSDT", "ETHUSDT"])
    interval: str = ""  # Timeframe/interval used (e.g., "1h", "4h", "1d") - may represent multiple intervals
    strategy: str = "tpe"
    sampler_name: str = "tpe"
    timeout_seconds: int = 0
    max_trials: Optional[int] = None
    early_stop_patience: Optional[int] = None
    min_runtime_seconds: int = 0
    stall_seconds: Optional[int] = None
    improvement_rate_floor: float = 0.0
    improvement_rate_window: int = 0
    backtester_overrides: Dict[str, Any] = field(default_factory=dict)
    holdout_ratio: float = 0.0
    holdout_gap_bars: int = 0
    holdout_metrics: Optional[BacktestMetrics] = None
    holdout_original_metrics: Optional[BacktestMetrics] = None
    holdout_per_symbol_metrics: Dict[str, Dict[str, BacktestMetrics]] = field(default_factory=dict)
    holdout_data_usage_info: Dict[str, Dict[str, DataUsageInfo]] = field(default_factory=dict)
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        # Format time nicely
        total_seconds = self.optimization_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds:.1f}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
        
        lines = [
            f"Optimization completed in {time_str} ({self.n_trials} trials)",
            f"",
            f"Performance Comparison:",
            f"  MCC (primary): {self.original_metrics.mcc:.3f} -> {self.best_metrics.mcc:.3f}",
            f"  ROC AUC:       {self.original_metrics.roc_auc:.3f} -> {self.best_metrics.roc_auc:.3f}",
            f"  Profit Factor:  {self.original_metrics.profit_factor:.2f} -> {self.best_metrics.profit_factor:.2f} ({self.improvement_pf:+.1f}%)",
            f"  Win Rate:       {self.original_metrics.win_rate:.1%} -> {self.best_metrics.win_rate:.1%}",
            f"  Dir. Accuracy:  {self.original_metrics.directional_accuracy:.1%} -> {self.best_metrics.directional_accuracy:.1%} ({self.improvement_accuracy:+.1f}%)",
            f"  Sharpe Ratio:   {self.original_metrics.sharpe_ratio:.2f} -> {self.best_metrics.sharpe_ratio:.2f}",
            f"  vs Random:      {self.best_metrics.improvement_over_random:+.1f}%",
            f"",
            f"Optimal Forecast Horizon: {self.optimal_horizon} hours",
        ]
        if self.datasets_used:
            interval_str = f" @ {self.interval}" if self.interval else ""
            lines.append(f"")
            lines.append(f"Historical Datasets Used: {', '.join(sorted(self.datasets_used))}{interval_str}")
        if self.holdout_metrics is not None and self.holdout_original_metrics is not None:
            lines.append("")
            lines.append("Lockbox (OOS) Performance:")
            lines.append(
                f"  MCC (primary): {self.holdout_original_metrics.mcc:.3f} -> "
                f"{self.holdout_metrics.mcc:.3f}"
            )
            lines.append(
                f"  ROC AUC:       {self.holdout_original_metrics.roc_auc:.3f} -> "
                f"{self.holdout_metrics.roc_auc:.3f}"
            )
            lines.append(
                f"  Profit Factor:  {self.holdout_original_metrics.profit_factor:.2f} -> "
                f"{self.holdout_metrics.profit_factor:.2f}"
            )
            lines.append(
                f"  Win Rate:       {self.holdout_original_metrics.win_rate:.1%} -> "
                f"{self.holdout_metrics.win_rate:.1%}"
            )
            lines.append(
                f"  Dir. Accuracy:  {self.holdout_original_metrics.directional_accuracy:.1%} -> "
                f"{self.holdout_metrics.directional_accuracy:.1%}"
            )
        return "\n".join(lines)
