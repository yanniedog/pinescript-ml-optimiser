"""
Optuna Optimizer for Pine Script Indicators
Uses TPE (Tree-Parzen Estimator) with early pruning for efficient optimization.
Optimized for Surface Pro with limited resources.
"""

import numpy as np
import pandas as pd
import optuna
import logging
import time
import sys
import threading
import re
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from pine_parser import ParseResult, Parameter
from pine_translator import PineTranslator, IndicatorResult
from backtester import WalkForwardBacktester, BacktestMetrics, WalkForwardFold
from objective import calculate_objective_score
from datetime import datetime

# Platform-specific keyboard handling
if sys.platform == 'win32':
    import msvcrt
else:
    import select
    import termios
    import tty

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimizable_params(parameters: List[Parameter]) -> List[Parameter]:
    """Filter parameters to only those worth optimizing."""
    skip_keywords = ['show', 'display', 'color', 'style', 'size', 'line']
    optimizable = []
    for p in parameters:
        name_lower = p.name.lower()
        title_lower = p.title.lower()

        # Skip visual/display parameters
        if any(kw in name_lower or kw in title_lower for kw in skip_keywords):
            continue

        # Skip bool parameters that are likely display toggles
        if p.param_type == 'bool' and any(kw in name_lower or kw in title_lower for kw in skip_keywords):
            continue

        optimizable.append(p)

    logger.info(f"Found {len(optimizable)} optimizable parameters out of {len(parameters)}")
    return optimizable


class KeyboardMonitor:
    """Monitor for 'Q' keyboard input to allow early stopping with confirmation."""
    
    def __init__(self):
        self.stop_requested = False
        self._monitor_thread = None
        self._running = False
    
    def start(self):
        """Start monitoring keyboard input."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
    
    def _get_key_windows(self):
        """Get a key press on Windows (non-blocking)."""
        if msvcrt.kbhit():
            return msvcrt.getch()
        return None
    
    def _get_key_confirm_windows(self):
        """Get a key press on Windows (blocking for confirmation)."""
        return msvcrt.getch()
    
    def _monitor_loop(self):
        """Background thread to monitor keyboard input."""
        while self._running:
            try:
                if sys.platform == 'win32':
                    key = self._get_key_windows()
                    if key and key.lower() in [b'q', b'Q']:
                        # Ask for confirmation
                        print("\n[?] Stop optimization and use current best? (Y/N): ", end='', flush=True)
                        confirm = self._get_key_confirm_windows()
                        if confirm and confirm.lower() in [b'y', b'Y']:
                            self.stop_requested = True
                            print("Y")
                            print("[!] Stopping optimization gracefully...")
                            break
                        else:
                            print("N")
                            print("[i] Continuing optimization... (press Q to quit)")
                else:
                    # Unix-like systems
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key.lower() == 'q':
                            print("\n[?] Stop optimization and use current best? (Y/N): ", end='', flush=True)
                            confirm = sys.stdin.read(1)
                            if confirm.lower() == 'y':
                                self.stop_requested = True
                                print("[!] Stopping optimization gracefully...")
                                break
                            else:
                                print("[i] Continuing optimization... (press Q to quit)")
                time.sleep(0.1)
            except Exception:
                pass


class RealtimeBestPlotter:
    """Realtime plot of best objective improvements."""

    def __init__(self):
        self._init_attempted = False
        self._enabled = False
        self._plt = None
        self._fig = None
        self._ax = None
        self._lines = {}
        self._series = {}
        self._indicator_colors = {}
        self._baseline_values = {}
        self._baseline_markers = {}
        self._colors = []
        self._color_index = 0
        self._start_times = {}
        self._best_objectives = {}

    def _ensure_ready(self) -> bool:
        if self._init_attempted:
            return self._enabled
        self._init_attempted = True
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except Exception as exc:
            logger.info("Realtime plot disabled (matplotlib not available): %s", exc)
            self._enabled = False
            return False

        try:
            self._plt = plt
            self._colors = list(cm.tab20.colors)
            self._plt.ion()
            self._fig, self._ax = self._plt.subplots()
            self._ax.set_title("Best Objective Improvement vs Baseline")
            self._ax.set_xlabel("Elapsed time (s)")
            self._ax.set_ylabel("Objective Delta (best - baseline)")
            self._ax.grid(True, alpha=0.3)
            self._fig.tight_layout()
            self._fig.show()
            self._fig.canvas.draw()
            self._enabled = True
            return True
        except Exception as exc:
            logger.info("Realtime plot disabled (matplotlib backend error): %s", exc)
            self._enabled = False
            return False

    def _get_indicator_color(self, indicator_name: str):
        color = self._indicator_colors.get(indicator_name)
        if color is not None:
            return color
        if not self._colors:
            color = (0.2, 0.2, 0.2)
        else:
            color = self._colors[self._color_index % len(self._colors)]
            self._color_index += 1
        self._indicator_colors[indicator_name] = color
        return color

    def start_indicator(self, indicator_name: str) -> None:
        if not self._ensure_ready():
            return
        now = time.time()
        self._start_times[indicator_name] = now
        self._best_objectives.pop(indicator_name, None)
        self._baseline_values.pop(indicator_name, None)

        series = self._series.get(indicator_name)
        if series:
            series["x"].clear()
            series["y"].clear()
            line = self._lines.get(indicator_name)
            if line is not None:
                line.set_data([], [])
            baseline = self._baseline_markers.get(indicator_name)
            if baseline is not None:
                baseline.set_data([], [])
            self._ax.relim()
            self._ax.autoscale_view()
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def set_baseline(self, indicator_name: str, objective: float) -> None:
        if not self._ensure_ready():
            return
        if not np.isfinite(objective):
            return

        self._baseline_values[indicator_name] = objective
        baseline = self._baseline_markers.get(indicator_name)
        if baseline is None:
            color = self._get_indicator_color(indicator_name)
            (baseline,) = self._ax.plot(
                [],
                [],
                linestyle='None',
                marker='x',
                markersize=6,
                alpha=0.7,
                label=f"{indicator_name} baseline (0)",
                color=color
            )
            self._baseline_markers[indicator_name] = baseline
            self._ax.legend(loc="best", fontsize=8)

        series = self._series.get(indicator_name)
        elapsed = series["x"][-1] if series and series["x"] else 0.0
        baseline.set_data([elapsed], [0.0])
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def update(self, indicator_name: str, objective: float) -> None:
        if not self._ensure_ready():
            return

        now = time.time()
        start_time = self._start_times.get(indicator_name)
        if start_time is None:
            start_time = now
            self._start_times[indicator_name] = start_time
        elapsed = now - start_time

        if not np.isfinite(objective):
            return

        best_objective = self._best_objectives.get(indicator_name)
        if best_objective is not None and objective <= best_objective:
            return
        self._best_objectives[indicator_name] = objective

        baseline_value = self._baseline_values.get(indicator_name)
        delta = objective - baseline_value if baseline_value is not None else objective
        series = self._series.setdefault(indicator_name, {"x": [], "y": []})
        series["x"].append(elapsed)
        series["y"].append(delta)

        line = self._lines.get(indicator_name)
        if line is None:
            color = self._get_indicator_color(indicator_name)
            (line,) = self._ax.plot(
                [],
                [],
                marker='o',
                linewidth=1.5,
                markersize=4,
                label=indicator_name,
                color=color
            )
            self._lines[indicator_name] = line
            self._ax.legend(loc="best", fontsize=8)

        line.set_data(series["x"], series["y"])
        baseline = self._baseline_markers.get(indicator_name)
        if baseline is not None:
            baseline.set_data([elapsed], [0.0])
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)


_REALTIME_PLOTTER = None


def get_realtime_plotter() -> RealtimeBestPlotter:
    global _REALTIME_PLOTTER
    if _REALTIME_PLOTTER is None:
        _REALTIME_PLOTTER = RealtimeBestPlotter()
    return _REALTIME_PLOTTER


class OptimizationProgressTracker:
    """Track and report progressive improvement during optimization.
    
    Uses the ORIGINAL CONFIG's performance as baseline, not the first trial.
    This means early trials may show negative improvement until ML finds
    something better than the original.
    """
    
    def __init__(self):
        self.start_time = None
        self.baseline_objective = None  # Original config's performance (set before optimization)
        self.original_params = {}  # Original config's parameters
        self.best_objective = None
        self.best_time = None
        # Full history with params: (elapsed, objective, pct_vs_baseline, avg_rate, marginal_rate, params_dict)
        self.improvement_history = []
    
    def set_baseline(self, baseline_objective: float, original_params: Dict[str, Any] = None):
        """Set the baseline objective (original config's performance)."""
        self.baseline_objective = baseline_objective
        self.original_params = original_params or {}
        
        # Format parameters for display
        if original_params:
            param_parts = []
            for name, value in sorted(original_params.items()):
                if isinstance(value, float):
                    if abs(value) < 0.0001:
                        val_str = f"{value:.2e}"
                    elif abs(value) < 1:
                        val_str = f"{value:.4f}"
                    else:
                        val_str = f"{value:.2f}"
                else:
                    val_str = str(value)
                param_parts.append(f"{name}={val_str}")
            params_str = ", ".join(param_parts)
        else:
            params_str = "N/A"
        
        # Use ANSI bold escape code for terminal output
        BOLD = '\033[1m'
        RESET = '\033[0m'
        logger.info(f"Baseline objective (original config): {BOLD}{baseline_objective:.4f}{RESET}")
        logger.info(f"Original parameters: {params_str}")
    
    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.best_objective = None
        self.best_time = None
        self.improvement_history = []
    
    def update(self, objective: float, params: Dict[str, Any] = None) -> Optional[dict]:
        """
        Update with a new objective value. Returns improvement info if this is a new best.
        
        Args:
            objective: The objective score for this trial
            params: The parameter configuration that achieved this objective
        
        Returns:
            dict with improvement info if new best, None otherwise
        
        Rates explained:
            - improvement_rate_pct: % improvement vs ORIGINAL CONFIG, per second elapsed
              (can be negative initially until ML beats the original)
            - marginal_rate_pct: % improvement from last best trial, per second since last best
              (recent rate - if this drops, you're seeing diminishing returns)
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Use baseline (original config) if set, otherwise first objective
        if self.baseline_objective is None:
            self.baseline_objective = objective
        
        if self.best_objective is None:
            self.best_objective = objective
            self.best_time = current_time
            # Record first trial vs baseline
            pct_vs_baseline = ((objective - self.baseline_objective) / self.baseline_objective * 100) if self.baseline_objective > 0 else 0
            avg_rate = pct_vs_baseline / elapsed if elapsed > 0 else 0
            # Store: (elapsed, objective, pct_vs_baseline, avg_rate, marginal_rate, params)
            self.improvement_history.append((elapsed, objective, pct_vs_baseline, avg_rate, 0, params.copy() if params else {}))
            return {
                'new_objective': objective,
                'old_objective': self.baseline_objective,
                'baseline_objective': self.baseline_objective,
                'pct_improvement_total': pct_vs_baseline,
                'improvement_rate_pct': avg_rate,
                'pct_improvement_marginal': 0,
                'marginal_rate_pct': 0,
                'elapsed_seconds': elapsed,
                'time_since_last_best': 0,
                'is_first': True,
                'params': params.copy() if params else {}
            }
        
        if objective > self.best_objective:
            # Calculate improvement as % vs ORIGINAL CONFIG (baseline)
            pct_vs_baseline = ((objective - self.baseline_objective) / self.baseline_objective * 100) if self.baseline_objective > 0 else 0
            improvement_rate_pct = pct_vs_baseline / elapsed if elapsed > 0 else 0  # %/sec
            
            # Calculate marginal improvement as % of previous best, per second
            time_since_last_best = current_time - self.best_time
            pct_improvement_marginal = ((objective - self.best_objective) / self.best_objective * 100) if self.best_objective > 0 else 0
            marginal_rate_pct = pct_improvement_marginal / time_since_last_best if time_since_last_best > 0 else 0  # %/sec
            
            result = {
                'new_objective': objective,
                'old_objective': self.best_objective,
                'baseline_objective': self.baseline_objective,
                'pct_improvement_total': pct_vs_baseline,  # vs original config
                'improvement_rate_pct': improvement_rate_pct,  # %/sec average vs original
                'pct_improvement_marginal': pct_improvement_marginal,
                'marginal_rate_pct': marginal_rate_pct,  # %/sec recent
                'elapsed_seconds': elapsed,
                'time_since_last_best': time_since_last_best,
                'is_first': False,
                'params': params.copy() if params else {}
            }
            
            # Store: (elapsed, objective, pct_vs_baseline, avg_rate, marginal_rate, params)
            self.improvement_history.append((elapsed, objective, pct_vs_baseline, improvement_rate_pct, marginal_rate_pct, params.copy() if params else {}))
            self.best_objective = objective
            self.best_time = current_time
            
            return result
        
        return None
    
    def get_summary(self) -> str:
        """Get a summary of the improvement trajectory vs original config."""
        if not self.improvement_history:
            return "No improvements recorded."
        
        lines = [f"Improvement History vs Original Config (baseline={self.baseline_objective:.4f}):"]
        for entry in self.improvement_history:
            elapsed, obj, pct_vs_baseline, avg_rate, marginal_rate, params = entry
            sign = "+" if pct_vs_baseline >= 0 else ""
            lines.append(f"  {elapsed:6.1f}s: objective={obj:.4f} ({sign}{pct_vs_baseline:.2f}% vs original)")
            lines.append(f"         avg rate: {avg_rate:+.3f}%/s, marginal: {marginal_rate:.3f}%/s")
        
        # Final improvement summary
        if self.improvement_history:
            final_entry = self.improvement_history[-1]
            final_pct = final_entry[2]
            final_elapsed = final_entry[0]
            if final_elapsed > 0:
                avg_rate = final_pct / final_elapsed
                lines.append(f"\nFinal: {'+' if final_pct >= 0 else ''}{final_pct:.2f}% vs original @ {avg_rate:.3f}%/sec avg rate")
        
        return "\n".join(lines)
    
    def get_detailed_history(self) -> List[dict]:
        """Get detailed improvement history for report generation."""
        history = []
        for entry in self.improvement_history:
            elapsed, obj, pct_vs_baseline, avg_rate, marginal_rate, params = entry
            history.append({
                'elapsed_seconds': elapsed,
                'objective': obj,
                'pct_vs_original': pct_vs_baseline,
                'avg_rate_pct_per_sec': avg_rate,
                'marginal_rate_pct_per_sec': marginal_rate,
                'params': params
            })
        return history


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
        return "\n".join(lines)


class PineOptimizer:
    """
    Optuna-based optimizer for Pine Script indicator parameters.
    
    Uses TPE sampler for sample-efficient Bayesian optimization.
    Implements early pruning to quickly discard poor configurations.
    """
    
    def __init__(
        self,
        parse_result: ParseResult,
        data: Dict[str, pd.DataFrame],  # Can be {symbol: DataFrame} or {symbol: {timeframe: DataFrame}}
        max_trials: Optional[int] = None,  # None = unlimited trials (use timeout)
        timeout_seconds: int = 300,  # 5 minutes default
        n_startup_trials: int = 20,
        pruning_warmup_trials: int = 30,
        min_improvement_threshold: float = 0.1,
        enable_keyboard_interrupt: bool = True,
        interval: str = "",
        sampler_name: str = "tpe",
        early_stop_patience: Optional[int] = None,
        seed_params: Optional[Dict[str, Any]] = None,
        backtester_overrides: Optional[Dict[str, Any]] = None,
        min_runtime_seconds: int = 15,
        stall_seconds: Optional[int] = None,
        improvement_rate_floor: float = 0.05,
        improvement_rate_window: int = 5
    ):
        """
        Initialize optimizer.
        
        Args:
            parse_result: Parsed Pine Script information
            data: Dict of symbol -> DataFrame OR symbol -> {timeframe -> DataFrame} with OHLCV data
            max_trials: Maximum optimization trials (None = unlimited, use timeout)
            timeout_seconds: Maximum time for optimization
            n_startup_trials: Random trials before TPE kicks in
            pruning_warmup_trials: Trials before pruning starts
            min_improvement_threshold: Minimum improvement to continue
            enable_keyboard_interrupt: Allow Q key to stop optimization
            sampler_name: Optuna sampler ("tpe" or "random")
            early_stop_patience: Stop after N trials without improvement
            seed_params: Initial parameter suggestion to evaluate early
            backtester_overrides: Overrides for backtester settings
            min_runtime_seconds: Minimum runtime before early timeout checks
            stall_seconds: Stop if no improvement for this many seconds
            improvement_rate_floor: Minimum avg improvement rate to keep running
            improvement_rate_window: Moving average window for improvement rate
        """
        self.parse_result = parse_result
        self.data = data
        self.max_trials = max_trials
        self.timeout_seconds = timeout_seconds
        self.n_startup_trials = n_startup_trials
        self.pruning_warmup_trials = pruning_warmup_trials
        self.min_improvement_threshold = min_improvement_threshold
        self.enable_keyboard_interrupt = enable_keyboard_interrupt
        self.interval = interval
        self.sampler_name = sampler_name
        self.early_stop_patience = early_stop_patience
        self.seed_params = seed_params
        self.backtester_overrides = backtester_overrides or {}
        self.min_runtime_seconds = min_runtime_seconds
        self.stall_seconds = stall_seconds
        self.improvement_rate_floor = improvement_rate_floor
        self.improvement_rate_window = max(1, improvement_rate_window)
        self.indicator_name = parse_result.indicator_name or "Indicator"
        self.realtime_plotter = get_realtime_plotter()
        
        # Extract parameter info
        self.parameters = parse_result.parameters
        self.original_params = {p.name: p.default for p in self.parameters}
        self.use_discrete_signals = bool(
            self.parse_result.signal_info.buy_conditions or self.parse_result.signal_info.sell_conditions
        )
        
        # Filter to optimizable parameters (exclude display/visual params)
        self.optimizable_params = self._get_optimizable_params()
        
        # Detect if data is multi-timeframe structure
        self.is_multi_timeframe = False
        if data:
            first_value = next(iter(data.values()))
            if isinstance(first_value, dict):
                self.is_multi_timeframe = True
        
        # Create translators and backtesters
        self.translators = {}  # {symbol: translator} or {(symbol, timeframe): translator}
        self.backtesters = {}  # {symbol: backtester} or {(symbol, timeframe): backtester}
        self.data_frames = {}  # {symbol: df} or {(symbol, timeframe): df}
        
        if self.is_multi_timeframe:
            # Multi-timeframe structure: {symbol: {timeframe: DataFrame}}
            for symbol, timeframes_dict in data.items():
                for timeframe, df in timeframes_dict.items():
                    key = (symbol, timeframe)
                    self.translators[key] = PineTranslator(parse_result, df)
                    backtester_kwargs = self._get_backtester_config(timeframe, df)
                    self.backtesters[key] = WalkForwardBacktester(df, **backtester_kwargs)
                    self.data_frames[key] = df
        else:
            # Single-timeframe structure: {symbol: DataFrame}
            for symbol, df in data.items():
                self.translators[symbol] = PineTranslator(parse_result, df)
                backtester_kwargs = self._get_backtester_config(self.interval, df)
                self.backtesters[symbol] = WalkForwardBacktester(df, **backtester_kwargs)
                self.data_frames[symbol] = df
        
        # Tracking
        self.best_objective = 0.0
        self.trial_count = 0
        self.start_time = None
        
        # Progress tracking and keyboard monitoring
        self.progress_tracker = OptimizationProgressTracker()
        self.keyboard_monitor = KeyboardMonitor() if enable_keyboard_interrupt else None
        self.user_stopped = False
        self.last_improvement_trial = None
        self.last_improvement_time = None
        self._plot_initialized = False

    def _parse_interval_seconds(self, interval: str) -> Optional[int]:
        """Convert interval string (e.g., 1h, 4h, 1d) to seconds."""
        if not interval:
            return None
        value_match = re.match(r'^(\d+)([mhdw])$', interval.strip().lower())
        if not value_match:
            return None
        value = int(value_match.group(1))
        unit = value_match.group(2)
        seconds_per_unit = {
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
        }
        return value * seconds_per_unit[unit]

    def _forecast_horizons_for_interval(self, interval: str, length: int) -> List[int]:
        """Pick efficient horizon set based on interval granularity."""
        bar_seconds = self._parse_interval_seconds(interval)
        if bar_seconds is None:
            return WalkForwardBacktester.FORECAST_HORIZONS

        # Use a compact horizon set for slow bars to reduce compute.
        if bar_seconds >= 86400:  # 1d+
            horizons = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        elif bar_seconds >= 14400:  # 4h+
            horizons = [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72]
        else:
            horizons = WalkForwardBacktester.FORECAST_HORIZONS

        # Guard against horizons longer than available data
        max_reasonable = max(1, length // 10)
        return [h for h in horizons if h <= max_reasonable] or [1]

    def _get_backtester_config(self, interval: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Tune backtester settings for interval and dataset size."""
        length = len(df)
        bar_seconds = self._parse_interval_seconds(interval)
        n_folds = 5
        embargo_bars = 72
        min_trades_per_fold = 10

        if bar_seconds is not None:
            # Keep embargo roughly ~72 hours worth of bars, but avoid over-penalizing slow intervals.
            embargo_bars = max(3, int((72 * 3600) / bar_seconds))

            # Reduce folds for slower intervals to save compute.
            if bar_seconds >= 86400 or length < 1500:
                n_folds = 3

            # Lower the minimum trade count for slow intervals to avoid zero baselines.
            if bar_seconds >= 86400:
                min_trades_per_fold = 2
            elif bar_seconds >= 14400:
                min_trades_per_fold = 5

        config = {
            "n_folds": n_folds,
            "embargo_bars": embargo_bars,
            "min_trades_per_fold": min_trades_per_fold,
            "forecast_horizons": self._forecast_horizons_for_interval(interval, length),
        }
        if self.backtester_overrides:
            config.update(self.backtester_overrides)
        return config

    def _get_improvement_rate_stats(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (avg_rate, moving_avg_rate, peak_rate) from improvement history."""
        history = self.progress_tracker.improvement_history
        if not history:
            return None, None, None
        avg_rates = [entry[3] for entry in history]
        avg_rate = sum(avg_rates) / len(avg_rates)
        window = min(self.improvement_rate_window, len(avg_rates))
        moving_avg = sum(avg_rates[-window:]) / window
        peak_rate = max(avg_rates)
        return avg_rate, moving_avg, peak_rate
    
    def _get_optimizable_params(self) -> List[Parameter]:
        """Filter parameters to only those worth optimizing."""
        return get_optimizable_params(self.parameters)
    
    def _suggest_param(self, trial: optuna.Trial, param: Parameter) -> Any:
        """Suggest a value for a parameter using Optuna."""
        min_val, max_val, step = param.get_search_space()
        
        if param.param_type == 'bool':
            return trial.suggest_categorical(param.name, [True, False])
        elif param.param_type == 'int':
            return trial.suggest_int(param.name, int(min_val), int(max_val), step=int(step) if step else 1)
        else:  # float
            return trial.suggest_float(param.name, float(min_val), float(max_val), step=float(step) if step else None)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        Returns a score to maximize (higher is better).
        """
        self.trial_count += 1
        
        # Check for user stop request
        if self.keyboard_monitor and self.keyboard_monitor.stop_requested:
            self.user_stopped = True
            raise optuna.TrialPruned()
        
        # Build parameter dict
        params = self.original_params.copy()
        for p in self.optimizable_params:
            params[p.name] = self._suggest_param(trial, p)
        
        # Evaluate across all symbols/timeframes
        total_objective = 0.0
        symbol_count = 0
        
        for key in self.translators:
            translator = self.translators[key]
            backtester = self.backtesters[key]
            
            try:
                # Run indicator with trial params
                indicator_result = translator.run_indicator(params)
                
                # Evaluate performance
                metrics = backtester.evaluate_indicator(indicator_result, use_discrete_signals=self.use_discrete_signals)
                
                # Calculate objective
                obj = backtester.calculate_objective(metrics)
                
                total_objective += obj
                symbol_count += 1
                
            except Exception as e:
                logger.debug(f"Trial failed for {key}: {e}")
                continue
        
        # Early pruning check - report once at the end
        if symbol_count > 0 and self.trial_count > self.pruning_warmup_trials:
            avg_obj = total_objective / symbol_count
            trial.report(avg_obj, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if symbol_count == 0:
            return 0.0
        
        avg_objective = total_objective / symbol_count
        
        # Track best and report improvement rate vs original config
        improvement_info = self.progress_tracker.update(avg_objective, params)
        if improvement_info:
            elapsed = improvement_info['elapsed_seconds']
            pct_vs_original = improvement_info['pct_improvement_total']
            rate_pct = improvement_info['improvement_rate_pct']
            baseline = improvement_info['baseline_objective']

            if self.realtime_plotter:
                if not self._plot_initialized:
                    self.realtime_plotter.start_indicator(self.indicator_name)
                    self.realtime_plotter.set_baseline(self.indicator_name, baseline)
                    self._plot_initialized = True
                self.realtime_plotter.update(self.indicator_name, avg_objective)
            
            # Format sign for improvement vs original
            sign = "+" if pct_vs_original >= 0 else ""
            
            # Build concise config string (only changed params)
            config_parts = []
            for pname, pval in params.items():
                orig_val = self.original_params.get(pname)
                if orig_val != pval:
                    if isinstance(pval, float):
                        config_parts.append(f"{pname}={pval:.4g}")
                    else:
                        config_parts.append(f"{pname}={pval}")
            config_str = ", ".join(config_parts) if config_parts else "(no changes)"
            
            # Format elapsed time as Xm Ys
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)
            elapsed_str = f"{elapsed_mins}m {elapsed_secs}s" if elapsed_mins > 0 else f"{elapsed_secs}s"
            
            if improvement_info.get('is_first'):
                # First trial
                avg_rate, moving_avg, peak_rate = self._get_improvement_rate_stats()
                logger.info(
                    f"Trial {self.trial_count}: FIRST = {avg_objective:.4f} "
                    f"({sign}{pct_vs_original:.2f}% vs original)"
                )
                if avg_rate is not None and moving_avg is not None:
                    logger.info(
                        f"  -> Avg rate: {avg_rate:+.3f}%/s | Moving avg ({self.improvement_rate_window}): "
                        f"{moving_avg:+.3f}%/s"
                    )
                logger.info(f"  -> Config: {config_str}")
            else:
                avg_rate, moving_avg, peak_rate = self._get_improvement_rate_stats()
                marginal_rate_pct = improvement_info['marginal_rate_pct']
                time_since_last = improvement_info['time_since_last_best']
                
                logger.info(
                    f"Trial {self.trial_count}: NEW BEST = {avg_objective:.4f} "
                    f"({sign}{pct_vs_original:.2f}% vs original) | {elapsed_str} | rate: {rate_pct:+.3f}%/s"
                )
                if avg_rate is not None and moving_avg is not None:
                    logger.info(
                        f"  -> Avg rate: {avg_rate:+.3f}%/s | Moving avg ({self.improvement_rate_window}): "
                        f"{moving_avg:+.3f}%/s"
                    )
                logger.info(f"  -> Config: {config_str}")
                
                # Warn about diminishing returns (only if we're actually improving)
                if pct_vs_original > 0 and len(self.progress_tracker.improvement_history) >= 3:
                    # History format: (elapsed, objective, pct_vs_baseline, avg_rate, marginal_rate, params)
                    recent_pcts = [entry[2] for entry in self.progress_tracker.improvement_history[-3:]]
                    # Check if improvement rate is slowing significantly
                    if len(recent_pcts) >= 2:
                        recent_gain = recent_pcts[-1] - recent_pcts[-2]
                        earlier_gain = recent_pcts[-2] - recent_pcts[-3] if len(recent_pcts) >= 3 else recent_gain
                        if earlier_gain > 0 and recent_gain < earlier_gain * 0.3:
                            logger.info("  -> [!] Diminishing returns - consider pressing Q to quit")
            
            # Also update internal tracking
            self.best_objective = avg_objective
            self.last_improvement_trial = self.trial_count
            self.last_improvement_time = time.time()
        
        return avg_objective
    
    def optimize(self) -> OptimizationResult:
        """
        Run the optimization process.
        
        Returns:
            OptimizationResult with best parameters and metrics
        """
        logger.info(f"Starting optimization with {len(self.optimizable_params)} parameters")
        trials_str = "unlimited" if self.max_trials is None else str(self.max_trials)
        logger.info(f"Max trials: {trials_str}, Timeout: {self.timeout_seconds}s ({self.timeout_seconds/60:.1f} min)")
        
        if self.enable_keyboard_interrupt:
            logger.info("Press Q at any time to stop optimization and use current best results")
        
        # Evaluate original config FIRST to establish baseline
        logger.info("Evaluating original config as baseline...")
        original_per_symbol = self._evaluate_params_per_symbol(self.original_params)
        
        # Aggregate metrics - handle both single and multi-timeframe structures
        if self.is_multi_timeframe:
            all_original_metrics = []
            for symbol_dict in original_per_symbol.values():
                all_original_metrics.extend(symbol_dict.values())
            original_metrics = self._aggregate_metrics(all_original_metrics)
        else:
            original_metrics = self._aggregate_metrics(list(original_per_symbol.values()))
        original_objective = self._calculate_avg_objective(original_per_symbol)
        
        self.start_time = time.time()
        self.trial_count = 0
        self.best_objective = 0.0
        self.user_stopped = False
        
        # Start progress tracking with original config as baseline
        self.progress_tracker.set_baseline(original_objective, self.original_params)
        self.progress_tracker.start()
        self._plot_initialized = False
        self.last_improvement_trial = 0
        self.last_improvement_time = self.start_time
        if self.stall_seconds is None:
            self.stall_seconds = max(10, len(self.optimizable_params) * 4)
        
        # Start keyboard monitoring if enabled
        if self.keyboard_monitor:
            self.keyboard_monitor.start()
        
        self.last_improvement_trial = None
        study = None
        if self.optimizable_params:
            # Create Optuna study with requested sampler
            if self.sampler_name.lower() == "random":
                sampler = optuna.samplers.RandomSampler(seed=42)
            else:
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=self.n_startup_trials,
                    seed=42
                )
            
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.pruning_warmup_trials,
                n_warmup_steps=len(self.data) // 2
            )
            
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )

            if self.seed_params:
                study.enqueue_trial(self.seed_params)
            
            # Custom callback to check for user stop request
            def stop_callback(study, trial):
                if self.keyboard_monitor and self.keyboard_monitor.stop_requested:
                    study.stop()
                if self.early_stop_patience is not None and self.last_improvement_trial is not None:
                    if (trial.number - self.last_improvement_trial) >= self.early_stop_patience:
                        study.stop()
                if self.last_improvement_time is not None:
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    if elapsed >= self.min_runtime_seconds:
                        if (time.time() - self.last_improvement_time) >= self.stall_seconds:
                            avg_rate, moving_avg, peak_rate = self._get_improvement_rate_stats()
                            if avg_rate is None or peak_rate is None:
                                study.stop()
                                return
                            
                            threshold = max(self.improvement_rate_floor, 0.2 * peak_rate)
                            if moving_avg is None or moving_avg < threshold:
                                study.stop()
            
            # Run optimization
            try:
                study.optimize(
                    self._objective,
                    n_trials=self.max_trials,
                    timeout=self.timeout_seconds,
                    show_progress_bar=False,
                    catch=(Exception,),
                    callbacks=[stop_callback] if self.keyboard_monitor else None
                )
            finally:
                # Stop keyboard monitor
                if self.keyboard_monitor:
                    self.keyboard_monitor.stop()
        else:
            logger.info("No optimizable parameters found. Skipping search.")
            if self.keyboard_monitor:
                self.keyboard_monitor.stop()
        
        optimization_time = time.time() - self.start_time
        
        # Report if user stopped
        if self.user_stopped or (self.keyboard_monitor and self.keyboard_monitor.stop_requested):
            logger.info(f"Optimization stopped by user after {optimization_time:.1f}s ({self.trial_count} trials)")
            logger.info("Using best parameters found so far...")
        
        # Get best params from study
        best_params_candidate = self.original_params.copy()
        if study is not None:
            for p in self.optimizable_params:
                if p.name in study.best_params:
                    best_params_candidate[p.name] = study.best_params[p.name]
        
        # Evaluate best params candidate (per-symbol/timeframe)
        optimized_per_symbol = (
            self._evaluate_params_per_symbol(best_params_candidate)
            if study is not None
            else original_per_symbol
        )
        
        # Aggregate metrics - handle both single and multi-timeframe structures
        if self.is_multi_timeframe:
            all_optimized_metrics = []
            for symbol_dict in optimized_per_symbol.values():
                all_optimized_metrics.extend(symbol_dict.values())
            best_metrics_candidate = self._aggregate_metrics(all_optimized_metrics)
        else:
            best_metrics_candidate = self._aggregate_metrics(list(optimized_per_symbol.values()))
        best_objective = self._calculate_avg_objective(optimized_per_symbol)
        
        # Build per-symbol metrics dict - handle both structures
        per_symbol_metrics = {}
        timeframes_used = {}
        data_usage_info = {}
        
        if self.is_multi_timeframe:
            # Multi-timeframe structure: {symbol: {timeframe: {'original': metrics, 'optimized': metrics}}}
            for symbol in original_per_symbol:
                per_symbol_metrics[symbol] = {}
                timeframes_used[symbol] = []
                data_usage_info[symbol] = {}
                
                for timeframe in original_per_symbol[symbol]:
                    per_symbol_metrics[symbol][timeframe] = {
                        'original': original_per_symbol[symbol].get(timeframe, BacktestMetrics()),
                        'optimized': optimized_per_symbol[symbol].get(timeframe, BacktestMetrics())
                    }
                    timeframes_used[symbol].append(timeframe)
                    
                    # Extract data usage info
                    key = (symbol, timeframe)
                    if key in self.backtesters and key in self.data_frames:
                        data_usage_info[symbol][timeframe] = self._extract_data_usage_info(
                            self.backtesters[key],
                            self.data_frames[key]
                        )
        else:
            # Single-timeframe structure: {symbol: {'original': metrics, 'optimized': metrics}}
            for symbol in original_per_symbol:
                per_symbol_metrics[symbol] = {
                    'original': original_per_symbol.get(symbol, BacktestMetrics()),
                    'optimized': optimized_per_symbol.get(symbol, BacktestMetrics())
                }
                
                # Extract data usage info
                if symbol in self.backtesters and symbol in self.data_frames:
                    # For single timeframe, we still need to track which timeframe was used
                    # This will be set from the interval parameter
                    data_usage_info[symbol] = {
                        '': self._extract_data_usage_info(
                            self.backtesters[symbol],
                            self.data_frames[symbol]
                        )
                    }
        
        # Only use optimized params if they're actually better
        if best_objective > original_objective:
            best_params = best_params_candidate
            best_metrics = best_metrics_candidate
            logger.info(f"Optimization improved performance: {original_objective:.4f} -> {best_objective:.4f}")
        else:
            best_params = self.original_params.copy()
            best_metrics = original_metrics
            # If keeping original, set optimized = original in per-symbol
            if self.is_multi_timeframe:
                for symbol in per_symbol_metrics:
                    for timeframe in per_symbol_metrics[symbol]:
                        per_symbol_metrics[symbol][timeframe]['optimized'] = per_symbol_metrics[symbol][timeframe]['original']
            else:
                for symbol in per_symbol_metrics:
                    per_symbol_metrics[symbol]['optimized'] = per_symbol_metrics[symbol]['original']
            logger.info(f"Original params were optimal. Keeping original configuration.")
        
        # Calculate improvements
        if original_metrics.profit_factor > 0:
            improvement_pf = (best_metrics.profit_factor - original_metrics.profit_factor) / original_metrics.profit_factor * 100
        else:
            improvement_pf = 100 if best_metrics.profit_factor > 0 else 0
        
        if original_metrics.directional_accuracy > 0:
            improvement_acc = (best_metrics.directional_accuracy - original_metrics.directional_accuracy) / original_metrics.directional_accuracy * 100
        else:
            improvement_acc = 0
        
        # Get list of datasets used (symbols)
        if self.is_multi_timeframe:
            datasets_used = sorted(list(self.data.keys()))
        else:
            datasets_used = sorted(list(self.data.keys()))
        
        result = OptimizationResult(
            best_params=best_params,
            original_params=self.original_params,
            best_metrics=best_metrics,
            original_metrics=original_metrics,
            n_trials=len(study.trials) if study is not None else 0,
            optimization_time=optimization_time,
            improvement_pf=improvement_pf,
            improvement_accuracy=improvement_acc,
            optimal_horizon=best_metrics.forecast_horizon,
            study=study,
            improvement_history=self.progress_tracker.get_detailed_history(),
            baseline_objective=self.progress_tracker.baseline_objective or 0.0,
            per_symbol_metrics=per_symbol_metrics,
            timeframes_used=timeframes_used,
            data_usage_info=data_usage_info,
            datasets_used=datasets_used,
            sampler_name=self.sampler_name,
            timeout_seconds=self.timeout_seconds,
            max_trials=self.max_trials,
            early_stop_patience=self.early_stop_patience,
            min_runtime_seconds=self.min_runtime_seconds,
            stall_seconds=self.stall_seconds,
            improvement_rate_floor=self.improvement_rate_floor,
            improvement_rate_window=self.improvement_rate_window,
            backtester_overrides=self.backtester_overrides
        )
        
        # Format total time nicely for logging
        total_seconds = optimization_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds:.1f}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds:.1f}s"
        else:
            time_str = f"{seconds:.1f}s"
        
        logger.info(f"\n{result.get_summary()}")
        logger.info(f"Total optimization time: {time_str}")
        if datasets_used:
            interval_str = f" @ {result.interval}" if result.interval else ""
            logger.info(f"Historical datasets used: {', '.join(datasets_used)}{interval_str}")
        
        # Log improvement trajectory summary
        if self.progress_tracker.improvement_history:
            logger.info("\n" + self.progress_tracker.get_summary())
        
        return result
    
    def _calculate_avg_objective(self, per_symbol_metrics: Dict[str, Any]) -> float:
        """Calculate average objective score using backtester settings."""
        total_objective = 0.0
        count = 0

        if self.is_multi_timeframe:
            for symbol, tf_dict in per_symbol_metrics.items():
                for timeframe, metrics in tf_dict.items():
                    key = (symbol, timeframe)
                    backtester = self.backtesters.get(key)
                    if backtester is None:
                        continue
                    total_objective += backtester.calculate_objective(metrics)
                    count += 1
        else:
            for symbol, metrics in per_symbol_metrics.items():
                backtester = self.backtesters.get(symbol)
                if backtester is None:
                    continue
                total_objective += backtester.calculate_objective(metrics)
                count += 1

        return total_objective / count if count > 0 else 0.0

    def _calculate_overall_objective(self, metrics: BacktestMetrics) -> float:
        """Calculate overall objective score for comparison."""
        return calculate_objective_score(metrics)
    
    def _aggregate_metrics(self, all_metrics: List[BacktestMetrics]) -> BacktestMetrics:
        """Aggregate metrics from multiple symbols/timeframes into a single BacktestMetrics."""
        if not all_metrics:
            return BacktestMetrics()
        
        # Filter out empty metrics
        valid_metrics = [m for m in all_metrics if m.total_trades > 0]
        if not valid_metrics:
            return BacktestMetrics()
        
        return BacktestMetrics(
            total_trades=sum(m.total_trades for m in valid_metrics),
            winning_trades=sum(m.winning_trades for m in valid_metrics),
            losing_trades=sum(m.losing_trades for m in valid_metrics),
            total_return=np.mean([m.total_return for m in valid_metrics]),
            avg_return=np.mean([m.avg_return for m in valid_metrics]),
            win_rate=np.mean([m.win_rate for m in valid_metrics]),
            profit_factor=np.mean([m.profit_factor for m in valid_metrics]),
            sharpe_ratio=np.mean([m.sharpe_ratio for m in valid_metrics]),
            max_drawdown=np.max([m.max_drawdown for m in valid_metrics]),
            avg_holding_bars=np.mean([m.avg_holding_bars for m in valid_metrics]),
            directional_accuracy=np.mean([m.directional_accuracy for m in valid_metrics]),
            forecast_horizon=int(np.median([m.forecast_horizon for m in valid_metrics])),
            improvement_over_random=np.mean([m.improvement_over_random for m in valid_metrics]),
            tail_capture_rate=np.mean([m.tail_capture_rate for m in valid_metrics]),
            consistency_score=np.mean([m.consistency_score for m in valid_metrics])
        )
    
    def _extract_data_usage_info(self, backtester: WalkForwardBacktester, df: pd.DataFrame) -> DataUsageInfo:
        """Extract data usage information from a backtester."""
        total_bars = len(df)
        date_range = (df['timestamp'].iloc[0], df['timestamp'].iloc[-1])
        
        folds_detail = []
        total_train_bars = 0
        total_test_bars = 0
        total_embargo_bars = 0
        
        for i, fold in enumerate(backtester.folds):
            train_bars = fold.train_end - fold.train_start
            test_bars = fold.test_end - fold.test_start
            embargo_bars = fold.embargo_bars
            
            train_start_date = df['timestamp'].iloc[fold.train_start] if fold.train_start < len(df) else None
            train_end_date = df['timestamp'].iloc[fold.train_end - 1] if fold.train_end > 0 and fold.train_end <= len(df) else None
            test_start_date = df['timestamp'].iloc[fold.test_start] if fold.test_start < len(df) else None
            test_end_date = df['timestamp'].iloc[fold.test_end - 1] if fold.test_end > 0 and fold.test_end <= len(df) else None
            
            folds_detail.append({
                'fold_num': i + 1,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'train_bars': train_bars,
                'test_bars': test_bars,
                'embargo_bars': embargo_bars,
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'test_start_date': test_start_date,
                'test_end_date': test_end_date
            })
            
            total_train_bars += train_bars
            total_test_bars += test_bars
            total_embargo_bars += embargo_bars
        
        # Calculate unused bars
        used_bars = total_train_bars + total_test_bars + total_embargo_bars
        unused_bars = max(0, total_bars - used_bars)
        
        # Analyze for bias issues
        bias_issues = self._analyze_bias_issues(backtester, total_bars, total_train_bars, total_test_bars, total_embargo_bars, unused_bars)
        
        return DataUsageInfo(
            total_bars=total_bars,
            date_range=date_range,
            n_folds=len(backtester.folds),
            train_ratio=backtester.train_ratio,
            embargo_bars=backtester.embargo_bars,
            folds=folds_detail,
            total_train_bars=total_train_bars,
            total_test_bars=total_test_bars,
            total_embargo_bars=total_embargo_bars,
            unused_bars=unused_bars,
            potential_bias_issues=bias_issues
        )
    
    def _analyze_bias_issues(self, backtester: WalkForwardBacktester, total_bars: int, 
                            total_train_bars: int, total_test_bars: int, 
                            total_embargo_bars: int, unused_bars: int) -> List[str]:
        """Analyze for potential bias issues in walk-forward validation."""
        issues = []
        
        # Check for overlapping test sets
        test_ranges = [(f.test_start, f.test_end) for f in backtester.folds]
        for i, (start1, end1) in enumerate(test_ranges):
            for j, (start2, end2) in enumerate(test_ranges):
                if i != j:
                    # Check for overlap
                    if not (end1 <= start2 or end2 <= start1):
                        issues.append(f"Overlapping test sets detected: Fold {i+1} and Fold {j+1}")
        
        # Check embargo adequacy (should be at least 2-3x forecast horizon)
        median_horizon = 24  # Default, will be updated if available
        if backtester.folds:
            # Estimate from typical horizons
            if backtester.embargo_bars < median_horizon * 2:
                issues.append(f"Embargo period ({backtester.embargo_bars} bars) may be insufficient (recommend at least {median_horizon * 2} bars)")
        
        # Check for unused data
        unused_pct = (unused_bars / total_bars * 100) if total_bars > 0 else 0
        if unused_pct > 10:
            issues.append(f"Warning: {unused_pct:.1f}% of data unused ({unused_bars} bars at end of dataset)")
        
        # Check test set size
        test_pct = (total_test_bars / total_bars * 100) if total_bars > 0 else 0
        if test_pct < 5:
            issues.append(f"Warning: Test set size ({test_pct:.1f}%) may be small for reliable statistics")
        
        # Check for data leakage risk (embargo too small relative to forecast horizon)
        if backtester.embargo_bars < 24:  # Less than 1 day at 1h
            issues.append(f"Warning: Embargo period ({backtester.embargo_bars} bars) is very short, risk of data leakage")
        
        # Check train/test ratio
        train_pct = (total_train_bars / total_bars * 100) if total_bars > 0 else 0
        if train_pct < 40:
            issues.append(f"Warning: Training data ({train_pct:.1f}%) is less than 40%, may lead to overfitting")
        
        return issues
    
    def _evaluate_params_per_symbol(self, params: Dict[str, Any]) -> Dict[str, BacktestMetrics]:
        """Evaluate a parameter set and return per-symbol metrics."""
        symbol_metrics = {}
        
        for key in self.translators:
            translator = self.translators[key]
            backtester = self.backtesters[key]
            
            try:
                indicator_result = translator.run_indicator(params)
                metrics = backtester.evaluate_indicator(indicator_result, use_discrete_signals=self.use_discrete_signals)
                
                if self.is_multi_timeframe:
                    symbol, timeframe = key
                    if symbol not in symbol_metrics:
                        symbol_metrics[symbol] = {}
                    symbol_metrics[symbol][timeframe] = metrics
                else:
                    symbol_metrics[key] = metrics
            except Exception as e:
                logger.debug(f"Evaluation failed for {key}: {e}")
                continue
        
        return symbol_metrics
    
    def _evaluate_params(self, params: Dict[str, Any]) -> BacktestMetrics:
        """Evaluate a parameter set across all symbols."""
        symbol_metrics = self._evaluate_params_per_symbol(params)
        return self._aggregate_metrics(list(symbol_metrics.values()))


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
        **kwargs: Additional arguments for PineOptimizer
        
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


def _optimize_multi_fidelity(
    parse_result: ParseResult,
    data: Dict[str, pd.DataFrame],
    interval: str,
    **kwargs
) -> OptimizationResult:
    """Two-stage optimization: quick subset pass, then full pass seeded by stage 1."""
    timeout_seconds = kwargs.pop("timeout_seconds", 300)
    stage1_budget = max(1, timeout_seconds // 2)
    stage2_budget = max(1, timeout_seconds - stage1_budget)

    # Subset data: first symbol only (or first symbol/timeframe for multi-timeframe)
    first_key = next(iter(data.keys()))
    subset_data = {first_key: data[first_key]}

    logger.info(
        f"Multi-fidelity stage 1/2: subset={first_key}, budget={stage1_budget}s"
    )
    stage1_overrides = {
        "n_folds": 2,
        "embargo_bars": 5,
        "min_trades_per_fold": 2,
        "forecast_horizons": [1, 2, 3, 5, 8, 13],
    }

    stage1_optimizer = PineOptimizer(
        parse_result,
        subset_data,
        interval=interval,
        timeout_seconds=stage1_budget,
        backtester_overrides=stage1_overrides,
        **kwargs
    )
    stage1_result = stage1_optimizer.optimize()

    # Stage 2: full data, seeded with stage 1 best params
    logger.info(
        f"Multi-fidelity stage 2/2: full_data={len(data)} symbols, budget={stage2_budget}s"
    )
    stage2_optimizer = PineOptimizer(
        parse_result,
        data,
        interval=interval,
        timeout_seconds=stage2_budget,
        seed_params=stage1_result.best_params,
        **kwargs
    )
    return stage2_optimizer.optimize()


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
