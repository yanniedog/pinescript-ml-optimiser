"""
Core optimizer logic for Pine Script indicators.
"""

import os
import sys
import re
import math
import time
import logging
import threading
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pine_parser import ParseResult, Parameter
from pine_translator import PineTranslator
from backtester import WalkForwardBacktester, BacktestMetrics
from objective import calculate_objective_score

from optimizer_types import DataUsageInfo, OptimizationResult
from optimizer_utils import _metrics_from_backtest, get_optimizable_params
from optimizer_ui import KeyboardMonitor, get_realtime_plotter
from optimizer_tracking import OptimizationProgressTracker

logger = logging.getLogger(__name__)

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
        improvement_rate_window: int = 5,
        holdout_ratio: float = 0.2,
        holdout_gap_bars: Optional[int] = None,
        indicator_label: Optional[str] = None,
        n_jobs: Optional[int] = None,
        fast_evaluation: bool = False
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
            holdout_ratio: Fraction of data reserved for lockbox evaluation (0 disables)
            holdout_gap_bars: Purge bars between optimization and holdout (None = auto)
            n_jobs: Number of parallel jobs for trial execution (None = auto: min(4, cpu_count()))
            fast_evaluation: Use reduced forecast horizons for faster evaluation during optimization
        """
        self.parse_result = parse_result
        self.data = data
        self.max_trials = max_trials
        self.timeout_seconds = timeout_seconds
        self.n_startup_trials = n_startup_trials
        self.pruning_warmup_trials = pruning_warmup_trials
        self.min_improvement_threshold = min_improvement_threshold
        self.enable_keyboard_interrupt = enable_keyboard_interrupt
        # Normalize interval - extract first interval if comma-separated
        # This ensures _parse_interval_seconds() receives a single interval format
        if interval and ',' in interval:
            # Extract first interval from comma-separated string
            self.interval = interval.split(',')[0].strip()
            logger.debug(f"Extracted first interval '{self.interval}' from comma-separated string '{interval}'")
        else:
            self.interval = interval
        
        self.sampler_name = sampler_name
        self.early_stop_patience = early_stop_patience
        self.seed_params = seed_params
        self.backtester_overrides = backtester_overrides or {}
        self.min_runtime_seconds = min_runtime_seconds
        self.stall_seconds = stall_seconds
        self.improvement_rate_floor = improvement_rate_floor
        self.improvement_rate_window = max(1, improvement_rate_window)
        self.holdout_ratio = max(0.0, min(0.95, holdout_ratio))
        self.holdout_gap_bars = holdout_gap_bars
        self.holdout_enabled = False
        self.holdout_gap_bars_effective = 0
        self.holdout_data = {}
        self.holdout_translators = {}
        self.holdout_backtesters = {}
        self.holdout_data_frames = {}
        self.indicator_name = indicator_label or parse_result.indicator_name or "Indicator"
        self.realtime_plotter = get_realtime_plotter()
        
        # Parallelization settings
        if n_jobs is None:
            # Conservative default for Windows Surface laptops: min(4, cpu_count())
            cpu_count = os.cpu_count() or 4
            self.n_jobs = min(4, cpu_count)
        else:
            self.n_jobs = max(1, n_jobs)  # At least 1 job
        
        self.fast_evaluation = fast_evaluation
        self._evaluation_lock = threading.Lock()  # For thread-safe progress tracking
        
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
        if self.data:
            first_value = next(iter(self.data.values()))
            if isinstance(first_value, dict):
                self.is_multi_timeframe = True

        # Split data for holdout if enabled
        if self.holdout_ratio > 0:
            train_data, holdout_data, gap_bars = self._split_data_for_holdout(self.data)
            self.data = train_data
            if holdout_data:
                self.holdout_enabled = True
                self.holdout_data = holdout_data
                self.holdout_gap_bars_effective = gap_bars
            else:
                self.holdout_ratio = 0.0
                self.holdout_gap_bars_effective = 0

        # Create translators and backtesters
        self.translators = {}  # {symbol: translator} or {(symbol, timeframe): translator}
        self.backtesters = {}  # {symbol: backtester} or {(symbol, timeframe): backtester}
        self.data_frames = {}  # {symbol: df} or {(symbol, timeframe): df}
        
        if self.is_multi_timeframe:
            # Multi-timeframe structure: {symbol: {timeframe: DataFrame}}
            for symbol, timeframes_dict in self.data.items():
                for timeframe, df in timeframes_dict.items():
                    key = (symbol, timeframe)
                    self.translators[key] = PineTranslator(parse_result, df)
                    backtester_kwargs = self._get_backtester_config(timeframe, df)
                    self.backtesters[key] = WalkForwardBacktester(df, **backtester_kwargs)
                    self.data_frames[key] = df
        else:
            # Single-timeframe structure: {symbol: DataFrame}
            for symbol, df in self.data.items():
                self.translators[symbol] = PineTranslator(parse_result, df)
                backtester_kwargs = self._get_backtester_config(self.interval, df)
                self.backtesters[symbol] = WalkForwardBacktester(df, **backtester_kwargs)
                self.data_frames[symbol] = df

        # Create holdout translators/backtesters if holdout is enabled
        if self.holdout_enabled:
            if self.is_multi_timeframe:
                for symbol, timeframes_dict in self.holdout_data.items():
                    for timeframe, df in timeframes_dict.items():
                        key = (symbol, timeframe)
                        self.holdout_translators[key] = PineTranslator(parse_result, df)
                        backtester_kwargs = self._get_backtester_config(timeframe, df)
                        self.holdout_backtesters[key] = WalkForwardBacktester(df, **backtester_kwargs)
                        self.holdout_data_frames[key] = df
            else:
                for symbol, df in self.holdout_data.items():
                    self.holdout_translators[symbol] = PineTranslator(parse_result, df)
                    backtester_kwargs = self._get_backtester_config(self.interval, df)
                    self.holdout_backtesters[symbol] = WalkForwardBacktester(df, **backtester_kwargs)
                    self.holdout_data_frames[symbol] = df
        
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
        self._baseline_metrics = None
        self._baseline_objective = None
        self._baseline_metrics_map = {}
        self._progress_log_interval = 5.0
        self._last_progress_log_time = 0.0

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

        # Use reduced horizons for fast evaluation during optimization
        horizons = self._forecast_horizons_for_interval(interval, length)
        if self.fast_evaluation:
            # Use subset of horizons: roughly every 3rd horizon for faster evaluation
            # Keep first few and last few for coverage
            if len(horizons) > 6:
                step = max(1, len(horizons) // 6)
                reduced = horizons[::step]
                # Ensure we have first and last
                if reduced[0] != horizons[0]:
                    reduced.insert(0, horizons[0])
                if reduced[-1] != horizons[-1]:
                    reduced.append(horizons[-1])
                horizons = sorted(set(reduced))
            else:
                # If already small, use all
                horizons = horizons

        config = {
            "n_folds": n_folds,
            "embargo_bars": embargo_bars,
            "min_trades_per_fold": min_trades_per_fold,
            "forecast_horizons": horizons,
        }
        if self.backtester_overrides:
            config.update(self.backtester_overrides)
        return config

    def _compute_holdout_gap_bars(self, interval: str, df: pd.DataFrame) -> int:
        """Compute a purge gap between optimization and holdout to avoid leakage."""
        if self.holdout_gap_bars is not None:
            return max(0, int(self.holdout_gap_bars))
        config = self._get_backtester_config(interval, df)
        horizons = config.get("forecast_horizons") or WalkForwardBacktester.FORECAST_HORIZONS
        max_horizon = max(horizons) if horizons else 1
        return max(int(config.get("embargo_bars", 0)), int(max_horizon))

    def _split_data_for_holdout(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], int]:
        """Split data into optimization and holdout sets with a purge gap."""
        if self.holdout_ratio <= 0:
            return data, {}, 0

        train_data: Dict[str, Any] = {}
        holdout_data: Dict[str, Any] = {}
        max_gap_bars = 0

        def split_df(label: str, interval: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], int]:
            gap = self._compute_holdout_gap_bars(interval, df)
            total = len(df)
            holdout_bars = int(total * self.holdout_ratio)
            min_holdout_bars = max(200, gap)
            min_train_bars = max(200, gap * 2)

            if holdout_bars < min_holdout_bars:
                logger.warning(
                    f"Holdout disabled for {label}: "
                    f"{holdout_bars} holdout bars < minimum {min_holdout_bars}."
                )
                return df, None, gap

            train_end = total - holdout_bars - gap
            if train_end < min_train_bars:
                logger.warning(
                    f"Holdout disabled for {label}: "
                    f"{train_end} training bars < minimum {min_train_bars} after gap={gap}."
                )
                return df, None, gap

            holdout_start = train_end + gap
            if holdout_start >= total:
                logger.warning(
                    f"Holdout disabled for {label}: holdout start {holdout_start} exceeds data length {total}."
                )
                return df, None, gap

            return df.iloc[:train_end].copy(), df.iloc[holdout_start:].copy(), gap

        if self.is_multi_timeframe:
            for symbol, timeframes in data.items():
                for timeframe, df in timeframes.items():
                    label = f"{symbol}@{timeframe}"
                    train_df, holdout_df, gap = split_df(label, timeframe, df)
                    max_gap_bars = max(max_gap_bars, gap)
                    train_data.setdefault(symbol, {})[timeframe] = train_df
                    if holdout_df is not None and len(holdout_df) > 0:
                        holdout_data.setdefault(symbol, {})[timeframe] = holdout_df
        else:
            for symbol, df in data.items():
                label = symbol
                train_df, holdout_df, gap = split_df(label, self.interval, df)
                max_gap_bars = max(max_gap_bars, gap)
                train_data[symbol] = train_df
                if holdout_df is not None and len(holdout_df) > 0:
                    holdout_data[symbol] = holdout_df

        return train_data, holdout_data, max_gap_bars

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
    
    def _adjust_max_for_step(self, min_val: float, max_val: float, step: Optional[float]) -> float:
        """
        Adjust max value to be compatible with step size.
        """
        if step is None or step == 0:
            return max_val
        
        n = math.floor((max_val - min_val) / step)
        adjusted_max = min_val + n * step
        return adjusted_max
    
    def _suggest_param(self, trial: optuna.Trial, param: Parameter) -> Any:
        """Suggest a value for a parameter using Optuna."""
        min_val, max_val, step = param.get_search_space()
        
        if param.param_type == 'bool':
            return trial.suggest_categorical(param.name, [True, False])
        elif param.param_type == 'int':
            step_int = int(step) if step else 1
            adjusted_max = int(self._adjust_max_for_step(float(min_val), float(max_val), float(step_int)))
            return trial.suggest_int(param.name, int(min_val), adjusted_max, step=step_int)
        else:  # float
            step_float = float(step) if step else None
            adjusted_max = self._adjust_max_for_step(float(min_val), float(max_val), step_float)
            return trial.suggest_float(param.name, float(min_val), adjusted_max, step=step_float)
    
    def _evaluate_symbol(self, key: Any, params: Dict[str, Any]) -> Tuple[Any, Optional[BacktestMetrics], Optional[float]]:
        """
        Evaluate a single symbol/timeframe with given parameters.
        Thread-safe helper for parallel evaluation.
        
        Returns:
            (key, metrics, objective) or (key, None, None) on failure
        """
        translator = self.translators[key]
        backtester = self.backtesters[key]
        
        try:
            # Run indicator with trial params
            indicator_result = translator.run_indicator(params)
            
            # Evaluate performance
            metrics = backtester.evaluate_indicator(indicator_result, use_discrete_signals=self.use_discrete_signals)

            # Calculate objective
            obj = backtester.calculate_objective(metrics)
            
            return (key, metrics, obj)
        except Exception as e:
            logger.debug(f"Trial failed for {key}: {e}")
            return (key, None, None)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        Returns a score to maximize (higher is better).
        Uses parallel evaluation of symbols when multiple symbols are present.
        """
        with self._evaluation_lock:
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
        # Use parallel execution if multiple symbols, sequential if single symbol
        symbol_keys = list(self.translators.keys())
        
        if len(symbol_keys) > 1:
            # Parallel evaluation for multiple symbols
            total_objective = 0.0
            symbol_count = 0
            metrics_list = []
            
            # Use ThreadPoolExecutor for parallel evaluation
            max_workers = min(len(symbol_keys), self.n_jobs * 2)  # Allow more workers for symbol evaluation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all symbol evaluations
                future_to_key = {
                    executor.submit(self._evaluate_symbol, key, params): key
                    for key in symbol_keys
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_key):
                    key, metrics, obj = future.result()
                    if metrics is not None and obj is not None:
                        total_objective += obj
                        symbol_count += 1
                        metrics_list.append(metrics)
        else:
            # Sequential evaluation for single symbol (no overhead)
            total_objective = 0.0
            symbol_count = 0
            metrics_list = []
            
            for key in symbol_keys:
                key_result, metrics, obj = self._evaluate_symbol(key, params)
                if metrics is not None and obj is not None:
                    total_objective += obj
                    symbol_count += 1
                    metrics_list.append(metrics)
        
        # Early pruning check - report once at the end
        if symbol_count > 0 and self.trial_count > self.pruning_warmup_trials:
            avg_obj = total_objective / symbol_count
            trial.report(avg_obj, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if symbol_count == 0:
            return 0.0
        
        avg_objective = total_objective / symbol_count
        now = time.time()
        elapsed_since_start = max(now - self.start_time if self.start_time else 0.0, 0.0)
        elapsed_for_rate = max(elapsed_since_start, 1e-6)
        trials_per_second = self.trial_count / elapsed_for_rate if elapsed_for_rate else 0.0
        
        # Compute full aggregated metrics for every trial (not just improvements)
        aggregated_metrics = self._aggregate_metrics(metrics_list)
        metrics_map = _metrics_from_backtest(aggregated_metrics)
        metrics_map["objective_best"] = avg_objective
        metrics_map["objective_overall"] = calculate_objective_score(aggregated_metrics)
        metrics_map["total_trials"] = self.trial_count
        metrics_map["trials_per_second"] = trials_per_second
        
        # Use complete metrics_map for trial progress tracking
        if self.realtime_plotter:
            self.realtime_plotter.record_trial_progress(
                self.indicator_name,
                self.trial_count,
                elapsed_since_start,
                metrics_map
            )

        # Track best and report improvement rate vs original config
        improvement_info = self.progress_tracker.update(avg_objective, params, trial_number=self.trial_count)
        progress_logged = False
        if improvement_info:
            elapsed = improvement_info['elapsed_seconds']
            pct_vs_original = improvement_info['pct_improvement_total']
            rate_pct = improvement_info['improvement_rate_pct']
            
            if self.realtime_plotter:
                params_for_plot = {
                    p.name: params.get(p.name)
                    for p in self.optimizable_params
                }
                # metrics_map already has total_trials and trials_per_second from earlier computation
                self.realtime_plotter.update(
                    self.indicator_name,
                    avg_objective,
                    metrics_map,
                    params_for_plot,
                    trial_number=self.trial_count
                )
            
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
                best_trial = improvement_info.get('best_trial_number', self.trial_count)
                logger.info(
                    f"Trial {self.trial_count}: FIRST = {avg_objective:.4f} "
                    f"({sign}{pct_vs_original:.2f}% vs original) | BEST TRIAL: {best_trial}"
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
                best_trial = improvement_info.get('best_trial_number', self.trial_count)
                previous_best_trial = improvement_info.get('previous_best_trial_number')
                
                logger.info(
                    f"Trial {self.trial_count}: NEW BEST = {avg_objective:.4f} "
                    f"({sign}{pct_vs_original:.2f}% vs original) | {elapsed_str} | rate: {rate_pct:+.3f}%/s | BEST TRIAL: {best_trial}"
                )
                if previous_best_trial is not None:
                    logger.info(f"  -> Previous best was Trial {previous_best_trial}")
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
            progress_logged = True
            self._last_progress_log_time = now
        
        if not progress_logged and (now - self._last_progress_log_time) >= self._progress_log_interval:
            best_trial = self.progress_tracker.best_trial_number
            best_trial_str = f" | BEST: Trial {best_trial}" if best_trial is not None else ""
            logger.info(
                f"Trial {self.trial_count}: total={self.trial_count} | trials/sec={trials_per_second:.2f} | "
                f"objective={avg_objective:.4f} | elapsed={elapsed_since_start:.1f}s{best_trial_str}"
            )
            self._last_progress_log_time = now

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
        # Always use full horizons for baseline evaluation (fair comparison)
        logger.info("Evaluating original config as baseline...")
        if self.fast_evaluation:
            # Temporarily disable fast_evaluation for baseline
            original_fast_eval = self.fast_evaluation
            self.fast_evaluation = False
            
            # Recreate backtesters with full horizons for baseline
            full_horizon_backtesters = {}
            if self.is_multi_timeframe:
                for symbol, timeframes_dict in self.data.items():
                    for timeframe, df in timeframes_dict.items():
                        key = (symbol, timeframe)
                        backtester_kwargs = self._get_backtester_config(timeframe, df)
                        full_horizon_backtesters[key] = WalkForwardBacktester(df, **backtester_kwargs)
            else:
                for symbol, df in self.data.items():
                    backtester_kwargs = self._get_backtester_config(self.interval, df)
                    full_horizon_backtesters[symbol] = WalkForwardBacktester(df, **backtester_kwargs)
            
            original_per_symbol = self._evaluate_params_per_symbol_for(
                self.original_params,
                self.translators,
                full_horizon_backtesters
            )
            
            # Restore fast_evaluation for optimization
            self.fast_evaluation = original_fast_eval
        else:
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

        self._baseline_metrics = original_metrics
        self._baseline_objective = original_objective
        self._baseline_metrics_map = _metrics_from_backtest(original_metrics)
        self._baseline_metrics_map["objective_best"] = original_objective
        self._baseline_metrics_map["objective_overall"] = calculate_objective_score(original_metrics)
        if self.realtime_plotter:
            self.realtime_plotter.start_indicator(self.indicator_name)
            self.realtime_plotter.set_baseline(self.indicator_name, original_objective)
            if self._baseline_metrics_map:
                self.realtime_plotter.set_baseline_metrics(self.indicator_name, self._baseline_metrics_map)

        self.start_time = time.time()
        self.trial_count = 0
        self.best_objective = 0.0
        self.user_stopped = False
        
        # Start progress tracking with original config as baseline
        self.progress_tracker.set_baseline(original_objective, self.original_params)
        self.progress_tracker.start()
        self.last_improvement_trial = 0
        # Don't start the stall timer until the first improvement completes
        self.last_improvement_time = None
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
                    n_jobs=self.n_jobs,
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
            try:
                best_params_candidate.update(study.best_params)
            except ValueError:
                pass  # No trials or all pruned
        
        # Final evaluation of best parameters with FULL BACKTEST SETTINGS (no shortcuts)
        logger.info("Performing final evaluation of best parameters...")
        if self.fast_evaluation:
            logger.info("  (Switching off fast_evaluation for final check)")
            original_fast_eval = self.fast_evaluation
            self.fast_evaluation = False
            
            # Recreate backtesters with full horizons
            full_horizon_backtesters = {}
            if self.is_multi_timeframe:
                for symbol, timeframes_dict in self.data.items():
                    for timeframe, df in timeframes_dict.items():
                        key = (symbol, timeframe)
                        backtester_kwargs = self._get_backtester_config(timeframe, df)
                        full_horizon_backtesters[key] = WalkForwardBacktester(df, **backtester_kwargs)
            else:
                for symbol, df in self.data.items():
                    backtester_kwargs = self._get_backtester_config(self.interval, df)
                    full_horizon_backtesters[symbol] = WalkForwardBacktester(df, **backtester_kwargs)
            
            # Evaluate best
            best_per_symbol = self._evaluate_params_per_symbol_for(
                best_params_candidate,
                self.translators,
                full_horizon_backtesters
            )
            
            self.fast_evaluation = original_fast_eval
        else:
            best_per_symbol = self._evaluate_params_per_symbol(best_params_candidate)
        
        if self.is_multi_timeframe:
            all_best_metrics = []
            for symbol_dict in best_per_symbol.values():
                all_best_metrics.extend(symbol_dict.values())
            best_metrics = self._aggregate_metrics(all_best_metrics)
        else:
            best_metrics = self._aggregate_metrics(list(best_per_symbol.values()))
        
        # If optimization failed to improve, fallback to original
        best_obj_score = calculate_objective_score(best_metrics)
        if best_obj_score < original_objective and not self.user_stopped:
            logger.info(f"Optimization did not improve baseline ({best_obj_score:.4f} < {original_objective:.4f}). Reverting.")
            best_params = self.original_params
            best_metrics = original_metrics
            best_per_symbol = original_per_symbol
        else:
            best_params = best_params_candidate
            
        # Compare metrics
        imp_pf = (
            (best_metrics.profit_factor - original_metrics.profit_factor) 
            / original_metrics.profit_factor * 100 
            if original_metrics.profit_factor > 0 else 0
        )
        imp_acc = (
            (best_metrics.directional_accuracy - original_metrics.directional_accuracy)
            / original_metrics.directional_accuracy * 100
            if original_metrics.directional_accuracy > 0 else 0
        )
        
        # Holdout evaluation
        holdout_metrics = None
        holdout_original_metrics = None
        holdout_per_symbol = {}
        
        if self.holdout_enabled:
            logger.info(f"Evaluating on LOCKBOX holdout data ({self.holdout_ratio:.0%})...")
            # Override embargo for holdout to match effective gap
            # This ensures consistent testing conditions
            
            # Create special backtesters for holdout with forced embargo
            holdout_eval_backtesters = {}
            if self.is_multi_timeframe:
                for key, bt in self.holdout_backtesters.items():
                    # Copy and modify
                    # We can't easily clone, so we rely on the gap being sufficient
                    # in the split_data logic.
                    holdout_eval_backtesters[key] = bt
            else:
                for key, bt in self.holdout_backtesters.items():
                    holdout_eval_backtesters[key] = bt

            # Evaluate original params on holdout
            holdout_orig_per_symbol = self._evaluate_holdout_params_per_symbol(self.original_params)
            if self.is_multi_timeframe:
                all_holdout_orig = []
                for symbol_dict in holdout_orig_per_symbol.values():
                    all_holdout_orig.extend(symbol_dict.values())
                holdout_original_metrics = self._aggregate_metrics(all_holdout_orig)
            else:
                holdout_original_metrics = self._aggregate_metrics(list(holdout_orig_per_symbol.values()))
            
            # Evaluate best params on holdout
            holdout_per_symbol = self._evaluate_holdout_params_per_symbol(best_params)
            if self.is_multi_timeframe:
                all_holdout_best = []
                for symbol_dict in holdout_per_symbol.values():
                    all_holdout_best.extend(symbol_dict.values())
                holdout_metrics = self._aggregate_metrics(all_holdout_best)
            else:
                holdout_metrics = self._aggregate_metrics(list(holdout_per_symbol.values()))

            logger.info(f"Holdout MCC: {holdout_original_metrics.mcc:.3f} -> {holdout_metrics.mcc:.3f}")
            logger.info(f"Holdout AUC: {holdout_original_metrics.roc_auc:.3f} -> {holdout_metrics.roc_auc:.3f}")

        # Extract data usage info
        data_usage_info = {}
        if self.is_multi_timeframe:
            for symbol, timeframes_dict in self.data_frames.items():
                # self.data_frames is {(symbol, timeframe): df}
                sym, tf = symbol
                backtester = self.backtesters[symbol]
                info = self._extract_data_usage_info(backtester, timeframes_dict)
                data_usage_info.setdefault(sym, {})[tf] = info
        else:
            for symbol, df in self.data_frames.items():
                backtester = self.backtesters[symbol]
                info = self._extract_data_usage_info(backtester, df)
                data_usage_info.setdefault(symbol, {})[self.interval] = info

        holdout_data_usage_info = {}
        if self.holdout_enabled:
            if self.is_multi_timeframe:
                for symbol, timeframes_dict in self.holdout_data_frames.items():
                    # self.holdout_data_frames is {(symbol, timeframe): df}
                    sym, tf = symbol
                    backtester = self.holdout_backtesters[symbol]
                    info = self._extract_data_usage_info(backtester, timeframes_dict)
                    holdout_data_usage_info.setdefault(sym, {})[tf] = info
            else:
                for symbol, df in self.holdout_data_frames.items():
                    backtester = self.holdout_backtesters[symbol]
                    info = self._extract_data_usage_info(backtester, df)
                    holdout_data_usage_info.setdefault(symbol, {})[self.interval] = info

        # Construct result
        datasets_used = sorted(list(set(
            s for s in (self.data.keys() if self.is_multi_timeframe else self.data.keys())
        )))
        
        # Prepare per-symbol metrics structure
        # {symbol: {'original': metrics, 'optimized': metrics}}
        # OR {symbol: {timeframe: {'original': metrics, 'optimized': metrics}}}
        per_symbol_metrics_result = {}
        timeframes_used_result = {}
        
        if self.is_multi_timeframe:
            for symbol, timeframes in original_per_symbol.items():
                per_symbol_metrics_result[symbol] = {}
                timeframes_used_result[symbol] = []
                for timeframe, orig_m in timeframes.items():
                    best_m = best_per_symbol.get(symbol, {}).get(timeframe)
                    per_symbol_metrics_result[symbol][timeframe] = {
                        "original": orig_m,
                        "optimized": best_m
                    }
                    timeframes_used_result[symbol].append(timeframe)
        else:
            for symbol, orig_m in original_per_symbol.items():
                best_m = best_per_symbol.get(symbol)
                per_symbol_metrics_result[symbol] = {
                    "original": orig_m,
                    "optimized": best_m
                }
                # Single interval used
                timeframes_used_result[symbol] = [self.interval]

        return OptimizationResult(
            best_params=best_params,
            original_params=self.original_params,
            best_metrics=best_metrics,
            original_metrics=original_metrics,
            n_trials=self.trial_count,
            optimization_time=optimization_time,
            improvement_pf=imp_pf,
            improvement_accuracy=imp_acc,
            optimal_horizon=best_metrics.forecast_horizon,
            study=study,
            improvement_history=self.progress_tracker.get_detailed_history(),
            baseline_objective=self._baseline_objective,
            per_symbol_metrics=per_symbol_metrics_result,
            timeframes_used=timeframes_used_result,
            data_usage_info=data_usage_info,
            datasets_used=datasets_used,
            interval=self.interval,
            strategy="tpe",
            sampler_name=self.sampler_name,
            timeout_seconds=self.timeout_seconds,
            max_trials=self.max_trials,
            early_stop_patience=self.early_stop_patience,
            min_runtime_seconds=self.min_runtime_seconds,
            stall_seconds=self.stall_seconds,
            improvement_rate_floor=self.improvement_rate_floor,
            improvement_rate_window=self.improvement_rate_window,
            backtester_overrides=self.backtester_overrides,
            holdout_ratio=self.holdout_ratio,
            holdout_gap_bars=self.holdout_gap_bars_effective,
            holdout_metrics=holdout_metrics,
            holdout_original_metrics=holdout_original_metrics,
            holdout_per_symbol_metrics=holdout_per_symbol if self.holdout_enabled else {},
            holdout_data_usage_info=holdout_data_usage_info
        )

    def _calculate_avg_objective(self, symbol_metrics: Dict[str, Any]) -> float:
        """Calculate average objective across all symbols/timeframes."""
        objectives = []
        if self.is_multi_timeframe:
            for tf_dict in symbol_metrics.values():
                for m in tf_dict.values():
                    if m:
                        objectives.append(calculate_objective_score(m))
        else:
            for m in symbol_metrics.values():
                if m:
                    objectives.append(calculate_objective_score(m))
        
        if not objectives:
            return 0.0
        return sum(objectives) / len(objectives)
    
    def _calculate_overall_objective(self, metrics: BacktestMetrics) -> float:
        """Calculate overall objective score for comparison."""
        return calculate_objective_score(metrics)
    
    def _aggregate_metrics(self, all_metrics: List[BacktestMetrics]) -> BacktestMetrics:
        """Aggregate metrics from multiple symbols/timeframes into a single BacktestMetrics."""
        if not all_metrics:
            return BacktestMetrics()
        
        # Filter out empty metrics and None values
        valid_metrics = [m for m in all_metrics if m is not None and m.total_trades > 0]
        if not valid_metrics:
            return BacktestMetrics()
        
        total_tp = sum(m.tp for m in valid_metrics)
        total_tn = sum(m.tn for m in valid_metrics)
        total_fp = sum(m.fp for m in valid_metrics)
        total_fn = sum(m.fn for m in valid_metrics)
        total_samples = sum(m.classification_samples for m in valid_metrics)
        if total_samples > 0:
            roc_auc = sum(m.roc_auc * m.classification_samples for m in valid_metrics) / total_samples
        else:
            roc_auc = 0.5

        denom = float(
            (total_tp + total_fp)
            * (total_tp + total_fn)
            * (total_tn + total_fp)
            * (total_tn + total_fn)
        )
        mcc = (total_tp * total_tn - total_fp * total_fn) / np.sqrt(denom) if denom > 0 else 0.0

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
            consistency_score=np.mean([m.consistency_score for m in valid_metrics]),
            mcc=mcc,
            roc_auc=roc_auc,
            classification_samples=total_samples,
            positive_labels=sum(m.positive_labels for m in valid_metrics),
            negative_labels=sum(m.negative_labels for m in valid_metrics),
            tp=total_tp,
            tn=total_tn,
            fp=total_fp,
            fn=total_fn,
            mcc_threshold=float(np.median([m.mcc_threshold for m in valid_metrics])) if valid_metrics else 0.0
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
    
    def _evaluate_params_per_symbol_for(
        self,
        params: Dict[str, Any],
        translators: Dict[Any, PineTranslator],
        backtesters: Dict[Any, WalkForwardBacktester]
    ) -> Dict[str, BacktestMetrics]:
        """Evaluate a parameter set and return per-symbol metrics for a given dataset."""
        symbol_metrics: Dict[str, Any] = {}

        for key in translators:
            translator = translators[key]
            backtester = backtesters[key]

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

    def _evaluate_params_per_symbol(self, params: Dict[str, Any]) -> Dict[str, BacktestMetrics]:
        """Evaluate a parameter set and return per-symbol metrics."""
        return self._evaluate_params_per_symbol_for(params, self.translators, self.backtesters)

    def _evaluate_holdout_params_per_symbol(self, params: Dict[str, Any]) -> Dict[str, BacktestMetrics]:
        """Evaluate a parameter set on the holdout dataset."""
        return self._evaluate_params_per_symbol_for(params, self.holdout_translators, self.holdout_backtesters)
    
    def _evaluate_params(self, params: Dict[str, Any]) -> BacktestMetrics:
        """Evaluate a parameter set across all symbols."""
        symbol_metrics = self._evaluate_params_per_symbol(params)
        return self._aggregate_metrics(list(symbol_metrics.values()))
