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
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from pine_parser import ParseResult, Parameter
from pine_translator import PineTranslator, IndicatorResult
from backtester import WalkForwardBacktester, BacktestMetrics

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


class OptimizationProgressTracker:
    """Track and report progressive improvement during optimization.
    
    Uses the ORIGINAL CONFIG's performance as baseline, not the first trial.
    This means early trials may show negative improvement until ML finds
    something better than the original.
    """
    
    def __init__(self):
        self.start_time = None
        self.baseline_objective = None  # Original config's performance (set before optimization)
        self.best_objective = None
        self.best_time = None
        # Full history with params: (elapsed, objective, pct_vs_baseline, avg_rate, marginal_rate, params_dict)
        self.improvement_history = []
    
    def set_baseline(self, baseline_objective: float):
        """Set the baseline objective (original config's performance)."""
        self.baseline_objective = baseline_objective
        logger.info(f"Baseline objective (original config): {baseline_objective:.4f}")
    
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
    per_symbol_metrics: Dict[str, Dict[str, BacktestMetrics]] = field(default_factory=dict)  # {symbol: {'original': metrics, 'optimized': metrics}}
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Optimization completed in {self.optimization_time:.1f}s ({self.n_trials} trials)",
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
        data: Dict[str, pd.DataFrame],
        max_trials: Optional[int] = None,  # None = unlimited trials (use timeout)
        timeout_seconds: int = 300,  # 5 minutes default
        n_startup_trials: int = 20,
        pruning_warmup_trials: int = 30,
        min_improvement_threshold: float = 0.1,
        enable_keyboard_interrupt: bool = True
    ):
        """
        Initialize optimizer.
        
        Args:
            parse_result: Parsed Pine Script information
            data: Dict of symbol -> DataFrame with OHLCV data
            max_trials: Maximum optimization trials (None = unlimited, use timeout)
            timeout_seconds: Maximum time for optimization
            n_startup_trials: Random trials before TPE kicks in
            pruning_warmup_trials: Trials before pruning starts
            min_improvement_threshold: Minimum improvement to continue
            enable_keyboard_interrupt: Allow Q key to stop optimization
        """
        self.parse_result = parse_result
        self.data = data
        self.max_trials = max_trials
        self.timeout_seconds = timeout_seconds
        self.n_startup_trials = n_startup_trials
        self.pruning_warmup_trials = pruning_warmup_trials
        self.min_improvement_threshold = min_improvement_threshold
        self.enable_keyboard_interrupt = enable_keyboard_interrupt
        
        # Extract parameter info
        self.parameters = parse_result.parameters
        self.original_params = {p.name: p.default for p in self.parameters}
        
        # Filter to optimizable parameters (exclude display/visual params)
        self.optimizable_params = self._get_optimizable_params()
        
        # Create translators and backtesters for each symbol
        self.translators = {}
        self.backtesters = {}
        
        for symbol, df in data.items():
            self.translators[symbol] = PineTranslator(parse_result, df)
            self.backtesters[symbol] = WalkForwardBacktester(df)
        
        # Tracking
        self.best_objective = 0.0
        self.trial_count = 0
        self.start_time = None
        
        # Progress tracking and keyboard monitoring
        self.progress_tracker = OptimizationProgressTracker()
        self.keyboard_monitor = KeyboardMonitor() if enable_keyboard_interrupt else None
        self.user_stopped = False
    
    def _get_optimizable_params(self) -> List[Parameter]:
        """Filter parameters to only those worth optimizing."""
        skip_keywords = ['show', 'display', 'color', 'style', 'size', 'line']
        
        optimizable = []
        for p in self.parameters:
            name_lower = p.name.lower()
            title_lower = p.title.lower()
            
            # Skip visual/display parameters
            if any(kw in name_lower or kw in title_lower for kw in skip_keywords):
                continue
            
            # Skip bool parameters that are likely display toggles
            if p.param_type == 'bool' and any(kw in name_lower or kw in title_lower for kw in skip_keywords):
                continue
            
            optimizable.append(p)
        
        logger.info(f"Found {len(optimizable)} optimizable parameters out of {len(self.parameters)}")
        return optimizable
    
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
        
        # Evaluate across all symbols
        total_objective = 0.0
        symbol_count = 0
        
        for symbol in self.translators:
            translator = self.translators[symbol]
            backtester = self.backtesters[symbol]
            
            try:
                # Run indicator with trial params
                indicator_result = translator.run_indicator(params)
                
                # Evaluate performance
                use_discrete = self.parse_result.signal_info.buy_conditions or self.parse_result.signal_info.sell_conditions
                metrics = backtester.evaluate_indicator(indicator_result, use_discrete_signals=bool(use_discrete))
                
                # Calculate objective
                obj = backtester.calculate_objective(metrics)
                
                total_objective += obj
                symbol_count += 1
                
            except Exception as e:
                logger.debug(f"Trial failed for {symbol}: {e}")
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
                logger.info(
                    f"Trial {self.trial_count}: FIRST = {avg_objective:.4f} "
                    f"({sign}{pct_vs_original:.2f}% vs original)"
                )
                logger.info(f"  -> Config: {config_str}")
            else:
                marginal_rate_pct = improvement_info['marginal_rate_pct']
                time_since_last = improvement_info['time_since_last_best']
                
                logger.info(
                    f"Trial {self.trial_count}: NEW BEST = {avg_objective:.4f} "
                    f"({sign}{pct_vs_original:.2f}% vs original) | {elapsed_str} | rate: {rate_pct:+.3f}%/s"
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
        original_metrics = self._evaluate_params(self.original_params)
        original_objective = self._calculate_overall_objective(original_metrics)
        
        self.start_time = time.time()
        self.trial_count = 0
        self.best_objective = 0.0
        self.user_stopped = False
        
        # Start progress tracking with original config as baseline
        self.progress_tracker.set_baseline(original_objective)
        self.progress_tracker.start()
        
        # Start keyboard monitoring if enabled
        if self.keyboard_monitor:
            self.keyboard_monitor.start()
        
        # Create Optuna study with TPE sampler
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
        
        # Custom callback to check for user stop request
        def stop_callback(study, trial):
            if self.keyboard_monitor and self.keyboard_monitor.stop_requested:
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
        
        optimization_time = time.time() - self.start_time
        
        # Report if user stopped
        if self.user_stopped or (self.keyboard_monitor and self.keyboard_monitor.stop_requested):
            logger.info(f"Optimization stopped by user after {optimization_time:.1f}s ({self.trial_count} trials)")
            logger.info("Using best parameters found so far...")
        
        # Evaluate original params first (per-symbol)
        original_per_symbol = self._evaluate_params_per_symbol(self.original_params)
        original_metrics = self._aggregate_metrics(list(original_per_symbol.values()))
        original_objective = self._calculate_overall_objective(original_metrics)
        
        # Get best params from study
        best_params_candidate = self.original_params.copy()
        for p in self.optimizable_params:
            if p.name in study.best_params:
                best_params_candidate[p.name] = study.best_params[p.name]
        
        # Evaluate best params candidate (per-symbol)
        optimized_per_symbol = self._evaluate_params_per_symbol(best_params_candidate)
        best_metrics_candidate = self._aggregate_metrics(list(optimized_per_symbol.values()))
        best_objective = self._calculate_overall_objective(best_metrics_candidate)
        
        # Build per-symbol metrics dict
        per_symbol_metrics = {}
        for symbol in self.translators:
            per_symbol_metrics[symbol] = {
                'original': original_per_symbol.get(symbol, BacktestMetrics()),
                'optimized': optimized_per_symbol.get(symbol, BacktestMetrics())
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
        
        result = OptimizationResult(
            best_params=best_params,
            original_params=self.original_params,
            best_metrics=best_metrics,
            original_metrics=original_metrics,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            improvement_pf=improvement_pf,
            improvement_accuracy=improvement_acc,
            optimal_horizon=best_metrics.forecast_horizon,
            study=study,
            improvement_history=self.progress_tracker.get_detailed_history(),
            baseline_objective=self.progress_tracker.baseline_objective or 0.0,
            per_symbol_metrics=per_symbol_metrics
        )
        
        logger.info(f"\n{result.get_summary()}")
        
        # Log improvement trajectory summary
        if self.progress_tracker.improvement_history:
            logger.info("\n" + self.progress_tracker.get_summary())
        
        return result
    
    def _calculate_overall_objective(self, metrics: BacktestMetrics) -> float:
        """Calculate overall objective score for comparison."""
        if metrics.total_trades < 10:
            return 0.0
        
        # Weighted combination - same as backtester
        pf_score = min(metrics.profit_factor, 5.0) / 5.0
        acc_score = max(0, min(1, (metrics.directional_accuracy - 0.5) * 2))
        sharpe_score = min(max(metrics.sharpe_ratio, 0), 3.0) / 3.0
        win_score = metrics.win_rate
        
        return 0.35 * pf_score + 0.30 * acc_score + 0.20 * sharpe_score + 0.15 * win_score
    
    def _aggregate_metrics(self, all_metrics: List[BacktestMetrics]) -> BacktestMetrics:
        """Aggregate metrics from multiple symbols into a single BacktestMetrics."""
        if not all_metrics:
            return BacktestMetrics()
        
        return BacktestMetrics(
            total_trades=sum(m.total_trades for m in all_metrics),
            winning_trades=sum(m.winning_trades for m in all_metrics),
            losing_trades=sum(m.losing_trades for m in all_metrics),
            total_return=np.mean([m.total_return for m in all_metrics]),
            avg_return=np.mean([m.avg_return for m in all_metrics]),
            win_rate=np.mean([m.win_rate for m in all_metrics]),
            profit_factor=np.mean([m.profit_factor for m in all_metrics]),
            sharpe_ratio=np.mean([m.sharpe_ratio for m in all_metrics]),
            max_drawdown=np.max([m.max_drawdown for m in all_metrics]),
            avg_holding_bars=np.mean([m.avg_holding_bars for m in all_metrics]),
            directional_accuracy=np.mean([m.directional_accuracy for m in all_metrics]),
            forecast_horizon=int(np.median([m.forecast_horizon for m in all_metrics])),
            improvement_over_random=np.mean([m.improvement_over_random for m in all_metrics])
        )
    
    def _evaluate_params_per_symbol(self, params: Dict[str, Any]) -> Dict[str, BacktestMetrics]:
        """Evaluate a parameter set and return per-symbol metrics."""
        symbol_metrics = {}
        
        for symbol in self.translators:
            translator = self.translators[symbol]
            backtester = self.backtesters[symbol]
            
            try:
                indicator_result = translator.run_indicator(params)
                use_discrete = bool(self.parse_result.signal_info.buy_conditions or self.parse_result.signal_info.sell_conditions)
                metrics = backtester.evaluate_indicator(indicator_result, use_discrete_signals=use_discrete)
                symbol_metrics[symbol] = metrics
            except Exception as e:
                logger.debug(f"Evaluation failed for {symbol}: {e}")
                continue
        
        return symbol_metrics
    
    def _evaluate_params(self, params: Dict[str, Any]) -> BacktestMetrics:
        """Evaluate a parameter set across all symbols."""
        symbol_metrics = self._evaluate_params_per_symbol(params)
        return self._aggregate_metrics(list(symbol_metrics.values()))


def optimize_indicator(
    parse_result: ParseResult,
    data: Dict[str, pd.DataFrame],
    **kwargs
) -> OptimizationResult:
    """
    Convenience function to run optimization.
    
    Args:
        parse_result: Parsed Pine Script
        data: Dict of symbol -> DataFrame
        **kwargs: Additional arguments for PineOptimizer
        
    Returns:
        OptimizationResult
    """
    optimizer = PineOptimizer(parse_result, data, **kwargs)
    return optimizer.optimize()


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
                'timestamp': pd.date_range('2020-01-01', periods=n, freq='1H'),
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

