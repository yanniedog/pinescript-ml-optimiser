"""
Progress tracking for the optimizer.
"""

import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class OptimizationProgressTracker:
    """Track and report progressive improvement during optimization.
    
    Uses the ORIGINAL CONFIG's performance as baseline, not the first trial.
    This means early trials may show negative improvement until ML finds
    something better than the original.
    
    Memory-optimized for long-running optimizations.
    """
    
    MAX_HISTORY_SIZE = 200  # Limit history to prevent memory bloat
    
    def __init__(self):
        self.start_time = None
        self.baseline_objective = None  # Original config's performance (set before optimization)
        self.original_params = {}  # Original config's parameters
        self.best_objective = None
        self.best_time = None
        self.best_trial_number = None  # Trial number that achieved the best objective
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
        self.best_trial_number = None
        self.improvement_history = []
    
    def update(self, objective: float, params: Dict[str, Any] = None, trial_number: Optional[int] = None) -> Optional[dict]:
        """
        Update with a new objective value. Returns improvement info if this is a new best.
        
        Args:
            objective: The objective score for this trial
            params: The parameter configuration that achieved this objective
            trial_number: The trial number (for tracking which trial is best)
        
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
            self.best_trial_number = trial_number
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
                'trial_number': trial_number,
                'best_trial_number': trial_number,
                'params': params.copy() if params else {}
            }
        
        if objective > self.best_objective:
            # Store old values before updating
            old_objective = self.best_objective
            old_best_trial = self.best_trial_number
            
            # Calculate improvement as % vs ORIGINAL CONFIG (baseline)
            pct_vs_baseline = ((objective - self.baseline_objective) / self.baseline_objective * 100) if self.baseline_objective > 0 else 0
            improvement_rate_pct = pct_vs_baseline / elapsed if elapsed > 0 else 0  # %/sec
            
            # Calculate marginal improvement as % of previous best, per second
            time_since_last_best = current_time - self.best_time
            pct_improvement_marginal = ((objective - old_objective) / old_objective * 100) if old_objective > 0 else 0
            marginal_rate_pct = pct_improvement_marginal / time_since_last_best if time_since_last_best > 0 else 0  # %/sec
            
            # Update best values
            self.best_objective = objective
            self.best_time = current_time
            self.best_trial_number = trial_number
            
            result = {
                'new_objective': objective,
                'old_objective': old_objective,
                'baseline_objective': self.baseline_objective,
                'pct_improvement_total': pct_vs_baseline,  # vs original config
                'improvement_rate_pct': improvement_rate_pct,  # %/sec average vs original
                'pct_improvement_marginal': pct_improvement_marginal,
                'marginal_rate_pct': marginal_rate_pct,  # %/sec recent
                'elapsed_seconds': elapsed,
                'time_since_last_best': time_since_last_best,
                'is_first': False,
                'trial_number': trial_number,
                'best_trial_number': trial_number,
                'previous_best_trial_number': old_best_trial,
                'params': params.copy() if params else {}
            }
            
            # Store: (elapsed, objective, pct_vs_baseline, avg_rate, marginal_rate, params)
            self.improvement_history.append((elapsed, objective, pct_vs_baseline, improvement_rate_pct, marginal_rate_pct, params.copy() if params else {}))
            
            # Trim history if it gets too long (keep first 10 and last N-10 entries)
            if len(self.improvement_history) > self.MAX_HISTORY_SIZE:
                self._trim_history()
            
            return result
        
        return None
    
    def _trim_history(self):
        """Trim history to prevent memory bloat while preserving key data points."""
        if len(self.improvement_history) <= self.MAX_HISTORY_SIZE:
            return
        
        # Keep first 10 entries (early history) and most recent entries
        keep_early = 10
        keep_total = self.MAX_HISTORY_SIZE
        keep_recent = keep_total - keep_early
        
        first_entries = self.improvement_history[:keep_early]
        recent_entries = self.improvement_history[-keep_recent:]
        self.improvement_history = first_entries + recent_entries
    
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
