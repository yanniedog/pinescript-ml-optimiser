"""
Walk-Forward Backtester
Implements walk-forward cross-validation with no look-ahead bias.
Calculates profitability metrics for indicator optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from pine_translator import IndicatorResult
from objective import calculate_objective_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Represents a single trade."""
    entry_bar: int
    exit_bar: int
    direction: TradeDirection
    entry_price: float
    exit_price: float
    
    @property
    def return_pct(self) -> float:
        if self.direction == TradeDirection.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100
    
    @property
    def holding_bars(self) -> int:
        return self.exit_bar - self.entry_bar


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_return: float = 0.0
    avg_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_bars: float = 0.0
    directional_accuracy: float = 0.0  # How well signals predict direction
    forecast_horizon: int = 0  # Optimal forecast horizon in bars
    improvement_over_random: float = 0.0
    tail_capture_rate: float = 0.0  # Ability to capture extreme moves
    consistency_score: float = 0.0  # Stability across walk-forward folds
    trades: List[Trade] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_return': self.total_return,
            'avg_return': self.avg_return,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_holding_bars': self.avg_holding_bars,
            'directional_accuracy': self.directional_accuracy,
            'forecast_horizon': self.forecast_horizon,
            'improvement_over_random': self.improvement_over_random,
            'tail_capture_rate': self.tail_capture_rate,
            'consistency_score': self.consistency_score
        }


@dataclass
class WalkForwardFold:
    """Represents a single fold in walk-forward validation."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_bars: int = 72  # 3 days at 1H timeframe


class WalkForwardBacktester:
    """
    Walk-forward backtester with strict no look-ahead bias.
    
    Key features:
    - Walk-forward cross-validation with embargo period
    - Multiple forecast horizon testing
    - Directional accuracy measurement
    - Profit factor and Sharpe ratio calculation
    
    CRITICAL: Each fold selects its optimal forecast horizon using ONLY its training data.
    Test data is never used for horizon selection, preventing lookahead bias. The selected
    horizon is then used to evaluate that fold's test data.
    """
    
    # Forecast horizons to test (in bars/hours for 1H data)
    # Broad continuum from 1 hour to 1 week (168 hours)
    FORECAST_HORIZONS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 168]
    EXTREME_RETURN_PERCENTILE = 0.8
    SIGNAL_STRENGTH_PERCENTILE = 0.75
    
    def __init__(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        embargo_bars: int = 72,
        train_ratio: float = 0.6,
        min_trades_per_fold: int = 10,
        forecast_horizons: Optional[List[int]] = None
    ):
        """
        Initialize backtester.
        
        Args:
            df: DataFrame with OHLCV data
            n_folds: Number of walk-forward folds
            embargo_bars: Bars to exclude between train/test (prevents leakage)
            train_ratio: Ratio of data for training in each fold
            min_trades_per_fold: Minimum trades required for valid fold
        """
        self.df = df
        self.n_folds = n_folds
        self.embargo_bars = embargo_bars
        self.train_ratio = train_ratio
        self.min_trades_per_fold = min_trades_per_fold
        self.forecast_horizons = forecast_horizons or self.FORECAST_HORIZONS
        
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.length = len(df)
        
        # Pre-calculate future returns for different horizons
        self._future_returns = {}
        for h in self.forecast_horizons:
            self._future_returns[h] = self._calculate_future_returns(h)
        
        # Create folds
        self.folds = self._create_folds()
    
    def _calculate_future_returns(self, horizon: int) -> np.ndarray:
        """
        Calculate future returns for each bar.
        CRITICAL: This is only used for evaluation, not prediction.
        """
        future_close = np.roll(self.close, -horizon)
        future_close[-horizon:] = np.nan
        returns = (future_close - self.close) / self.close * 100
        return returns
    
    def _create_folds(self) -> List[WalkForwardFold]:
        """Create walk-forward validation folds."""
        folds = []
        
        # Calculate fold sizes
        fold_size = self.length // self.n_folds
        train_size = int(fold_size * self.train_ratio)
        test_size = fold_size - train_size - self.embargo_bars
        
        if test_size < 100:
            # Adjust if test size is too small
            test_size = 100
            train_size = fold_size - test_size - self.embargo_bars
        
        for i in range(self.n_folds):
            start = i * fold_size
            train_start = start
            train_end = start + train_size
            test_start = train_end + self.embargo_bars
            test_end = min(start + fold_size, self.length)
            
            if test_end <= test_start:
                continue
            
            folds.append(WalkForwardFold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_bars=self.embargo_bars
            ))
        
        return folds
    
    def evaluate_indicator(
        self,
        indicator_result: IndicatorResult,
        use_discrete_signals: bool = True
    ) -> BacktestMetrics:
        """
        Evaluate indicator performance across all folds and horizons.
        
        CRITICAL: Each fold selects its optimal horizon using ONLY its training data,
        then evaluates test data with that fold-specific horizon. This prevents lookahead bias.
        
        Args:
            indicator_result: Result from running the indicator
            use_discrete_signals: Use buy/sell signals vs directional
            
        Returns:
            Aggregated metrics across all folds
        """
        # Aggregate metrics across folds - each fold selects its own optimal horizon
        all_trades = []
        fold_metrics = []
        fold_horizons = []
        
        for fold in self.folds:
            # CRITICAL: Select optimal horizon using ONLY this fold's training data
            # This prevents lookahead bias by ensuring test data is never used for horizon selection
            fold_best_horizon = self._find_optimal_horizon(
                indicator_result,
                use_discrete_signals,
                fold=fold  # Pass fold to restrict to training data only
            )
            fold_horizons.append(fold_best_horizon)
            
            # Evaluate test data using the horizon selected from training data
            metrics = self._evaluate_fold(
                indicator_result,
                fold,
                fold_best_horizon,  # Use fold-specific horizon
                use_discrete_signals
            )
            if metrics.total_trades >= self.min_trades_per_fold:
                fold_metrics.append(metrics)
                all_trades.extend(metrics.trades)
        
        if not fold_metrics:
            return BacktestMetrics()
        
        # Aggregate metrics
        total_trades = sum(m.total_trades for m in fold_metrics)
        winning_trades = sum(m.winning_trades for m in fold_metrics)
        losing_trades = sum(m.losing_trades for m in fold_metrics)
        
        if total_trades == 0:
            return BacktestMetrics()
        
        # Calculate returns
        returns = [t.return_pct for t in all_trades]
        total_return = sum(returns)
        avg_return = np.mean(returns) if returns else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)
        
        # Use median horizon across folds for Sharpe ratio calculation
        median_horizon = int(np.median(fold_horizons)) if fold_horizons else 24
        
        # Sharpe ratio (annualized for hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760 / median_horizon)
        else:
            sharpe = 0
        
        # Max drawdown - use compound equity curve, not sum of percentages
        if len(returns) > 0:
            # Convert percentage returns to multipliers and compound them
            equity = 100 * np.cumprod(1 + np.array(returns) / 100)
            running_max = np.maximum.accumulate(equity)
            # Calculate drawdown as percentage of peak
            drawdowns = (equity - running_max) / running_max * 100
            max_drawdown = abs(np.min(drawdowns))
            # Cap at 100% - can't lose more than total equity (no leverage modeled)
            max_drawdown = min(100.0, max_drawdown)
        else:
            max_drawdown = 0
        
        # Directional accuracy (average across folds, clamped to valid range)
        directional_accuracy = np.mean([m.directional_accuracy for m in fold_metrics])
        directional_accuracy = max(0.0, min(1.0, directional_accuracy))  # Safety clamp

        # Extreme move capture rate (average across folds)
        tail_capture_rate = np.mean([m.tail_capture_rate for m in fold_metrics]) if fold_metrics else 0.0

        # Consistency across folds (profit factor + directional accuracy stability)
        consistency_score = self._calculate_consistency_score(fold_metrics)
        
        # Improvement over random (50% baseline)
        improvement_over_random = (directional_accuracy - 0.5) / 0.5 * 100 if directional_accuracy > 0.5 else 0
        
        # Average holding bars
        avg_holding = np.mean([t.holding_bars for t in all_trades]) if all_trades else 0
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return=total_return,
            avg_return=avg_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            avg_holding_bars=avg_holding,
            directional_accuracy=directional_accuracy,
            forecast_horizon=median_horizon,  # Use median of fold-specific horizons
            improvement_over_random=improvement_over_random,
            tail_capture_rate=tail_capture_rate,
            consistency_score=consistency_score,
            trades=all_trades
        )

    def _calculate_consistency_score(self, fold_metrics: List[BacktestMetrics]) -> float:
        """Estimate stability across folds (0-1). Higher = more consistent."""
        if not fold_metrics:
            return 0.0

        def score(values: List[float]) -> float:
            if not values:
                return 0.0
            mean = np.mean(values)
            if mean <= 0:
                return 0.0
            cv = np.std(values) / mean
            return 1 / (1 + cv)

        pf_values = [m.profit_factor for m in fold_metrics if m.total_trades > 0]
        acc_values = [m.directional_accuracy for m in fold_metrics if m.total_trades > 0]

        pf_score = score(pf_values)
        acc_score = score(acc_values)

        return float(np.mean([pf_score, acc_score]))

    def _get_train_range_for_horizon(
        self,
        fold: WalkForwardFold,
        horizon: int
    ) -> Optional[Tuple[int, int]]:
        """Return a safe training range that avoids label leakage into embargo/test."""
        start_idx = fold.train_start
        boundary = min(fold.train_end, fold.test_start)
        if fold.train_end > fold.test_start:
            logger.warning(
                f"Fold training end ({fold.train_end}) exceeds test start ({fold.test_start}). "
                f"Truncating to prevent lookahead bias."
            )
        effective_end = boundary - horizon

        if start_idx < 0 or effective_end <= start_idx:
            logger.warning(
                f"Insufficient training range after horizon trim: "
                f"start={start_idx}, end={effective_end}, horizon={horizon}."
            )
            return None

        return start_idx, effective_end
    
    def _find_optimal_horizon(
        self,
        indicator_result: IndicatorResult,
        use_discrete_signals: bool,
        fold: Optional[WalkForwardFold] = None
    ) -> int:
        """
        Find the forecast horizon with best directional accuracy.
        
        Args:
            indicator_result: Indicator result with signals
            use_discrete_signals: Whether to use buy/sell signals vs directional
            fold: Optional fold to restrict search to training data only.
                  If None, uses all data (for backwards compatibility, but should be avoided)
        
        Returns:
            Optimal forecast horizon in bars
        """
        best_horizon = self.forecast_horizons[0] if self.forecast_horizons else 24  # Default
        best_accuracy = 0.0

        if fold is None:
            # No fold provided - use all data (backwards compatibility)
            # WARNING: This can cause lookahead bias if test data is included
            logger.warning(
                "_find_optimal_horizon called without fold parameter. "
                "This may cause lookahead bias if test data is included."
            )
            start_idx = None
            end_idx = None

        for horizon in self.forecast_horizons:
            if fold is not None:
                train_range = self._get_train_range_for_horizon(fold, horizon)
                if train_range is None:
                    continue
                start_idx, end_idx = train_range
            accuracy = self._calculate_directional_accuracy(
                indicator_result,
                horizon,
                use_discrete_signals,
                start_idx=start_idx,
                end_idx=end_idx
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_horizon = horizon

        return best_horizon
    
    def _calculate_directional_accuracy(
        self,
        indicator_result: IndicatorResult,
        horizon: int,
        use_discrete_signals: bool,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None
    ) -> float:
        """
        Calculate how accurately the indicator predicts future direction.
        CRITICAL: Only uses past indicator values to predict future returns.
        
        Args:
            indicator_result: Indicator result with signals
            horizon: Forecast horizon in bars
            use_discrete_signals: Whether to use buy/sell signals vs directional
            start_idx: Optional start index to restrict calculation (default: 0)
            end_idx: Optional end index to restrict calculation (default: end of data)
        """
        future_returns = self._future_returns[horizon]
        
        # Apply data range restrictions if provided
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(future_returns)
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(future_returns)))
        end_idx = max(start_idx, min(end_idx, len(future_returns)))
        
        # VALIDATION: Ensure we have valid data range
        if end_idx <= start_idx:
            logger.warning(
                f"Invalid data range for directional accuracy: start={start_idx}, end={end_idx}. "
                f"Returning default accuracy."
            )
            return 0.5
        
        # VALIDATION: Log when using restricted range (training data only)
        if start_idx > 0 or end_idx < len(future_returns):
            logger.debug(
                f"Calculating directional accuracy on restricted range: "
                f"bars {start_idx} to {end_idx} (total data: {len(future_returns)})"
            )
        
        # Slice arrays to the specified range
        future_returns_slice = future_returns[start_idx:end_idx]
        
        if use_discrete_signals:
            # Use buy/sell signals
            buy_signals = indicator_result.buy_signals[start_idx:end_idx]
            sell_signals = indicator_result.sell_signals[start_idx:end_idx]
            
            # Buy signal should predict positive returns
            buy_correct = np.sum((buy_signals) & (future_returns_slice > 0) & ~np.isnan(future_returns_slice))
            buy_total = np.sum((buy_signals) & ~np.isnan(future_returns_slice))
            
            # Sell signal should predict negative returns
            sell_correct = np.sum((sell_signals) & (future_returns_slice < 0) & ~np.isnan(future_returns_slice))
            sell_total = np.sum((sell_signals) & ~np.isnan(future_returns_slice))
            
            total_correct = buy_correct + sell_correct
            total = buy_total + sell_total
            
        else:
            # Use directional signal
            signal = indicator_result.combined_signal[start_idx:end_idx]
            
            # Positive signal should predict positive returns
            pos_correct = np.sum((signal > 0) & (future_returns_slice > 0) & ~np.isnan(future_returns_slice))
            pos_total = np.sum((signal > 0) & ~np.isnan(future_returns_slice))
            
            # Negative signal should predict negative returns
            neg_correct = np.sum((signal < 0) & (future_returns_slice < 0) & ~np.isnan(future_returns_slice))
            neg_total = np.sum((signal < 0) & ~np.isnan(future_returns_slice))
            
            total_correct = pos_correct + neg_correct
            total = pos_total + neg_total
        
        return total_correct / total if total > 0 else 0.5

    def _calculate_tail_capture_rate(
        self,
        indicator_result: IndicatorResult,
        fold: WalkForwardFold,
        horizon: int,
        use_discrete_signals: bool
    ) -> float:
        """
        Measure how well the indicator captures significant highs/lows.

        Thresholds are derived from training data only to avoid lookahead bias.
        """
        train_range = self._get_train_range_for_horizon(fold, horizon)
        if train_range is None:
            return 0.0
        train_start, train_end = train_range
        train_returns = self._future_returns[horizon][train_start:train_end]
        train_returns = train_returns[~np.isnan(train_returns)]

        if len(train_returns) < 50:
            return 0.0

        high_threshold = np.percentile(train_returns, self.EXTREME_RETURN_PERCENTILE * 100)
        low_threshold = np.percentile(train_returns, (1 - self.EXTREME_RETURN_PERCENTILE) * 100)

        test_end = fold.test_end
        effective_end = min(test_end, self.length) - horizon
        if effective_end <= fold.test_start:
            return 0.0

        test_returns = self._future_returns[horizon][fold.test_start:effective_end]
        valid_mask = ~np.isnan(test_returns)

        if not np.any(valid_mask):
            return 0.0

        up_moves = (test_returns >= high_threshold) & valid_mask
        down_moves = (test_returns <= low_threshold) & valid_mask

        if use_discrete_signals:
            buy_sigs = indicator_result.buy_signals[fold.test_start:effective_end]
            sell_sigs = indicator_result.sell_signals[fold.test_start:effective_end]
        else:
            train_signal = indicator_result.combined_signal[train_start:train_end]
            train_signal = train_signal[~np.isnan(train_signal)]
            if len(train_signal) < 50:
                return 0.0
            strength_threshold = np.percentile(
                np.abs(train_signal),
                self.SIGNAL_STRENGTH_PERCENTILE * 100
            )
            strength_threshold = max(0.1, strength_threshold)
            test_signal = indicator_result.combined_signal[fold.test_start:effective_end]
            buy_sigs = test_signal >= strength_threshold
            sell_sigs = test_signal <= -strength_threshold

        captured_up = np.sum(buy_sigs & up_moves)
        captured_down = np.sum(sell_sigs & down_moves)
        total_extremes = np.sum(up_moves) + np.sum(down_moves)

        signal_count = np.sum(buy_sigs | sell_sigs)
        captured_total = captured_up + captured_down

        recall = captured_total / total_extremes if total_extremes > 0 else 0.0
        precision = captured_total / signal_count if signal_count > 0 else 0.0

        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _evaluate_fold(
        self,
        indicator_result: IndicatorResult,
        fold: WalkForwardFold,
        horizon: int,
        use_discrete_signals: bool
    ) -> BacktestMetrics:
        """
        Evaluate indicator on a single fold's test set.
        
        CRITICAL: This method ONLY uses test data (test_start to test_end).
        The horizon parameter should have been selected using only training data
        from this fold to prevent lookahead bias.
        """
        test_start = fold.test_start
        test_end = fold.test_end
        effective_end = min(test_end, self.length) - horizon
        
        if effective_end <= test_start:
            return BacktestMetrics()
        
        # VALIDATION: Ensure we're only using test data, not training data
        if test_start < fold.train_end:
            logger.warning(
                f"Test start ({test_start}) is before training end ({fold.train_end}). "
                f"This may indicate data contamination."
            )
        
        trades = []
        future_returns = self._future_returns[horizon]
        
        if use_discrete_signals:
            # Trade on discrete buy/sell signals
            buy_signals = indicator_result.buy_signals[test_start:effective_end]
            sell_signals = indicator_result.sell_signals[test_start:effective_end]
            
            for i in range(len(buy_signals)):
                bar = test_start + i
                exit_bar = bar + horizon
                
                if buy_signals[i]:
                    trades.append(Trade(
                        entry_bar=bar,
                        exit_bar=exit_bar,
                        direction=TradeDirection.LONG,
                        entry_price=self.close[bar],
                        exit_price=self.close[exit_bar]
                    ))
                
                if sell_signals[i]:
                    trades.append(Trade(
                        entry_bar=bar,
                        exit_bar=exit_bar,
                        direction=TradeDirection.SHORT,
                        entry_price=self.close[bar],
                        exit_price=self.close[exit_bar]
                    ))
        else:
            # Trade on directional signal
            signal = indicator_result.combined_signal[test_start:effective_end]
            prev_signal = np.roll(signal, 1)
            prev_signal[0] = 0
            
            # Enter on signal crossing threshold
            threshold = 0.3
            
            for i in range(len(signal)):
                bar = test_start + i
                exit_bar = bar + horizon
                
                # Long entry
                if signal[i] > threshold and prev_signal[i] <= threshold:
                    trades.append(Trade(
                        entry_bar=bar,
                        exit_bar=exit_bar,
                        direction=TradeDirection.LONG,
                        entry_price=self.close[bar],
                        exit_price=self.close[exit_bar]
                    ))
                
                # Short entry
                if signal[i] < -threshold and prev_signal[i] >= -threshold:
                    trades.append(Trade(
                        entry_bar=bar,
                        exit_bar=exit_bar,
                        direction=TradeDirection.SHORT,
                        entry_price=self.close[bar],
                        exit_price=self.close[exit_bar]
                    ))
        
        if not trades:
            return BacktestMetrics()
        
        # Calculate metrics
        returns = [t.return_pct for t in trades]
        winning = [t for t in trades if t.return_pct > 0]
        losing = [t for t in trades if t.return_pct <= 0]
        
        total_return = sum(returns)
        avg_return = np.mean(returns)
        win_rate = len(winning) / len(trades)
        
        gross_profit = sum(t.return_pct for t in winning)
        gross_loss = abs(sum(t.return_pct for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)

        # Sharpe ratio (annualized for hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(8760 / horizon)
        else:
            sharpe_ratio = 0.0

        # Max drawdown - use compound equity curve
        equity = 100 * np.cumprod(1 + np.array(returns) / 100)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        max_drawdown = min(100.0, max_drawdown)
        
        # Directional accuracy for this fold
        test_returns = future_returns[test_start:effective_end]
        valid_mask = ~np.isnan(test_returns)  # Filter out NaN future returns
        
        if use_discrete_signals:
            buy_sigs = indicator_result.buy_signals[test_start:effective_end]
            sell_sigs = indicator_result.sell_signals[test_start:effective_end]
            
            # Only count signals where we have valid future returns
            correct = np.sum((buy_sigs & (test_returns > 0) & valid_mask) | 
                           (sell_sigs & (test_returns < 0) & valid_mask))
            total_sigs = np.sum((buy_sigs | sell_sigs) & valid_mask)
            directional_accuracy = correct / total_sigs if total_sigs > 0 else 0.5
        else:
            sig = indicator_result.combined_signal[test_start:effective_end]
            # Use consistent threshold (0.1) for both correct and total_sigs
            threshold = 0.1
            correct = np.sum(((sig > threshold) & (test_returns > 0) & valid_mask) | 
                           ((sig < -threshold) & (test_returns < 0) & valid_mask))
            total_sigs = np.sum(((sig > threshold) | (sig < -threshold)) & valid_mask)
            directional_accuracy = correct / total_sigs if total_sigs > 0 else 0.5
        
        # Safety clamp: accuracy should be between 0 and 1
        directional_accuracy = max(0.0, min(1.0, directional_accuracy))

        tail_capture_rate = self._calculate_tail_capture_rate(
            indicator_result,
            fold,
            horizon,
            use_discrete_signals
        )
        
        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_return=total_return,
            avg_return=avg_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            directional_accuracy=directional_accuracy,
            forecast_horizon=horizon,
            tail_capture_rate=tail_capture_rate,
            trades=trades
        )
    
    def calculate_objective(self, metrics: BacktestMetrics) -> float:
        """
        Calculate optimization objective from metrics.
        Higher is better.
        """
        return calculate_objective_score(
            metrics,
            min_trades=self.min_trades_per_fold,
            min_trades_penalty=50
        )


def create_backtester(df: pd.DataFrame, **kwargs) -> WalkForwardBacktester:
    """Factory function to create a backtester instance."""
    return WalkForwardBacktester(df, **kwargs)


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    n = 5000
    
    # Generate trending price data
    trend = np.cumsum(np.random.randn(n) * 0.1)
    noise = np.random.randn(n) * 0.5
    close = 100 + trend + noise
    
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='h'),
        'open': close + np.random.randn(n) * 0.2,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Create mock indicator result
    signal = np.random.randn(n) * 0.5  # Random signal
    buy_signals = signal > 1.0
    sell_signals = signal < -1.0
    
    indicator_result = IndicatorResult(
        main_values=signal * 50,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        combined_signal=np.clip(signal, -1, 1)
    )
    
    # Run backtest
    backtester = WalkForwardBacktester(df)
    metrics = backtester.evaluate_indicator(indicator_result)
    
    print("\nBacktest Results:")
    print(f"  Total trades: {metrics.total_trades}")
    print(f"  Win rate: {metrics.win_rate:.1%}")
    print(f"  Profit factor: {metrics.profit_factor:.2f}")
    print(f"  Directional accuracy: {metrics.directional_accuracy:.1%}")
    print(f"  Optimal forecast horizon: {metrics.forecast_horizon}h")
    print(f"  Improvement over random: {metrics.improvement_over_random:.1f}%")
