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
            'improvement_over_random': self.improvement_over_random
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
    """
    
    # Forecast horizons to test (in bars/hours for 1H data)
    # Broad continuum from 1 hour to 1 week (168 hours)
    FORECAST_HORIZONS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 168]
    
    def __init__(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        embargo_bars: int = 72,
        train_ratio: float = 0.6,
        min_trades_per_fold: int = 10
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
        
        self.close = df['close'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.length = len(df)
        
        # Pre-calculate future returns for different horizons
        self._future_returns = {}
        for h in self.FORECAST_HORIZONS:
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
        
        Args:
            indicator_result: Result from running the indicator
            use_discrete_signals: Use buy/sell signals vs directional
            
        Returns:
            Aggregated metrics across all folds
        """
        # Find optimal forecast horizon
        best_horizon = self._find_optimal_horizon(indicator_result, use_discrete_signals)
        
        # Aggregate metrics across folds
        all_trades = []
        fold_metrics = []
        
        for fold in self.folds:
            metrics = self._evaluate_fold(
                indicator_result,
                fold,
                best_horizon,
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
        
        # Sharpe ratio (annualized for hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760 / best_horizon)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Directional accuracy
        directional_accuracy = np.mean([m.directional_accuracy for m in fold_metrics])
        
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
            forecast_horizon=best_horizon,
            improvement_over_random=improvement_over_random,
            trades=all_trades
        )
    
    def _find_optimal_horizon(
        self,
        indicator_result: IndicatorResult,
        use_discrete_signals: bool
    ) -> int:
        """Find the forecast horizon with best directional accuracy."""
        best_horizon = 24  # Default
        best_accuracy = 0.0
        
        for horizon in self.FORECAST_HORIZONS:
            accuracy = self._calculate_directional_accuracy(
                indicator_result,
                horizon,
                use_discrete_signals
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_horizon = horizon
        
        return best_horizon
    
    def _calculate_directional_accuracy(
        self,
        indicator_result: IndicatorResult,
        horizon: int,
        use_discrete_signals: bool
    ) -> float:
        """
        Calculate how accurately the indicator predicts future direction.
        CRITICAL: Only uses past indicator values to predict future returns.
        """
        future_returns = self._future_returns[horizon]
        
        if use_discrete_signals:
            # Use buy/sell signals
            buy_signals = indicator_result.buy_signals
            sell_signals = indicator_result.sell_signals
            
            # Buy signal should predict positive returns
            buy_correct = np.sum((buy_signals) & (future_returns > 0) & ~np.isnan(future_returns))
            buy_total = np.sum((buy_signals) & ~np.isnan(future_returns))
            
            # Sell signal should predict negative returns
            sell_correct = np.sum((sell_signals) & (future_returns < 0) & ~np.isnan(future_returns))
            sell_total = np.sum((sell_signals) & ~np.isnan(future_returns))
            
            total_correct = buy_correct + sell_correct
            total = buy_total + sell_total
            
        else:
            # Use directional signal
            signal = indicator_result.combined_signal
            
            # Positive signal should predict positive returns
            pos_correct = np.sum((signal > 0) & (future_returns > 0) & ~np.isnan(future_returns))
            pos_total = np.sum((signal > 0) & ~np.isnan(future_returns))
            
            # Negative signal should predict negative returns
            neg_correct = np.sum((signal < 0) & (future_returns < 0) & ~np.isnan(future_returns))
            neg_total = np.sum((signal < 0) & ~np.isnan(future_returns))
            
            total_correct = pos_correct + neg_correct
            total = pos_total + neg_total
        
        return total_correct / total if total > 0 else 0.5
    
    def _evaluate_fold(
        self,
        indicator_result: IndicatorResult,
        fold: WalkForwardFold,
        horizon: int,
        use_discrete_signals: bool
    ) -> BacktestMetrics:
        """Evaluate indicator on a single fold's test set."""
        test_start = fold.test_start
        test_end = fold.test_end
        
        trades = []
        future_returns = self._future_returns[horizon]
        
        if use_discrete_signals:
            # Trade on discrete buy/sell signals
            buy_signals = indicator_result.buy_signals[test_start:test_end]
            sell_signals = indicator_result.sell_signals[test_start:test_end]
            
            for i in range(len(buy_signals)):
                bar = test_start + i
                exit_bar = min(bar + horizon, self.length - 1)
                
                if exit_bar >= self.length:
                    continue
                
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
            signal = indicator_result.combined_signal[test_start:test_end]
            prev_signal = np.roll(signal, 1)
            prev_signal[0] = 0
            
            # Enter on signal crossing threshold
            threshold = 0.3
            
            for i in range(len(signal)):
                bar = test_start + i
                exit_bar = min(bar + horizon, self.length - 1)
                
                if exit_bar >= self.length:
                    continue
                
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
        
        # Directional accuracy for this fold
        test_returns = future_returns[test_start:test_end]
        if use_discrete_signals:
            buy_sigs = indicator_result.buy_signals[test_start:test_end]
            sell_sigs = indicator_result.sell_signals[test_start:test_end]
            
            correct = np.sum((buy_sigs & (test_returns > 0)) | (sell_sigs & (test_returns < 0)))
            total_sigs = np.sum(buy_sigs | sell_sigs)
            directional_accuracy = correct / total_sigs if total_sigs > 0 else 0.5
        else:
            sig = indicator_result.combined_signal[test_start:test_end]
            correct = np.sum(((sig > 0) & (test_returns > 0)) | ((sig < 0) & (test_returns < 0)))
            total_sigs = np.sum((sig > 0.1) | (sig < -0.1))
            directional_accuracy = correct / total_sigs if total_sigs > 0 else 0.5
        
        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_return=total_return,
            avg_return=avg_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            directional_accuracy=directional_accuracy,
            forecast_horizon=horizon,
            trades=trades
        )
    
    def calculate_objective(self, metrics: BacktestMetrics) -> float:
        """
        Calculate optimization objective from metrics.
        Higher is better.
        """
        if metrics.total_trades < self.min_trades_per_fold:
            return 0.0
        
        # Weighted combination of metrics
        # Profit factor: capped at 5 for stability
        pf_score = min(metrics.profit_factor, 5.0) / 5.0
        
        # Directional accuracy: above 50% is good
        acc_score = (metrics.directional_accuracy - 0.5) * 2  # Scale to 0-1 for 50-100%
        acc_score = max(0, min(1, acc_score))
        
        # Sharpe ratio: capped at 3 for stability
        sharpe_score = min(max(metrics.sharpe_ratio, 0), 3.0) / 3.0
        
        # Win rate bonus
        win_score = metrics.win_rate
        
        # Combine with weights
        objective = (
            0.35 * pf_score +
            0.30 * acc_score +
            0.20 * sharpe_score +
            0.15 * win_score
        )
        
        # Penalty for too few trades
        if metrics.total_trades < 50:
            objective *= metrics.total_trades / 50
        
        return objective


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
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='1H'),
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

