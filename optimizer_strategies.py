"""
Optimization strategies for the optimizer.
"""

import logging
from typing import Dict, Any
import pandas as pd
from pine_parser import ParseResult
from optimizer_types import OptimizationResult
from optimizer_core import PineOptimizer

logger = logging.getLogger(__name__)

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
