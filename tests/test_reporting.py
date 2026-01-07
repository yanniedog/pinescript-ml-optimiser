import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from backtester import BacktestMetrics
from interactive_optimizer import _serialize_metrics, write_unified_report
from objective import calculate_objective_score


class ReportingVerificationTests(unittest.TestCase):
    def test_objective_score_returns_zero_for_insufficient_trades(self):
        metrics = SimpleNamespace(
            total_trades=4,
            profit_factor=2.5,
            directional_accuracy=0.7,
            sharpe_ratio=0.6,
            win_rate=0.55,
            tail_capture_rate=0.4,
            consistency_score=0.3,
            max_drawdown=15.0,
        )
        score = calculate_objective_score(metrics, min_trades=10, min_trades_penalty=50)
        self.assertEqual(0.0, score)

    def test_objective_score_applies_penalty_between_thresholds(self):
        metrics = SimpleNamespace(
            total_trades=20,
            profit_factor=2.0,
            directional_accuracy=0.7,
            sharpe_ratio=0.8,
            win_rate=0.6,
            tail_capture_rate=0.4,
            consistency_score=0.5,
            max_drawdown=10.0,
        )
        base_score = (
            0.25 * (min(metrics.profit_factor, 5.0) / 5.0) +
            0.20 * max(0, min(1, (metrics.directional_accuracy - 0.5) * 2)) +
            0.15 * (min(max(metrics.sharpe_ratio, 0), 3.0) / 3.0) +
            0.10 * metrics.win_rate +
            0.15 * metrics.tail_capture_rate +
            0.10 * metrics.consistency_score +
            0.05 * (1 - min(max(metrics.max_drawdown, 0.0), 100.0) / 100.0)
        )
        expected_score = base_score * (metrics.total_trades / 50.0)
        actual_score = calculate_objective_score(metrics, min_trades=10, min_trades_penalty=50)
        self.assertTrue(math.isclose(expected_score, actual_score, rel_tol=1e-9))

    def test_serialize_metrics_returns_expected_dictionary(self):
        metrics = BacktestMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_return=5.0,
            avg_return=0.5,
            win_rate=0.6,
            profit_factor=1.8,
            sharpe_ratio=0.9,
            max_drawdown=12.0,
            avg_holding_bars=5.0,
            directional_accuracy=0.65,
            forecast_horizon=24,
            improvement_over_random=30.0,
            tail_capture_rate=0.4,
            consistency_score=0.7,
        )
        serialized = _serialize_metrics(metrics)
        expected = {
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "total_return": 5.0,
            "avg_return": 0.5,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "sharpe_ratio": 0.9,
            "max_drawdown": 12.0,
            "avg_holding_bars": 5.0,
            "directional_accuracy": 0.65,
            "forecast_horizon": 24,
            "improvement_over_random": 30.0,
            "tail_capture_rate": 0.4,
            "consistency_score": 0.7,
        }
        self.assertEqual(expected, serialized)

    def test_write_unified_report_outputs_reflect_input_metrics(self):
        run_info = {
            "indicator_directory": "pinescripts",
            "interval": "1d",
            "symbols": "BTCUSDT",
            "timeout_seconds": 60,
            "generated_all": False,
            "total_indicators": 2,
        }
        result_high = {
            "indicator_name": "IndicatorHigh",
            "file_name": "high.pine",
            "objective_best": 1.25,
            "baseline_objective": 0.5,
            "optimization_time": 12.3,
            "n_trials": 5,
            "best_metrics": {
                "profit_factor": 1.2,
                "win_rate": 0.65,
                "sharpe_ratio": 1.1,
                "max_drawdown": 10.0,
            },
            "output_pine": "optimized_high.pine",
            "output_report": "report_high.txt",
            "config": {
                "strategy": "tpe",
                "sampler": "tpe",
                "timeout_seconds": 120,
                "max_trials": 5,
                "early_stop_patience": 2,
                "min_runtime_seconds": 10,
                "stall_seconds": 5,
                "improvement_rate_floor": 0.01,
                "improvement_rate_window": 10,
            },
        }
        result_low = {
            "indicator_name": "IndicatorLow",
            "file_name": "low.pine",
            "objective_best": 0.75,
            "baseline_objective": 0.3,
            "optimization_time": 17.7,
            "n_trials": 3,
            "best_metrics": {
                "profit_factor": 0.8,
                "win_rate": 0.4,
                "sharpe_ratio": 0.5,
                "max_drawdown": 20.0,
            },
            "output_pine": "optimized_low.pine",
            "output_report": "report_low.txt",
            "config": {
                "strategy": "tpe",
                "sampler": "tpe",
                "timeout_seconds": 120,
                "max_trials": 3,
                "early_stop_patience": 1,
                "min_runtime_seconds": 5,
                "stall_seconds": 3,
                "improvement_rate_floor": 0.02,
                "improvement_rate_window": 5,
            },
        }
        with TemporaryDirectory() as tempdir:
            summary_path = Path(tempdir, "reports", "summary.txt")
            json_path = Path(tempdir, "reports", "summary.json")
            write_unified_report(summary_path, json_path, run_info, [result_high, result_low])

            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertIn("Indicators: 2", summary_text)
            self.assertIn("Avg objective: 1.0000", summary_text)
            self.assertIn("Max objective: 1.2500", summary_text)
            self.assertIn("Min objective: 0.7500", summary_text)
            self.assertIn("Avg optimization time: 15.0s", summary_text)
            self.assertIn("Total optimization time: 30.0s", summary_text)
            self.assertIn("IndicatorHigh: objective=1.2500 pf=1.20", summary_text)
            self.assertIn("IndicatorLow: objective=0.7500 pf=0.80", summary_text)
            self.assertIn("IndicatorHigh (high.pine)", summary_text)
            self.assertIn("IndicatorLow (low.pine)", summary_text)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run"], run_info)
            self.assertEqual(payload["indicators"], [result_high, result_low])
