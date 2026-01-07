import time
from pathlib import Path
import shutil

from data_manager import DataManager
from pine_parser import parse_pine_script
from optimizer import optimize_indicator
from output_generator import generate_outputs
from objective import calculate_objective_score as objective_score


INDICATOR_DIR = Path("pinescripts")
INDICATORS = [
    "ADX.pine",
    "Aroon.pine",
    "BollingerBands.pine",
    "RSI.pine",
]
INTERVAL = "1h"
SYMBOLS = ["BTCUSDT", "SOLUSDT", "PAXGUSDT"]
METHOD_BUDGET_SECONDS = 120


def calculate_objective_score(metrics) -> float:
    return objective_score(metrics)


def load_data(dm: DataManager):
    data = {}
    for symbol in SYMBOLS:
        if not dm.symbol_exists(symbol, INTERVAL):
            raise RuntimeError(f"Missing data for {symbol} @ {INTERVAL}")
        data[symbol] = dm.load_symbol(symbol, INTERVAL)
    return data


def copy_outputs(outputs, indicator_name, method_name, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    report_src = Path(outputs["report"])
    pine_src = Path(outputs["pine_script"])

    report_dst = out_dir / f"{indicator_name}_{method_name}_report.txt"
    pine_dst = out_dir / f"{indicator_name}_{method_name}.pine"

    shutil.copyfile(report_src, report_dst)
    shutil.copyfile(pine_src, pine_dst)

    return report_dst, pine_dst


def run_single_method(parse_result, pine_path, data, method_name, method_cfg):
    start = time.time()
    result = optimize_indicator(
        parse_result,
        data,
        interval=INTERVAL,
        timeout_seconds=method_cfg.get("timeout_seconds", METHOD_BUDGET_SECONDS),
        sampler_name=method_cfg.get("sampler_name", "tpe"),
        early_stop_patience=method_cfg.get("early_stop_patience"),
        backtester_overrides=method_cfg.get("backtester_overrides"),
        seed_params=method_cfg.get("seed_params")
    )
    elapsed = time.time() - start
    outputs = generate_outputs(parse_result, result, str(pine_path))
    return result, outputs, elapsed


def run_multi_fidelity(parse_result, pine_path, data, method_cfg):
    stage_budget = METHOD_BUDGET_SECONDS // 2
    symbols = list(data.keys())
    subset_symbol = symbols[0]
    subset_data = {subset_symbol: data[subset_symbol]}

    cheap_overrides = {
        "n_folds": 2,
        "embargo_bars": 5,
        "min_trades_per_fold": 2,
        "forecast_horizons": [1, 2, 3, 5, 8, 13],
    }

    stage1_cfg = {
        "timeout_seconds": stage_budget,
        "sampler_name": "tpe",
        "early_stop_patience": method_cfg.get("early_stop_patience"),
        "backtester_overrides": cheap_overrides,
    }

    stage1_result, _, stage1_elapsed = run_single_method(
        parse_result, pine_path, subset_data, "multi_fidelity_stage1", stage1_cfg
    )

    stage2_cfg = {
        "timeout_seconds": METHOD_BUDGET_SECONDS - stage_budget,
        "sampler_name": "tpe",
        "early_stop_patience": method_cfg.get("early_stop_patience"),
        "seed_params": stage1_result.best_params,
    }

    stage2_result, outputs, stage2_elapsed = run_single_method(
        parse_result, pine_path, data, "multi_fidelity_stage2", stage2_cfg
    )

    return stage2_result, outputs, stage1_elapsed + stage2_elapsed


def main():
    if not INDICATOR_DIR.exists():
        raise RuntimeError(f"Indicator directory not found: {INDICATOR_DIR}")

    dm = DataManager()
    data = load_data(dm)

    methods = [
        {
            "name": "tpe_baseline",
            "cfg": {}
        },
        {
            "name": "tpe_early_stop",
            "cfg": {"early_stop_patience": 40}
        },
        {
            "name": "random_search",
            "cfg": {"sampler_name": "random"}
        },
        {
            "name": "multi_fidelity",
            "cfg": {"early_stop_patience": 30}
        },
        {
            "name": "reduced_folds_horizons",
            "cfg": {
                "backtester_overrides": {
                    "n_folds": 2,
                    "embargo_bars": 12,
                    "min_trades_per_fold": 2,
                    "forecast_horizons": [1, 2, 3, 5, 8, 13, 21],
                }
            }
        },
    ]

    summary = []
    reports_root = Path("benchmark_reports")
    reports_root.mkdir(exist_ok=True)

    for method in methods:
        method_name = method["name"]
        out_dir = reports_root / method_name
        print(f"\n=== Running method: {method_name} ===")

        for indicator_file in INDICATORS:
            pine_path = INDICATOR_DIR / indicator_file
            if not pine_path.exists():
                raise RuntimeError(f"Indicator not found: {pine_path}")

            parse_result = parse_pine_script(str(pine_path))

            if method_name == "multi_fidelity":
                result, outputs, elapsed = run_multi_fidelity(parse_result, pine_path, data, method["cfg"])
            else:
                result, outputs, elapsed = run_single_method(parse_result, pine_path, data, method_name, method["cfg"])

            indicator_name = pine_path.stem
            report_path, pine_out = copy_outputs(outputs, indicator_name, method_name, out_dir)

            objective = calculate_objective_score(result.best_metrics)
            summary.append({
                "method": method_name,
                "indicator": indicator_name,
                "elapsed": elapsed,
                "objective": objective,
                "profit_factor": result.best_metrics.profit_factor,
                "win_rate": result.best_metrics.win_rate,
                "directional_accuracy": result.best_metrics.directional_accuracy,
                "sharpe_ratio": result.best_metrics.sharpe_ratio,
                "report": str(report_path),
                "pine": str(pine_out),
            })

            print(f"  {indicator_name}: obj={objective:.4f} time={elapsed:.1f}s report={report_path}")

    # Print simple aggregate summary
    print("\n=== Aggregate Summary ===")
    by_method = {}
    for row in summary:
        method = row["method"]
        if method not in by_method:
            by_method[method] = {"objective": [], "elapsed": []}
        by_method[method]["objective"].append(row["objective"])
        by_method[method]["elapsed"].append(row["elapsed"])

    for method, vals in by_method.items():
        avg_obj = sum(vals["objective"]) / len(vals["objective"])
        avg_time = sum(vals["elapsed"]) / len(vals["elapsed"])
        print(f"{method}: avg_objective={avg_obj:.4f} avg_time={avg_time:.1f}s")


if __name__ == "__main__":
    main()
