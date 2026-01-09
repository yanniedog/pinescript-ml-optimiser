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
import socket
import webbrowser
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

_METRIC_DEFS = {
    "objective_best": {"label": "Objective (avg)"},
    "objective_delta": {"label": "Objective Delta"},
    "objective_overall": {"label": "Objective (overall)"},
    "profit_factor": {"label": "Profit Factor"},
    "win_rate": {"label": "Win Rate"},
    "directional_accuracy": {"label": "Directional Accuracy"},
    "sharpe_ratio": {"label": "Sharpe Ratio"},
    "max_drawdown": {"label": "Max Drawdown"},
    "total_return": {"label": "Total Return"},
    "avg_return": {"label": "Avg Return"},
    "total_trades": {"label": "Total Trades"},
    "winning_trades": {"label": "Winning Trades"},
    "losing_trades": {"label": "Losing Trades"},
    "avg_holding_bars": {"label": "Avg Holding Bars"},
    "forecast_horizon": {"label": "Forecast Horizon"},
    "improvement_over_random": {"label": "Improvement vs Random"},
    "tail_capture_rate": {"label": "Tail Capture Rate"},
    "consistency_score": {"label": "Consistency Score"},
}
_METRIC_KEYS = list(_METRIC_DEFS.keys())


def _metrics_from_backtest(metrics: Optional[BacktestMetrics]) -> Dict[str, float]:
    if metrics is None:
        return {}
    return {
        "total_trades": metrics.total_trades,
        "winning_trades": metrics.winning_trades,
        "losing_trades": metrics.losing_trades,
        "total_return": metrics.total_return,
        "avg_return": metrics.avg_return,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown,
        "avg_holding_bars": metrics.avg_holding_bars,
        "directional_accuracy": metrics.directional_accuracy,
        "forecast_horizon": metrics.forecast_horizon,
        "improvement_over_random": metrics.improvement_over_random,
        "tail_capture_rate": metrics.tail_capture_rate,
        "consistency_score": metrics.consistency_score,
    }


def _metric_label(metric_key: str) -> str:
    return _METRIC_DEFS.get(metric_key, {}).get("label", metric_key)


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 0.0001:
            return f"{value:.2e}"
        return f"{value:.4g}"
    return str(value)


def _format_params(params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return "N/A"
    parts = []
    for name in sorted(params.keys()):
        parts.append(f"{name}={_format_param_value(params[name])}")
    return ", ".join(parts)


def _last_non_none(values: List[Optional[float]]) -> Optional[float]:
    for val in reversed(values):
        if val is not None:
            return val
    return None


def _compute_rate_series(
    elapsed_vals: List[float],
    metric_vals: List[Optional[float]],
    baseline_value: Optional[float]
) -> List[Optional[float]]:
    baseline = baseline_value if baseline_value is not None else 0.0
    rates = []
    for elapsed, val in zip(elapsed_vals, metric_vals):
        if val is None or elapsed is None:
            rates.append(None)
            continue
        if elapsed <= 0:
            rates.append(0.0)
            continue
        rates.append((val - baseline) / elapsed)
    return rates


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
        self._ui_axes = {}
        self._ui_widgets = {}
        self._lines = {}
        self._series = {}
        self._indicator_colors = {}
        self._baseline_values = {}
        self._colors = []
        self._color_index = 0
        self._start_times = {}
        self._best_objectives = {}
        self._last_draw_time = 0.0
        self._min_draw_interval = 0.25
        self._last_autoscale_time = 0.0
        self._autoscale_interval = 1.0
        self._max_points = 300
        self._legend = None
        self._legend_enabled = True
        self._legend_max_lines = 25
        self._legend_map = {}
        self._last_legend_update = 0.0
        self._legend_update_interval = 2.0
        self._line_meta = {}
        self._y_metric = "objective_delta"
        self._x_mode = "elapsed"
        self._band_metric = None
        self._band_min = None
        self._band_max = None
        self._filters = {
            "indicator": set(),
            "symbol": set(),
            "timeframe": set()
        }
        self._status_text = None
        self._indicator_input = None
        self._symbol_input = None
        self._timeframe_input = None
        self._toggle_input = None

    def _ensure_ready(self) -> bool:
        if self._init_attempted:
            return self._enabled
        self._init_attempted = True
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.widgets import RadioButtons, TextBox, Button
        except Exception as exc:
            logger.info("Realtime plot disabled (matplotlib not available): %s", exc)
            self._enabled = False
            return False

        try:
            self._plt = plt
            self._colors = list(cm.tab20.colors)
            self._plt.ion()
            self._fig, self._ax = self._plt.subplots()
            self._fig.subplots_adjust(right=0.78, bottom=0.18)
            self._ax.set_title("Optimization Progress")
            self._ax.set_xlabel("Elapsed time (s)")
            self._ax.set_ylabel(_metric_label(self._y_metric))
            self._ax.grid(True, alpha=0.3)
            self._status_text = self._ax.text(
                0.01,
                0.99,
                "Filters: all | Y: objective_delta | X: elapsed | UI: indicator/symbol/timeframe | Click line to toggle",
                transform=self._ax.transAxes,
                va="top",
                ha="left",
                fontsize=8
            )
            self._init_ui(RadioButtons, TextBox, Button)
            self._fig.tight_layout()
            self._fig.show()
            self._fig.canvas.draw()
            self._fig.canvas.mpl_connect("pick_event", self._on_pick)
            self._fig.canvas.mpl_connect("key_press_event", self._on_key_press)
            self._enabled = True
            logger.info(
                "Realtime plot controls: UI panel or keys A(all) I(indicator) "
                "S(symbol) T(timeframe) L(legend) Y(y-axis) X(x-axis) B(band)."
            )
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

    def _parse_label(self, label: str):
        indicator = label
        symbol = ""
        timeframe = ""
        if ":" in label:
            indicator, remainder = label.split(":", 1)
            if "@" in remainder:
                symbol, timeframe = remainder.split("@", 1)
            else:
                symbol = remainder
        return indicator.strip(), symbol.strip(), timeframe.strip()

    def _register_line(self, label: str) -> None:
        indicator, symbol, timeframe = self._parse_label(label)
        self._line_meta[label] = {
            "indicator": indicator.lower(),
            "symbol": symbol.lower(),
            "timeframe": timeframe.lower()
        }

    def _update_status(self, text: str) -> None:
        if self._status_text is None:
            return
        self._status_text.set_text(text)

    def _update_axis_labels(self) -> None:
        if self._ax is None:
            return
        if self._x_mode == "rate":
            x_label = f"Improvement rate ({_metric_label(self._y_metric)} / s)"
        else:
            x_label = "Elapsed time (s)"
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(_metric_label(self._y_metric))

    def _get_series_xy(self, label: str) -> Tuple[List[float], List[Optional[float]]]:
        series = self._series.get(label)
        if not series:
            return [], []
        x_vals = series.get("x", [])
        y_vals = series.get("metrics", {}).get(self._y_metric, [])
        if self._x_mode == "rate":
            baseline = self._baseline_values.get(label, {}).get(self._y_metric)
            x_vals = _compute_rate_series(x_vals, y_vals, baseline)
        return x_vals, y_vals

    def _refresh_lines(self) -> None:
        for label, line in self._lines.items():
            x_vals, y_vals = self._get_series_xy(label)
            line.set_data(x_vals, y_vals)
        self._update_axis_labels()
        self._redraw(force=True)

    def _set_band_filter(self, metric: Optional[str], band_min: Optional[float], band_max: Optional[float]) -> None:
        self._band_metric = metric
        self._band_min = band_min
        self._band_max = band_max

    def _init_ui(self, RadioButtons, TextBox, Button) -> None:
        self._ui_axes["indicator_input"] = self._fig.add_axes([0.80, 0.74, 0.18, 0.05])
        self._indicator_input = TextBox(self._ui_axes["indicator_input"], "Indicators", initial="")

        self._ui_axes["symbol_input"] = self._fig.add_axes([0.80, 0.67, 0.18, 0.05])
        self._symbol_input = TextBox(self._ui_axes["symbol_input"], "Symbols", initial="")

        self._ui_axes["timeframe_input"] = self._fig.add_axes([0.80, 0.60, 0.18, 0.05])
        self._timeframe_input = TextBox(self._ui_axes["timeframe_input"], "Timeframes", initial="")

        self._ui_axes["apply"] = self._fig.add_axes([0.80, 0.53, 0.18, 0.05])
        self._ui_widgets["apply"] = Button(self._ui_axes["apply"], "Apply Filters")
        self._ui_widgets["apply"].on_clicked(self._on_apply_filters)

        self._ui_axes["clear"] = self._fig.add_axes([0.80, 0.47, 0.18, 0.05])
        self._ui_widgets["clear"] = Button(self._ui_axes["clear"], "Clear Filters")
        self._ui_widgets["clear"].on_clicked(self._on_clear_filters)

        self._ui_axes["legend"] = self._fig.add_axes([0.80, 0.41, 0.18, 0.05])
        self._ui_widgets["legend"] = Button(self._ui_axes["legend"], "Toggle Legend")
        self._ui_widgets["legend"].on_clicked(self._on_toggle_legend)

        self._ui_axes["toggle_input"] = self._fig.add_axes([0.80, 0.33, 0.18, 0.05])
        self._toggle_input = TextBox(self._ui_axes["toggle_input"], "Toggle", initial="")

        self._ui_axes["toggle_btn"] = self._fig.add_axes([0.80, 0.27, 0.18, 0.05])
        self._ui_widgets["toggle_btn"] = Button(self._ui_axes["toggle_btn"], "Toggle Lines")
        self._ui_widgets["toggle_btn"].on_clicked(self._on_toggle_lines)

    def _on_apply_filters(self, _event) -> None:
        indicator_text = self._indicator_input.text if self._indicator_input else ""
        symbol_text = self._symbol_input.text if self._symbol_input else ""
        timeframe_text = self._timeframe_input.text if self._timeframe_input else ""

        self._filters["indicator"] = {
            t.strip().lower()
            for t in indicator_text.split(",")
            if t.strip()
        }
        self._filters["symbol"] = {
            t.strip().upper()
            for t in symbol_text.split(",")
            if t.strip()
        }
        self._filters["timeframe"] = {
            t.strip().lower()
            for t in timeframe_text.split(",")
            if t.strip()
        }
        self._apply_filters()

    def _on_clear_filters(self, _event) -> None:
        if self._indicator_input:
            self._indicator_input.set_val("")
        if self._symbol_input:
            self._symbol_input.set_val("")
        if self._timeframe_input:
            self._timeframe_input.set_val("")
        self._filters = {"indicator": set(), "symbol": set(), "timeframe": set()}
        self._apply_filters()

    def _on_toggle_legend(self, _event) -> None:
        self._legend_enabled = not self._legend_enabled
        self._refresh_legend(force=True)
        self._redraw(force=True)

    def _on_toggle_lines(self, _event) -> None:
        query = self._toggle_input.text if self._toggle_input else ""
        self._toggle_lines_by_query(query)

    def _refresh_legend(self, force: bool = False) -> None:
        if not self._legend_enabled or len(self._lines) > self._legend_max_lines:
            if self._legend is not None:
                self._legend.remove()
                self._legend = None
                self._legend_map = {}
            return
        now = time.time()
        if not force and (now - self._last_legend_update) < self._legend_update_interval:
            return
        self._legend = self._ax.legend(loc="best", fontsize=8)
        self._legend_map = {}
        for legline, origline in zip(self._legend.legendHandles, self._lines.values()):
            legline.set_picker(5)
            legline.set_alpha(1.0 if origline.get_visible() else 0.2)
            self._legend_map[legline] = origline
        self._last_legend_update = now

    def _line_matches_filter(self, label: str) -> bool:
        meta = self._line_meta.get(label, {})
        indicator_match = True
        symbol_match = True
        timeframe_match = True

        if self._filters["indicator"]:
            indicator_val = meta.get("indicator", "")
            indicator_match = any(token in indicator_val for token in self._filters["indicator"])
        if self._filters["symbol"]:
            symbol_val = meta.get("symbol", "").upper()
            symbol_match = symbol_val in self._filters["symbol"]
        if self._filters["timeframe"]:
            timeframe_val = meta.get("timeframe", "")
            timeframe_match = timeframe_val in self._filters["timeframe"]

        if not (indicator_match and symbol_match and timeframe_match):
            return False

        if self._band_metric:
            series = self._series.get(label, {})
            values = series.get("metrics", {}).get(self._band_metric, [])
            latest = _last_non_none(values)
            if latest is None:
                return False
            if self._band_min is not None and latest < self._band_min:
                return False
            if self._band_max is not None and latest > self._band_max:
                return False

        return True

    def _apply_filters(self) -> None:
        for label, line in self._lines.items():
            line.set_visible(self._line_matches_filter(label))
        summary_parts = []
        if self._filters["indicator"]:
            summary_parts.append(f"indicator={','.join(sorted(self._filters['indicator']))}")
        if self._filters["symbol"]:
            summary_parts.append(f"symbol={','.join(sorted(self._filters['symbol']))}")
        if self._filters["timeframe"]:
            summary_parts.append(f"timeframe={','.join(sorted(self._filters['timeframe']))}")
        if self._band_metric:
            band_min = "" if self._band_min is None else f"{self._band_min:g}"
            band_max = "" if self._band_max is None else f"{self._band_max:g}"
            summary_parts.append(f"band={self._band_metric}[{band_min},{band_max}]")
        summary = "Filters: " + (", ".join(summary_parts) if summary_parts else "all")
        summary += f" | Y: {self._y_metric} | X: {self._x_mode} | UI: indicator/symbol/timeframe | Click line to toggle"
        self._update_status(summary)
        self._refresh_legend(force=True)
        self._redraw(force=True)

    def _redraw(self, force: bool = False) -> None:
        if not self._enabled:
            return
        now = time.time()
        if not force and (now - self._last_draw_time) < self._min_draw_interval:
            return
        if now - self._last_autoscale_time >= self._autoscale_interval:
            self._ax.relim()
            self._ax.autoscale_view()
            self._last_autoscale_time = now
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)
        self._last_draw_time = now

    def _on_pick(self, event) -> None:
        artist = event.artist
        if artist in self._legend_map:
            line = self._legend_map[artist]
            line.set_visible(not line.get_visible())
            self._refresh_legend(force=True)
            self._redraw(force=True)
            return
        for line in self._lines.values():
            if artist == line:
                line.set_visible(not line.get_visible())
                self._refresh_legend(force=True)
                self._redraw(force=True)
                return

    def _prompt_filter(self, mode: str, prompt: str) -> None:
        try:
            value = input(prompt).strip()
        except Exception:
            return
        if not value:
            return
        if mode == "indicator":
            self._filters["indicator"] = {value.lower()}
        elif mode == "symbol":
            self._filters["symbol"] = {value.upper()}
        elif mode == "timeframe":
            self._filters["timeframe"] = {value.lower()}
        self._apply_filters()

    def _prompt_metric(self) -> None:
        try:
            value = input("Y metric (e.g., objective_delta, profit_factor): ").strip()
        except Exception:
            return
        if not value:
            return
        if value not in _METRIC_DEFS:
            self._update_status(f"Unknown metric '{value}'.")
            return
        self._y_metric = value
        self._refresh_lines()
        self._apply_filters()

    def _prompt_band_filter(self) -> None:
        try:
            value = input("Band filter (metric:min:max), blank clears: ").strip()
        except Exception:
            return
        if not value:
            self._set_band_filter(None, None, None)
            self._apply_filters()
            return
        parts = value.split(":")
        metric = parts[0].strip()
        if metric not in _METRIC_DEFS:
            self._update_status(f"Unknown metric '{metric}'.")
            return
        band_min = None
        band_max = None
        if len(parts) > 1 and parts[1].strip():
            try:
                band_min = float(parts[1].strip())
            except ValueError:
                self._update_status(f"Invalid band min '{parts[1]}'.")
                return
        if len(parts) > 2 and parts[2].strip():
            try:
                band_max = float(parts[2].strip())
            except ValueError:
                self._update_status(f"Invalid band max '{parts[2]}'.")
                return
        self._set_band_filter(metric, band_min, band_max)
        self._apply_filters()

    def _toggle_lines_by_query(self, query: str) -> None:
        if not query:
            return
        query_lower = query.lower()
        toggled = 0
        for label, line in self._lines.items():
            if query_lower in label.lower():
                line.set_visible(not line.get_visible())
                toggled += 1
        if toggled == 0:
            self._update_status(f"Filter: {self._filter_mode} | No lines matched '{query}'")
        self._refresh_legend(force=True)
        self._redraw(force=True)

    def _on_key_press(self, event) -> None:
        if event.key is None:
            return
        key = event.key.lower()
        if key == "a":
            self._filters = {"indicator": set(), "symbol": set(), "timeframe": set()}
            self._apply_filters()
        elif key == "i":
            self._prompt_filter("indicator", "Filter indicator (substring): ")
        elif key == "s":
            self._prompt_filter("symbol", "Filter symbol (exact, e.g., ADAUSDT): ")
        elif key == "t":
            self._prompt_filter("timeframe", "Filter timeframe (exact, e.g., 5m): ")
        elif key == "l":
            self._legend_enabled = not self._legend_enabled
            self._refresh_legend(force=True)
            self._redraw(force=True)
        elif key == "y":
            self._prompt_metric()
        elif key == "x":
            self._x_mode = "rate" if self._x_mode == "elapsed" else "elapsed"
            self._refresh_lines()
            self._apply_filters()
        elif key == "b":
            self._prompt_band_filter()

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
            series["metrics"] = {}
            series["params"] = []
        else:
            self._series[indicator_name] = {"x": [], "metrics": {}, "params": []}
        line = self._lines.get(indicator_name)
        if line is not None:
            line.set_data([], [])
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def set_baseline(self, indicator_name: str, objective: float) -> None:
        if not np.isfinite(objective):
            return
        self.set_baseline_metrics(indicator_name, {"objective_best": objective})

    def set_baseline_metrics(self, indicator_name: str, metrics: Dict[str, float]) -> None:
        if not metrics:
            return
        baseline = self._baseline_values.setdefault(indicator_name, {})
        baseline.update(metrics)
        if "objective_best" in baseline:
            baseline["objective_delta"] = 0.0

    def update(
        self,
        indicator_name: str,
        objective: float,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
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

        metrics_map = dict(metrics or {})
        if "objective_best" not in metrics_map:
            metrics_map["objective_best"] = objective
        baseline_obj = self._baseline_values.get(indicator_name, {}).get("objective_best")
        metrics_map.setdefault(
            "objective_delta",
            metrics_map["objective_best"] - baseline_obj if baseline_obj is not None else metrics_map["objective_best"]
        )
        series = self._series.setdefault(indicator_name, {"x": [], "metrics": {}, "params": []})
        series["x"].append(elapsed)
        for key in _METRIC_KEYS:
            series["metrics"].setdefault(key, []).append(metrics_map.get(key))
        series["params"].append(_format_params(params))
        if len(series["x"]) > self._max_points:
            series["x"] = series["x"][-self._max_points:]
            for key in _METRIC_KEYS:
                series["metrics"][key] = series["metrics"][key][-self._max_points:]
            series["params"] = series["params"][-self._max_points:]

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
            line.set_picker(5)
            self._register_line(indicator_name)
            if not self._line_matches_filter(indicator_name):
                line.set_visible(False)
            self._refresh_legend()

        x_vals, y_vals = self._get_series_xy(indicator_name)
        line.set_data(x_vals, y_vals)
        self._redraw()


class PlotlyRealtimePlotter:
    """Realtime plot using Plotly Dash with filter controls."""

    def __init__(self):
        self._init_attempted = False
        self._enabled = False
        self._app = None
        self._port = None
        self._thread = None
        self._start_times = {}
        self._series = {}
        self._baseline_values = {}
        self._line_meta = {}
        self._max_points = 300
        self._lock = threading.Lock()
        self._opened = False
        self._last_options = {"indicator": [], "symbol": [], "timeframe": []}
        self._default_y_metric = "objective_delta"
        self._default_x_mode = "elapsed"
        self._default_band_metric = "objective_overall"

    def _parse_label(self, label: str):
        indicator = label
        symbol = ""
        timeframe = ""
        if ":" in label:
            indicator, remainder = label.split(":", 1)
            if "@" in remainder:
                symbol, timeframe = remainder.split("@", 1)
            else:
                symbol = remainder
        return indicator.strip(), symbol.strip(), timeframe.strip()

    def _register_line(self, label: str) -> None:
        indicator, symbol, timeframe = self._parse_label(label)
        self._line_meta[label] = {
            "indicator": indicator,
            "symbol": symbol.upper(),
            "timeframe": timeframe.lower()
        }

    def _find_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def _wait_for_server(self, url: str, timeout_s: float = 5.0) -> bool:
        import urllib.request
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(0.2)
        return False

    def _ensure_ready(self) -> bool:
        if self._init_attempted:
            return self._enabled
        self._init_attempted = True
        try:
            from dash import Dash, dcc, html, Input, Output, State, ctx
            from dash.dependencies import ALL
            import plotly.graph_objects as go
        except Exception as exc:
            logger.info("Plotly realtime plot disabled (dash/plotly not available): %s", exc)
            self._enabled = False
            return False

        self._port = self._find_free_port()
        self._app = Dash(__name__)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        logging.getLogger("dash").setLevel(logging.ERROR)
        self._app.logger.setLevel(logging.ERROR)
        metric_options = [{"label": _metric_label(key), "value": key} for key in _METRIC_KEYS]

        self._app.layout = html.Div(
            style={
                "fontFamily": "Helvetica, Arial, sans-serif",
                "padding": "12px",
                "backgroundColor": "#f7f7f7"
            },
            children=[
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        html.H3("Optimization Metric Explorer", style={"margin": "0 0 6px 0"}),
                        html.Div(
                            "Filter by indicator, symbol, timeframe, and metric bands.",
                            style={"fontSize": "12px", "color": "#555"}
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gap": "8px", "marginBottom": "8px", "gridTemplateColumns": "2fr 2fr 2fr auto"},
                    children=[
                        dcc.Dropdown(
                            id="indicator-filter",
                            options=[],
                            multi=True,
                            placeholder="Indicators"
                        ),
                        dcc.Dropdown(
                            id="symbol-filter",
                            options=[],
                            multi=True,
                            placeholder="Symbols"
                        ),
                        dcc.Dropdown(
                            id="timeframe-filter",
                            options=[],
                            multi=True,
                            placeholder="Timeframes"
                        ),
                        html.Button("Clear Filters", id="clear-filters", n_clicks=0),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gap": "8px", "marginBottom": "8px", "gridTemplateColumns": "2fr 1fr 2fr 1fr 1fr"},
                    children=[
                        dcc.Dropdown(
                            id="y-metric",
                            options=metric_options,
                            value=self._default_y_metric,
                            clearable=False
                        ),
                        dcc.Dropdown(
                            id="x-axis",
                            options=[
                                {"label": "Elapsed (s)", "value": "elapsed"},
                                {"label": "Improvement rate (/s)", "value": "rate"},
                            ],
                            value=self._default_x_mode,
                            clearable=False
                        ),
                        dcc.Dropdown(
                            id="band-metric",
                            options=metric_options,
                            value=self._default_band_metric,
                            clearable=True,
                            placeholder="Band metric"
                        ),
                        dcc.Input(
                            id="band-min",
                            type="number",
                            placeholder="Band min",
                            debounce=True
                        ),
                        dcc.Input(
                            id="band-max",
                            type="number",
                            placeholder="Band max",
                            debounce=True
                        ),
                    ],
                ),
                dcc.Graph(
                    id="objective-graph",
                    config={"displayModeBar": True, "responsive": True},
                    style={"height": "72vh"}
                ),
                dcc.Store(id="hidden-series", data=[]),
                html.Div(
                    id="legend-box",
                    style={
                        "marginTop": "8px",
                        "padding": "6px 8px",
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "6px",
                        "fontSize": "12px",
                        "maxHeight": "140px",
                        "overflowY": "auto"
                    },
                    children="Legend: no series yet."
                ),
                html.Div(
                    id="click-details",
                    style={
                        "marginTop": "8px",
                        "padding": "8px 10px",
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "6px",
                        "fontSize": "12px"
                    },
                    children="Click a line to reveal the exact combo and values."
                ),
                dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
            ],
        )

        def build_options():
            with self._lock:
                indicators = sorted({m["indicator"] for m in self._line_meta.values() if m["indicator"]})
                symbols = sorted({m["symbol"] for m in self._line_meta.values() if m["symbol"]})
                timeframes = sorted({m["timeframe"] for m in self._line_meta.values() if m["timeframe"]})
            self._last_options = {
                "indicator": indicators,
                "symbol": symbols,
                "timeframe": timeframes
            }
            return (
                [{"label": v, "value": v} for v in indicators],
                [{"label": v, "value": v} for v in symbols],
                [{"label": v, "value": v} for v in timeframes],
            )

        def build_figure(indicators, symbols, timeframes, y_metric, x_mode, band_metric, band_min, band_max, hidden):
            with self._lock:
                series = {
                    k: {
                        "x": list(v.get("x", [])),
                        "metrics": {mk: list(mv) for mk, mv in v.get("metrics", {}).items()},
                        "params": list(v.get("params", [])),
                    }
                    for k, v in self._series.items()
                }
                meta = dict(self._line_meta)
                baselines = {k: dict(v) for k, v in self._baseline_values.items()}

            indicators = set(indicators or [])
            symbols = set(symbols or [])
            timeframes = set(timeframes or [])
            if y_metric not in _METRIC_DEFS:
                y_metric = self._default_y_metric
            if x_mode not in ("elapsed", "rate"):
                x_mode = self._default_x_mode
            if band_metric not in _METRIC_DEFS:
                band_metric = None
            band_enabled = band_metric is not None and (band_min is not None or band_max is not None)

            fig = go.Figure()
            hidden_set = set(hidden or [])
            visible_labels = []
            for label, points in series.items():
                meta_info = meta.get(label, {})
                if indicators and meta_info.get("indicator") not in indicators:
                    continue
                if symbols and meta_info.get("symbol") not in symbols:
                    continue
                if timeframes and meta_info.get("timeframe") not in timeframes:
                    continue

                x_vals = points.get("x", [])
                y_vals = points.get("metrics", {}).get(y_metric, [])
                params_vals = points.get("params", [])
                if not x_vals:
                    continue
                if not y_vals:
                    continue
                if len(params_vals) < len(x_vals):
                    params_vals = params_vals + ["N/A"] * (len(x_vals) - len(params_vals))
                elif len(params_vals) > len(x_vals):
                    params_vals = params_vals[-len(x_vals):]

                if band_enabled:
                    band_vals = points.get("metrics", {}).get(band_metric, [])
                    latest_band = _last_non_none(band_vals)
                    if latest_band is None:
                        continue
                    if band_min is not None and latest_band < band_min:
                        continue
                    if band_max is not None and latest_band > band_max:
                        continue

                visible_labels.append(label)

                if x_mode == "rate":
                    baseline_val = baselines.get(label, {}).get(y_metric)
                    x_vals = _compute_rate_series(x_vals, y_vals, baseline_val)

                x_label = "rate_per_s" if x_mode == "rate" else "elapsed_s"
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=label,
                        uid=label,
                        customdata=[label] * len(x_vals),
                        text=params_vals,
                        line={"width": 2.5},
                        marker={"size": 6, "opacity": 0.15},
                        visible="legendonly" if label in hidden_set else True,
                        hovertemplate=(
                            "combo=%{customdata}<br>"
                            "params=%{text}<br>"
                            f"{_metric_label(y_metric)}=%{{y:.4f}}<br>"
                            f"{x_label}=%{{x:.4f}}<extra></extra>"
                        )
                    )
                )

            if x_mode == "rate":
                x_title = f"Improvement rate ({_metric_label(y_metric)} / s)"
            else:
                x_title = "Elapsed (s)"
            fig.update_layout(
                template="plotly_white",
                margin={"l": 40, "r": 10, "t": 30, "b": 40},
                showlegend=False,
                hovermode="closest",
                hoverdistance=50,
                spikedistance=50,
                uirevision="keep"
            )
            fig.update_xaxes(title=x_title)
            fig.update_yaxes(title=_metric_label(y_metric))
            return fig, visible_labels

        @self._app.callback(
            Output("objective-graph", "figure"),
            Output("indicator-filter", "options"),
            Output("symbol-filter", "options"),
            Output("timeframe-filter", "options"),
            Output("indicator-filter", "value"),
            Output("symbol-filter", "value"),
            Output("timeframe-filter", "value"),
            Output("band-min", "value"),
            Output("band-max", "value"),
            Output("legend-box", "children"),
            Input("update-interval", "n_intervals"),
            Input("indicator-filter", "value"),
            Input("symbol-filter", "value"),
            Input("timeframe-filter", "value"),
            Input("clear-filters", "n_clicks"),
            Input("y-metric", "value"),
            Input("x-axis", "value"),
            Input("band-metric", "value"),
            Input("band-min", "value"),
            Input("band-max", "value"),
            Input("hidden-series", "data"),
        )
        def update_plot(
            _,
            indicator_vals,
            symbol_vals,
            timeframe_vals,
            clear_clicks,
            y_metric,
            x_mode,
            band_metric,
            band_min,
            band_max,
            hidden_series
        ):
            if ctx.triggered_id == "clear-filters":
                indicator_vals = []
                symbol_vals = []
                timeframe_vals = []
                band_min = None
                band_max = None

            indicator_opts, symbol_opts, timeframe_opts = build_options()

            indicator_vals = [v for v in (indicator_vals or []) if v in self._last_options["indicator"]]
            symbol_vals = [v for v in (symbol_vals or []) if v in self._last_options["symbol"]]
            timeframe_vals = [v for v in (timeframe_vals or []) if v in self._last_options["timeframe"]]

            fig, visible_labels = build_figure(
                indicator_vals,
                symbol_vals,
                timeframe_vals,
                y_metric,
                x_mode,
                band_metric,
                band_min,
                band_max,
                hidden_series
            )
            if visible_labels:
                hidden_set = set(hidden_series or [])
                legend_children = []
                for label in visible_labels:
                    is_hidden = label in hidden_set
                    legend_children.append(
                        html.Button(
                            label,
                            id={"type": "legend-item", "label": label},
                            n_clicks=0,
                            **{"data-label": label},
                            style={
                                "display": "block",
                                "width": "100%",
                                "textAlign": "left",
                                "border": "none",
                                "background": "none",
                                "padding": "2px 0",
                                "cursor": "pointer",
                                "color": "#999" if is_hidden else "#222"
                            }
                        )
                    )
            else:
                legend_children = "Legend: no series match the filters."
            return (
                fig,
                indicator_opts,
                symbol_opts,
                timeframe_opts,
                indicator_vals,
                symbol_vals,
                timeframe_vals,
                band_min,
                band_max,
                legend_children,
            )

        @self._app.callback(
            Output("hidden-series", "data"),
            Input({"type": "legend-item", "label": ALL}, "n_clicks"),
            State("hidden-series", "data"),
        )
        def toggle_hidden_series(_, hidden_series):
            if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
                return hidden_series or []
            triggered = ctx.triggered[0] if ctx.triggered else None
            clicks = triggered.get("value") if triggered else None
            if not clicks:
                return hidden_series or []
            label = ctx.triggered_id.get("label")
            if not label:
                return hidden_series or []
            hidden = list(hidden_series or [])
            if label in hidden:
                hidden.remove(label)
            else:
                hidden.append(label)
            return hidden

        @self._app.callback(
            Output("click-details", "children"),
            Input("objective-graph", "clickData"),
            Input("objective-graph", "hoverData"),
        )
        def show_click_details(click_data, hover_data):
            prop_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            if prop_id.endswith("clickData") and click_data:
                source = click_data
            elif prop_id.endswith("hoverData") and hover_data:
                source = hover_data
            else:
                source = click_data or hover_data
            if not source or "points" not in source or not source["points"]:
                return "Click a line to reveal the exact combo and values."
            point = source["points"][0]
            combo = point.get("customdata") or "N/A"
            params = point.get("text") or "N/A"
            x_val = point.get("x")
            y_val = point.get("y")
            return f"combo={combo} | params={params} | x={x_val:.4f} | y={y_val:.4f}"

        def run_server():
            run_fn = getattr(self._app, "run", None)
            if callable(run_fn):
                run_fn(host="127.0.0.1", port=self._port, debug=False, use_reloader=False)
            else:
                self._app.run_server(
                    host="127.0.0.1",
                    port=self._port,
                    debug=False,
                    use_reloader=False
                )

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        self._enabled = True
        logger.info("Plotly realtime chart running at http://127.0.0.1:%s", self._port)
        url = f"http://127.0.0.1:{self._port}"
        print(f"[PLOTLY] Realtime chart: {url}")
        ready = self._wait_for_server(url)
        if not ready:
            logger.warning("Plotly server did not respond at %s", url)
        if not self._opened:
            try:
                webbrowser.open(url)
                self._opened = True
            except Exception:
                pass
        return True

    def start_indicator(self, indicator_name: str) -> None:
        if not self._ensure_ready():
            return
        with self._lock:
            self._start_times[indicator_name] = time.time()
            self._series[indicator_name] = {"x": [], "metrics": {}, "params": []}
            self._baseline_values.pop(indicator_name, None)
            self._register_line(indicator_name)

    def set_baseline(self, indicator_name: str, objective: float) -> None:
        if not self._ensure_ready():
            return
        with self._lock:
            self._baseline_values.setdefault(indicator_name, {})["objective_best"] = objective
            self._baseline_values[indicator_name]["objective_delta"] = 0.0

    def set_baseline_metrics(self, indicator_name: str, metrics: Dict[str, float]) -> None:
        if not self._ensure_ready():
            return
        if not metrics:
            return
        with self._lock:
            baseline = self._baseline_values.setdefault(indicator_name, {})
            baseline.update(metrics)
            if "objective_best" in baseline:
                baseline["objective_delta"] = 0.0

    def update(
        self,
        indicator_name: str,
        objective: float,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self._ensure_ready():
            return
        now = time.time()
        with self._lock:
            start_time = self._start_times.get(indicator_name)
            if start_time is None:
                start_time = now
                self._start_times[indicator_name] = start_time
            elapsed = now - start_time
            metrics_map = dict(metrics or {})
            if "objective_best" not in metrics_map:
                metrics_map["objective_best"] = objective
            baseline_obj = self._baseline_values.get(indicator_name, {}).get("objective_best")
            metrics_map.setdefault(
                "objective_delta",
                metrics_map["objective_best"] - baseline_obj if baseline_obj is not None else metrics_map["objective_best"]
            )
            series = self._series.setdefault(indicator_name, {"x": [], "metrics": {}, "params": []})
            series["x"].append(elapsed)
            for key in _METRIC_KEYS:
                series["metrics"].setdefault(key, []).append(metrics_map.get(key))
            series["params"].append(_format_params(params))
            if len(series["x"]) > self._max_points:
                series["x"] = series["x"][-self._max_points:]
                for key in _METRIC_KEYS:
                    series["metrics"][key] = series["metrics"][key][-self._max_points:]
                series["params"] = series["params"][-self._max_points:]
            if indicator_name not in self._line_meta:
                self._register_line(indicator_name)


_REALTIME_PLOTTER = None


def get_realtime_plotter() -> RealtimeBestPlotter:
    global _REALTIME_PLOTTER
    if _REALTIME_PLOTTER is None:
        try:
            plotly_plotter = PlotlyRealtimePlotter()
            if plotly_plotter._ensure_ready():
                _REALTIME_PLOTTER = plotly_plotter
            else:
                _REALTIME_PLOTTER = RealtimeBestPlotter()
        except Exception:
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
        indicator_label: Optional[str] = None
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
        self._plot_initialized = False
        self._baseline_metrics = None
        self._baseline_objective = None
        self._baseline_metrics_map = {}

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
        metrics_list = []
        
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
                metrics_list.append(metrics)
                
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

            aggregated_metrics = self._aggregate_metrics(metrics_list)
            metrics_map = _metrics_from_backtest(aggregated_metrics)
            metrics_map["objective_best"] = avg_objective
            metrics_map["objective_overall"] = calculate_objective_score(aggregated_metrics)

            if self.realtime_plotter:
                params_for_plot = {
                    p.name: params.get(p.name)
                    for p in self.optimizable_params
                }
                if not self._plot_initialized:
                    self.realtime_plotter.start_indicator(self.indicator_name)
                    self.realtime_plotter.set_baseline(self.indicator_name, baseline)
                    if self._baseline_metrics_map:
                        self.realtime_plotter.set_baseline_metrics(self.indicator_name, self._baseline_metrics_map)
                    self._plot_initialized = True
                self.realtime_plotter.update(self.indicator_name, avg_objective, metrics_map, params_for_plot)
            
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

        self._baseline_metrics = original_metrics
        self._baseline_objective = original_objective
        self._baseline_metrics_map = _metrics_from_backtest(original_metrics)
        self._baseline_metrics_map["objective_best"] = original_objective
        self._baseline_metrics_map["objective_overall"] = calculate_objective_score(original_metrics)
        
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

        # Evaluate lockbox holdout performance if enabled
        holdout_metrics = None
        holdout_original_metrics = None
        holdout_per_symbol_metrics = {}
        holdout_data_usage_info = {}

        if self.holdout_enabled and self.holdout_translators:
            logger.info("Evaluating lockbox holdout performance...")
            holdout_original_per_symbol = self._evaluate_holdout_params_per_symbol(self.original_params)

            if self.is_multi_timeframe:
                all_holdout_original = []
                for symbol_dict in holdout_original_per_symbol.values():
                    all_holdout_original.extend(symbol_dict.values())
                holdout_original_metrics = self._aggregate_metrics(all_holdout_original)
            else:
                holdout_original_metrics = self._aggregate_metrics(list(holdout_original_per_symbol.values()))

            if best_params == self.original_params:
                holdout_best_per_symbol = holdout_original_per_symbol
                holdout_metrics = holdout_original_metrics
            else:
                holdout_best_per_symbol = self._evaluate_holdout_params_per_symbol(best_params)
                if self.is_multi_timeframe:
                    all_holdout_best = []
                    for symbol_dict in holdout_best_per_symbol.values():
                        all_holdout_best.extend(symbol_dict.values())
                    holdout_metrics = self._aggregate_metrics(all_holdout_best)
                else:
                    holdout_metrics = self._aggregate_metrics(list(holdout_best_per_symbol.values()))

            if self.is_multi_timeframe:
                for symbol in holdout_original_per_symbol:
                    holdout_per_symbol_metrics[symbol] = {}
                    holdout_data_usage_info[symbol] = {}
                    for timeframe in holdout_original_per_symbol[symbol]:
                        holdout_per_symbol_metrics[symbol][timeframe] = {
                            'original': holdout_original_per_symbol[symbol].get(timeframe, BacktestMetrics()),
                            'optimized': holdout_best_per_symbol.get(symbol, {}).get(timeframe, BacktestMetrics())
                        }
                        key = (symbol, timeframe)
                        if key in self.holdout_backtesters and key in self.holdout_data_frames:
                            holdout_data_usage_info[symbol][timeframe] = self._extract_data_usage_info(
                                self.holdout_backtesters[key],
                                self.holdout_data_frames[key]
                            )
            else:
                for symbol in holdout_original_per_symbol:
                    holdout_per_symbol_metrics[symbol] = {
                        'original': holdout_original_per_symbol.get(symbol, BacktestMetrics()),
                        'optimized': holdout_best_per_symbol.get(symbol, BacktestMetrics())
                    }
                    if symbol in self.holdout_backtesters and symbol in self.holdout_data_frames:
                        holdout_data_usage_info[symbol] = {
                            '': self._extract_data_usage_info(
                                self.holdout_backtesters[symbol],
                                self.holdout_data_frames[symbol]
                            )
                        }
        
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
            backtester_overrides=self.backtester_overrides,
            holdout_ratio=self.holdout_ratio,
            holdout_gap_bars=self.holdout_gap_bars_effective,
            holdout_metrics=holdout_metrics,
            holdout_original_metrics=holdout_original_metrics,
            holdout_per_symbol_metrics=holdout_per_symbol_metrics,
            holdout_data_usage_info=holdout_data_usage_info
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
