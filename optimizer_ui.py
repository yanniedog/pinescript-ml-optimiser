"""
UI components for the optimizer, including keyboard monitoring and realtime plotting.
"""

import sys
import time
import math
import threading
import logging
import socket
import webbrowser
import json
from typing import Dict, Any, List, Optional, Tuple, Set

# Platform-specific keyboard handling
if sys.platform == 'win32':
    import msvcrt
else:
    import select
    # termios and tty are only available on Unix
    try:
        import termios
        import tty
    except ImportError:
        pass

from optimizer_utils import (
    _metric_label, 
    _compute_rate_series, 
    _METRIC_KEYS,
    _format_param_value,
    _format_params,
    _downsample_series_for_plot
)

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
        self._max_render_points = 500  # Points per line for efficient rendering (no storage limit)
        self._legend = None
        self._legend_enabled = True
        self._legend_max_lines = 25
        self._legend_map = {}
        self._last_legend_update = 0.0
        self._legend_update_interval = 2.0
        self._line_meta = {}
        self._y_metric = "objective_delta"
        self._x_mode = "elapsed"
        self._x_modes = ["elapsed", "rate", "trial", "trials_per_second"]
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
            self._plt.pause(0.01)  # Required for Windows interactivity
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
        elif self._x_mode == "trial":
            x_label = "Trial #"
        elif self._x_mode == "trials_per_second":
            x_label = "Trials / second"
        else:
            x_label = "Elapsed time (s)"
        self._ax.set_xlabel(x_label)
        self._ax.set_ylabel(_metric_label(self._y_metric))

    def _get_series_xy(self, label: str) -> Tuple[List[float], List[Optional[float]]]:
        series = self._series.get(label)
        if not series:
            return [], []
        
        # Downsample for efficient rendering
        downsampled = _downsample_series_for_plot(series, _METRIC_KEYS, self._max_render_points)
        
        x_vals = downsampled.get("x", [])
        y_vals = downsampled.get("metrics", {}).get(self._y_metric, [])
        if self._x_mode == "rate":
            baseline = self._baseline_values.get(label, {}).get(self._y_metric)
            x_vals = _compute_rate_series(x_vals, y_vals, baseline)
        elif self._x_mode == "trial":
            x_vals = downsampled.get("trials", [])
        elif self._x_mode == "trials_per_second":
            x_vals = downsampled.get("metrics", {}).get("trials_per_second", [])
        return x_vals, y_vals

    def _refresh_lines(self) -> None:
        for label, line in self._lines.items():
            x_vals, y_vals = self._get_series_xy(label)
            line.set_data(x_vals, y_vals)
        self._update_axis_labels()
        self._redraw(force=True)

    def _init_ui(self, RadioButtons, TextBox, Button):
        """Initialize interactive widgets."""
        # Y-Axis Selector
        ax_y = self._fig.add_axes([0.82, 0.55, 0.15, 0.35])
        self._ui_axes["y_select"] = ax_y
        metric_labels = [_metric_label(k) for k in _METRIC_KEYS]
        # Shorten labels for UI
        short_labels = [l.split("(")[0].strip() for l in metric_labels]
        rb_y = RadioButtons(ax_y, short_labels, active=1)  # Default objective_delta
        
        def set_y(label):
            idx = short_labels.index(label)
            self._y_metric = _METRIC_KEYS[idx]
            self._refresh_lines()
            self._autoscale()
        
        rb_y.on_clicked(set_y)
        self._ui_widgets["y_select"] = rb_y
        
        # X-Axis Selector
        ax_x = self._fig.add_axes([0.82, 0.35, 0.15, 0.15])
        self._ui_axes["x_select"] = ax_x
        rb_x = RadioButtons(ax_x, ["Elapsed", "Rate", "Trial", "Trials/s"], active=0)
        
        def set_x(label):
            mode = label.lower().replace("/", "_per_")
            self._x_mode = mode if mode != "trials_per_s" else "trials_per_second"
            self._refresh_lines()
            self._autoscale()
        
        rb_x.on_clicked(set_x)
        self._ui_widgets["x_select"] = rb_x
        
        # Filters
        ax_fi = self._fig.add_axes([0.15, 0.05, 0.2, 0.04])
        self._indicator_input = TextBox(ax_fi, "Ind:", initial="")
        def submit_ind(text):
            self._apply_filter("indicator", text)
        self._indicator_input.on_submit(submit_ind)
        self._ui_widgets["filter_ind"] = self._indicator_input
        
        ax_fs = self._fig.add_axes([0.45, 0.05, 0.2, 0.04])
        self._symbol_input = TextBox(ax_fs, "Sym:", initial="")
        def submit_sym(text):
            self._apply_filter("symbol", text)
        self._symbol_input.on_submit(submit_sym)
        self._ui_widgets["filter_sym"] = self._symbol_input
        
        ax_ft = self._fig.add_axes([0.75, 0.05, 0.15, 0.04])
        self._timeframe_input = TextBox(ax_ft, "TF:", initial="")
        def submit_tf(text):
            self._apply_filter("timeframe", text)
        self._timeframe_input.on_submit(submit_tf)
        self._ui_widgets["filter_tf"] = self._timeframe_input

        # Clear Filters Button
        ax_cl = self._fig.add_axes([0.82, 0.25, 0.15, 0.05])
        btn_cl = Button(ax_cl, "Clear Filters")
        def clear_filters(event):
            self._filters = {"indicator": set(), "symbol": set(), "timeframe": set()}
            self._indicator_input.set_val("")
            self._symbol_input.set_val("")
            self._timeframe_input.set_val("")
            self._update_visibility()
        btn_cl.on_clicked(clear_filters)
        self._ui_widgets["clear"] = btn_cl

    def _apply_filter(self, category: str, text: str) -> None:
        """Apply a filter string (comma-separated) to a category."""
        tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
        self._filters[category] = set(tokens)
        self._update_visibility()

    def _matches_filters(self, label: str) -> bool:
        """Check if a line label matches current filters."""
        if not any(self._filters.values()):
            return True
        
        meta = self._line_meta.get(label)
        if not meta:
            # Try to parse on fly if missing
            self._register_line(label)
            meta = self._line_meta.get(label)
        
        if self._filters["indicator"]:
            if not any(f in meta["indicator"] for f in self._filters["indicator"]):
                return False
        if self._filters["symbol"]:
            if not any(f in meta["symbol"] for f in self._filters["symbol"]):
                return False
        if self._filters["timeframe"]:
            if not any(f in meta["timeframe"] for f in self._filters["timeframe"]):
                return False
        return True

    def _update_visibility(self) -> None:
        """Update line visibility based on filters."""
        for label, line in self._lines.items():
            visible = self._matches_filters(label)
            line.set_visible(visible)
            # Also hide series if filtered out to avoid autoscaling on hidden data
            if not visible and label in self._series:
                 # This is tricky in matplotlib; set_visible handles drawing, but autoscale uses data limits.
                 # We can rely on manual autoscaling logic or just let it be.
                 pass
        
        filter_status = []
        for k, v in self._filters.items():
            if v:
                filter_status.append(f"{k}={','.join(v)}")
        
        status = f"Filters: {' | '.join(filter_status) if filter_status else 'all'}"
        self._update_status(status)
        self._autoscale()

    def _on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'a':
            self._filters = {"indicator": set(), "symbol": set(), "timeframe": set()}
            if self._indicator_input: self._indicator_input.set_val("")
            if self._symbol_input: self._symbol_input.set_val("")
            if self._timeframe_input: self._timeframe_input.set_val("")
            self._update_visibility()
        elif event.key == 'l':
            self._legend_enabled = not self._legend_enabled
            self._update_legend()
        elif event.key == 'y':
            # Cycle Y metric
            current_idx = _METRIC_KEYS.index(self._y_metric)
            next_idx = (current_idx + 1) % len(_METRIC_KEYS)
            self._y_metric = _METRIC_KEYS[next_idx]
            self._refresh_lines()
            self._autoscale()
        elif event.key == 'x':
            # Cycle X mode
            current_idx = self._x_modes.index(self._x_mode)
            next_idx = (current_idx + 1) % len(self._x_modes)
            self._x_mode = self._x_modes[next_idx]
            self._refresh_lines()
            self._autoscale()

    def _on_pick(self, event):
        """Handle clicking on lines to toggle isolation/visibility."""
        artist = event.artist
        label = artist.get_label()
        if label in self._lines:
            # If shift is held, isolate this one
            if event.mouseevent.key == 'shift':
                for l, line in self._lines.items():
                    line.set_visible(l == label)
            else:
                # Toggle
                visible = not artist.get_visible()
                artist.set_visible(visible)
            self._update_legend()  # Update legend to reflect visibility changes
            self._redraw()

    def start_indicator(self, indicator_name: str) -> None:
        """Register start time for an indicator."""
        if not self._ensure_ready():
            return
        if indicator_name not in self._start_times:
            self._start_times[indicator_name] = time.time()

    def set_baseline(self, indicator_name: str, baseline_objective: float) -> None:
        """Set baseline objective for an indicator."""
        if not self._ensure_ready():
            return
        self._baseline_values.setdefault(indicator_name, {})["objective_best"] = baseline_objective

    def set_baseline_metrics(self, indicator_name: str, metrics: Dict[str, float]) -> None:
        """Set all baseline metrics."""
        if not self._ensure_ready():
            return
        self._baseline_values[indicator_name] = metrics.copy()

    def update(
        self,
        indicator_name: str,
        best_objective: float,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        trial_number: int = 0
    ) -> None:
        """Update plot with new best objective."""
        if not self._ensure_ready():
            return

        current_time = time.time()
        elapsed = current_time - self._start_times.get(indicator_name, current_time)
        
        self._register_line(indicator_name)
        
        # Update series data
        if indicator_name not in self._series:
            self._series[indicator_name] = {"x": [], "metrics": {}, "params": [], "trials": []}
            
        series = self._series[indicator_name]
        series["x"].append(elapsed)
        series.setdefault("trials", []).append(trial_number)
        series.setdefault("params", []).append(params)
        
        for key in _METRIC_KEYS:
            val = metrics.get(key)
            series.setdefault("metrics", {}).setdefault(key, []).append(val)

        # Update Plot Line (only if matplotlib _ax is available)
        if self._ax is not None:
            x_vals, y_vals = self._get_series_xy(indicator_name)
            
            if indicator_name not in self._lines:
                color = self._get_indicator_color(indicator_name.split(":")[0])
                line, = self._ax.plot(
                    x_vals, 
                    y_vals, 
                    label=indicator_name,
                    color=color,
                    marker='o',
                    markersize=4,
                    picker=5  # Enable picking
                )
                self._lines[indicator_name] = line
            else:
                self._lines[indicator_name].set_data(x_vals, y_vals)
                
            self._best_objectives[indicator_name] = best_objective
            self._redraw()

    def record_trial_progress(
        self,
        combo_label: str,
        trial_number: int,
        elapsed_seconds: float,
        metrics: Dict[str, float]
    ) -> None:
        """Track trial progress as a separate '[trials]' series."""
        if not self._ensure_ready():
            return
        # This implementation was just adding to memory in the original code but not plotting separate lines.
        # To avoid cluttering the UI with thousands of points, we can either:
        # 1. Add a scatter series for trials
        # 2. Just store it for now.
        # Given the complexity, we'll store it in a separate series key but maybe not plot it by default.
        progress_label = f"{combo_label} [trials]"
        self._register_line(progress_label)
        
        series = self._series.setdefault(
            progress_label, 
            {"x": [], "metrics": {}, "params": [], "trials": []}
        )
        series["x"].append(elapsed_seconds)
        series.setdefault("trials", []).append(trial_number)
        series.setdefault("params", []).append(f"trial={trial_number}")
        for key in _METRIC_KEYS:
            series.setdefault("metrics", {}).setdefault(key, []).append(metrics.get(key))

    def _redraw(self, force: bool = False) -> None:
        """Redraw plot if interval passed."""
        now = time.time()
        if not force and (now - self._last_draw_time) < self._min_draw_interval:
            return
            
        self._last_draw_time = now
        
        if (now - self._last_autoscale_time) > self._autoscale_interval:
            self._autoscale()
            self._last_autoscale_time = now
            
        if (now - self._last_legend_update) > self._legend_update_interval:
            self._update_legend()
            self._last_legend_update = now
            
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass

    def _autoscale(self) -> None:
        """Autoscale axes."""
        if self._ax:
            self._ax.relim()
            self._ax.autoscale_view()

    def _update_legend(self) -> None:
        """Update legend efficiently."""
        if not self._legend_enabled or not self._ax:
            if self._legend:
                self._legend.remove()
                self._legend = None
            return

        handles, labels = self._ax.get_legend_handles_labels()
        # Filter visible
        visible_handles = []
        visible_labels = []
        for h, l in zip(handles, labels):
            if h.get_visible():
                visible_handles.append(h)
                visible_labels.append(l)
        
        if not visible_handles:
            return

        # Limit legend size
        if len(visible_handles) > self._legend_max_lines:
            visible_handles = visible_handles[:self._legend_max_lines]
            visible_labels = visible_labels[:self._legend_max_lines]
            visible_labels[-1] = f"... (+{len(handles)-self._legend_max_lines})"

        self._legend = self._ax.legend(
            visible_handles, 
            visible_labels, 
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize='small'
        )


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
        self._max_render_points = 500  # Points per line for efficient rendering (no storage limit)
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

        self._port = 9107
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
                                 {"label": "Trial #", "value": "trial"},
                                 {"label": "Trials/sec (running rate)", "value": "trials_per_second"},
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
                html.Div(
                    style={"display": "flex", "gap": "16px", "marginBottom": "8px", "alignItems": "center"},
                    children=[
                        dcc.Checklist(
                            id="improving-mcc-only",
                            options=[{"label": " Show only improving MCC values", "value": "improving_mcc"}],
                            value=[],
                            style={"fontSize": "13px"}
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
                # Filter out [trials] suffix labels that clutter the filter dropdowns
                indicators = sorted({
                    m["indicator"] for m in self._line_meta.values() 
                    if m["indicator"] and not m["indicator"].endswith("[trials]")
                })
                symbols = sorted({
                    m["symbol"] for m in self._line_meta.values() 
                    if m["symbol"] and not m["symbol"].endswith("[trials]")
                })
                timeframes = sorted({
                    m["timeframe"] for m in self._line_meta.values() 
                    if m["timeframe"] and not m["timeframe"].endswith("[trials]")
                })
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

        def _last_non_none(lst):
            """Return the last non-None value in a list, or None if all None."""
            for val in reversed(lst):
                if val is not None:
                    return val
            return None

        def build_figure(indicators, symbols, timeframes, y_metric, x_mode, band_metric, band_min, band_max, hidden, improving_mcc_only=False):
            from optimizer_utils import _METRIC_DEFS
            with self._lock:
                # Copy and downsample series for efficient rendering
                series = {}
                for k, v in self._series.items():
                    full_series = {
                        "x": list(v.get("x", [])),
                        "metrics": {mk: list(mv) for mk, mv in v.get("metrics", {}).items()},
                        "params": list(v.get("params", [])),
                        "trials": list(v.get("trials", [])),
                    }
                    # Downsample for rendering efficiency
                    series[k] = _downsample_series_for_plot(full_series, _METRIC_KEYS, self._max_render_points)
                meta = dict(self._line_meta)
                baselines = {k: dict(v) for k, v in self._baseline_values.items()}

            indicators = set(indicators or [])
            symbols = set(symbols or [])
            timeframes = set(timeframes or [])
            if y_metric not in _METRIC_DEFS:
                y_metric = self._default_y_metric
            if x_mode not in ("elapsed", "rate", "trial", "trials_per_second"):
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

                x_elapsed = points.get("x", [])
                trial_vals = points.get("trials", [])
                y_vals = points.get("metrics", {}).get(y_metric, [])
                params_vals = points.get("params", [])
                mcc_vals = points.get("metrics", {}).get("mcc", [])
                
                if x_mode == "trial":
                    if not trial_vals:
                        continue
                    x_vals = trial_vals
                elif x_mode == "trials_per_second":
                    x_vals = points.get("metrics", {}).get("trials_per_second", [])
                    if not x_vals:
                        continue
                else:
                    x_vals = x_elapsed
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

                # Filter for improving MCC values only
                if improving_mcc_only and mcc_vals:
                    improving_indices = [0]  # Always keep the first point
                    best_mcc_so_far = mcc_vals[0] if mcc_vals[0] is not None else float('-inf')
                    for i in range(1, len(mcc_vals)):
                        if mcc_vals[i] is not None and mcc_vals[i] > best_mcc_so_far:
                            improving_indices.append(i)
                            best_mcc_so_far = mcc_vals[i]
                    
                    # Filter all arrays to only keep improving points
                    if improving_indices:
                        x_vals = [x_vals[i] for i in improving_indices if i < len(x_vals)]
                        y_vals = [y_vals[i] for i in improving_indices if i < len(y_vals)]
                        params_vals = [params_vals[i] for i in improving_indices if i < len(params_vals)]
                        if not x_vals or not y_vals:
                            continue

                visible_labels.append(label)

                if x_mode == "rate":
                    baseline_val = baselines.get(label, {}).get(y_metric)
                    x_vals = _compute_rate_series(x_vals, y_vals, baseline_val)

                if x_mode == "rate":
                    x_label = "rate_per_s"
                elif x_mode == "trial":
                    x_label = "trial"
                elif x_mode == "trials_per_second":
                    x_label = "trials_per_second"
                else:
                    x_label = "elapsed_s"
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
            elif x_mode == "trial":
                x_title = "Trial #"
            elif x_mode == "trials_per_second":
                x_title = "Trials/sec"
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
            Input("improving-mcc-only", "value"),
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
            hidden_series,
            improving_mcc_only
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

            # Check if improving MCC filter is enabled
            show_improving_only = "improving_mcc" in (improving_mcc_only or [])

            fig, visible_labels = build_figure(
                indicator_vals,
                symbol_vals,
                timeframe_vals,
                y_metric,
                x_mode,
                band_metric,
                band_min,
                band_max,
                hidden_series,
                improving_mcc_only=show_improving_only
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
                return "Click a line to reveal the exact combo and values."
            try:
                point = source["points"][0]
                combo = point.get("customdata", "unknown")
                params = point.get("text", "")
                y_val = point.get("y", "")
                x_val = point.get("x", "")
                # Handle non-numeric values gracefully
                y_str = f"{y_val:.4f}" if isinstance(y_val, (int, float)) else str(y_val)
                x_str = f"{x_val:.4f}" if isinstance(x_val, (int, float)) else str(x_val)
                return f"Combo: {combo} | Params: {params} | Y: {y_str} | X: {x_str}"
            except Exception:
                return "Click a line to reveal the exact combo and values."

        # Start server thread
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        # Wait for server to start, then open browser
        url = f"http://127.0.0.1:{self._port}"
        if self._wait_for_server(url):
            if not self._opened:
                webbrowser.open(url)
                self._opened = True
            logger.info(f"Plotly dashboard running at {url}")
            self._enabled = True
        else:
            logger.warning("Plotly server failed to start in time.")
            self._enabled = False

        return self._enabled

    def _run_server(self):
        try:
            self._app.run(
                debug=False,
                port=self._port,
                host="0.0.0.0",
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Plotly server error: {e}")

    def start_indicator(self, indicator_name: str) -> None:
        """Register start time for an indicator."""
        if not self._ensure_ready():
            return
        with self._lock:
            if indicator_name not in self._start_times:
                self._start_times[indicator_name] = time.time()

    def set_baseline(self, indicator_name: str, baseline_objective: float) -> None:
        """Set baseline objective for an indicator."""
        if not self._ensure_ready():
            return
        with self._lock:
            self._baseline_values.setdefault(indicator_name, {})["objective_best"] = baseline_objective

    def set_baseline_metrics(self, indicator_name: str, metrics: Dict[str, float]) -> None:
        """Set all baseline metrics."""
        if not self._ensure_ready():
            return
        with self._lock:
            self._baseline_values[indicator_name] = metrics.copy()

    def update(
        self,
        indicator_name: str,
        best_objective: float,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        trial_number: int = 0
    ) -> None:
        """Update plot with new best objective."""
        if not self._ensure_ready():
            return
        
        # Filter out rejected trials (-inf from hard cutoffs)
        if not math.isfinite(best_objective):
            return

        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._start_times.get(indicator_name, current_time)

            self._register_line(indicator_name)

            # Update series data
            if indicator_name not in self._series:
                self._series[indicator_name] = {"x": [], "metrics": {}, "params": [], "trials": []}

            series = self._series[indicator_name]
            series["x"].append(elapsed)
            series.setdefault("trials", []).append(trial_number)
            series.setdefault("params", []).append(_format_params(params))

            for key in _METRIC_KEYS:
                val = metrics.get(key)
                series.setdefault("metrics", {}).setdefault(key, []).append(val)

    def record_trial_progress(
        self,
        combo_label: str,
        trial_number: int,
        elapsed_seconds: float,
        metrics: Dict[str, float]
    ) -> None:
        """Track trial progress as a separate '[trials]' series."""
        if not self._ensure_ready():
            return
        
        # Filter out rejected trials (-inf from hard cutoffs)
        obj_best = metrics.get("objective_best")
        if obj_best is not None and not math.isfinite(obj_best):
            return

        with self._lock:
            progress_label = f"{combo_label} [trials]"
            self._register_line(progress_label)

            series = self._series.setdefault(
                progress_label,
                {"x": [], "metrics": {}, "params": [], "trials": []}
            )
            series["x"].append(elapsed_seconds)
            series.setdefault("trials", []).append(trial_number)
            series.setdefault("params", []).append(f"trial={trial_number}")
            for key in _METRIC_KEYS:
                series.setdefault("metrics", {}).setdefault(key, []).append(metrics.get(key))

    def shutdown(self):
        """Gracefully shutdown the plotter."""
        self._enabled = False


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
