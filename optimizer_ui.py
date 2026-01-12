"""
UI components for the optimizer, including keyboard monitoring and realtime plotting.
"""

import sys
import time
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
    _format_param_value
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
        x_vals = series.get("x", [])
        y_vals = series.get("metrics", {}).get(self._y_metric, [])
        if self._x_mode == "rate":
            baseline = self._baseline_values.get(label, {}).get(self._y_metric)
            x_vals = _compute_rate_series(x_vals, y_vals, baseline)
        elif self._x_mode == "trial":
            x_vals = series.get("trials", [])
        elif self._x_mode == "trials_per_second":
            x_vals = series.get("metrics", {}).get("trials_per_second", [])
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
            
        # Limit points to avoid slowdown
        if len(series["x"]) > self._max_points:
            series["x"] = series["x"][-self._max_points:]
            series["trials"] = series["trials"][-self._max_points:]
            series["params"] = series["params"][-self._max_points:]
            for key in _METRIC_KEYS:
                series["metrics"][key] = series["metrics"][key][-self._max_points:]

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
            
        if len(series["x"]) > self._max_points:
            series["x"] = series["x"][-self._max_points:]
            series["trials"] = series["trials"][-self._max_points:]
            series["params"] = series["params"][-self._max_points:]
            for key in _METRIC_KEYS:
                series["metrics"][key] = series["metrics"][key][-self._max_points:]

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


class PlotlyRealtimePlotter(RealtimeBestPlotter):
    """
    Realtime plotter using Plotly (browser-based).
    Starts a local server and updates a dashboard.
    Optimized for Windows Surface with proper threading.
    """
    def __init__(self):
        super().__init__()
        self._server_thread = None
        self._port = 8050
        self._app = None
        self._data_lock = threading.RLock()  # Use RLock for Windows compatibility
        self._pending_updates = []
        self._host = "127.0.0.1"
        self._url = f"http://{self._host}:{self._port}"
        self._shutdown_event = threading.Event()
        self._max_data_points = 500  # Limit data for memory efficiency

    def _ensure_ready(self) -> bool:
        if self._init_attempted:
            return self._enabled
        self._init_attempted = True
        
        try:
            import dash
            from dash import dcc, html
            from dash.dependencies import Input, Output
            import plotly.graph_objs as go
        except ImportError as exc:
            logger.info("Plotly plotter disabled (dash/plotly not installed): %s", exc)
            self._enabled = False
            return False

        try:
            # Find free port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self._host, 0))
            self._port = sock.getsockname()[1]
            sock.close()
            self._url = f"http://{self._host}:{self._port}"

            self._app = dash.Dash(__name__, update_title=None)
            self._app.layout = html.Div([
                html.H3("Pine Script Optimization Progress"),
                html.Div([
                    html.Label("Metric:"),
                    dcc.Dropdown(
                        id='metric-select',
                        options=[{'label': _metric_label(k), 'value': k} for k in _METRIC_KEYS],
                        value='objective_delta',
                        clearable=False
                    ),
                ], style={'width': '300px', 'display': 'inline-block'}),
                dcc.Graph(id='live-graph', animate=False),
                dcc.Interval(
                    id='graph-update',
                    interval=2000, # 2s update
                    n_intervals=0
                )
            ])

            @self._app.callback(
                Output('live-graph', 'figure'),
                [Input('graph-update', 'n_intervals'),
                 Input('metric-select', 'value')]
            )
            def update_graph_scatter(n, metric_key):
                with self._data_lock:
                    traces = []
                    for label, series in self._series.items():
                        if "[trials]" in label: 
                            continue # Skip high-freq trial data for now
                        
                        y_data = series.get("metrics", {}).get(metric_key, [])
                        if not y_data:
                            continue
                            
                        x_data = series.get("x", [])
                        
                        traces.append(go.Scatter(
                            x=list(x_data),
                            y=list(y_data),
                            name=label,
                            mode='lines+markers',
                            hovertext=[str(p) for p in series.get("params", [])]
                        ))
                    
                    layout = go.Layout(
                        xaxis={'title': 'Elapsed Time (s)'},
                        yaxis={'title': _metric_label(metric_key)},
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                        hovermode='closest',
                        showlegend=True
                    )
                    return {'data': traces, 'layout': layout}

            # Run server in thread
            logger.info(f"Starting Plotly dashboard at {self._url}")
            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._server_thread.start()
            
            # Open browser
            webbrowser.open(self._url)
            
            self._enabled = True
            return True
            
        except Exception as exc:
            logger.info("Plotly plotter disabled (init error): %s", exc)
            self._enabled = False
            return False

    def _run_server(self):
        # Suppress Flask banner and configure for Windows
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        try:
            # Use threaded=True for better Windows compatibility
            self._app.run_server(
                debug=False, 
                port=self._port, 
                host=self._host, 
                use_reloader=False,
                threaded=True  # Better for Windows
            )
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"Plotly server error: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the plotter."""
        self._shutdown_event.set()
        self._enabled = False

    # Override update methods to use lock with data limiting
    def update(self, indicator_name, best_objective, metrics, params, trial_number=0):
        with self._data_lock:
            super().update(indicator_name, best_objective, metrics, params, trial_number)
            # Limit data points per series for memory efficiency
            self._trim_series_data()

    def record_trial_progress(self, combo_label, trial_number, elapsed_seconds, metrics):
        with self._data_lock:
            super().record_trial_progress(combo_label, trial_number, elapsed_seconds, metrics)
            self._trim_series_data()
    
    def _trim_series_data(self):
        """Trim series data to prevent memory bloat on long runs."""
        for label, series in self._series.items():
            if len(series.get("x", [])) > self._max_data_points:
                # Keep only the most recent data points
                series["x"] = series["x"][-self._max_data_points:]
                if "trials" in series:
                    series["trials"] = series["trials"][-self._max_data_points:]
                if "params" in series:
                    series["params"] = series["params"][-self._max_data_points:]
                for key in series.get("metrics", {}):
                    series["metrics"][key] = series["metrics"][key][-self._max_data_points:]


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
