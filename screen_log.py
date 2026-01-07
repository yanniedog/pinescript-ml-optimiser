import sys
import logging
from datetime import datetime
from pathlib import Path

_SCREEN_LOG_ENABLED = False
_SCREEN_LOG_PATH = None
_LOG_FILE = None


class TeeWriter:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        if not data:
            return 0
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


def enable_screen_log(log_dir: str = "logs", prefix: str = "screen") -> str:
    """Mirror stdout/stderr to a timestamped log file (idempotent)."""
    global _SCREEN_LOG_ENABLED, _SCREEN_LOG_PATH, _LOG_FILE

    if _SCREEN_LOG_ENABLED:
        return _SCREEN_LOG_PATH

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir_path / f"{prefix}_{timestamp}.log"

    _LOG_FILE = log_path.open("a", encoding="utf-8")
    sys.stdout = TeeWriter(sys.stdout, _LOG_FILE)
    sys.stderr = TeeWriter(sys.stderr, _LOG_FILE)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = sys.stderr

    _SCREEN_LOG_ENABLED = True
    _SCREEN_LOG_PATH = str(log_path)
    print(f"[LOG] Screen output is being captured to: {log_path}")
    return _SCREEN_LOG_PATH
