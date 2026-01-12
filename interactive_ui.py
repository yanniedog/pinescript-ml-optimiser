"""
UI utilities for the interactive optimizer.
"""

import re
import builtins
from pathlib import Path
from typing import Optional, Callable
from interactive_types import GoBack
from data_manager import DataManager, INTERVAL_NAMES

# Allow overriding input to support go-back shortcuts.
_ORIGINAL_INPUT = builtins.input


def _input_with_go_back(prompt=""):
    response = _ORIGINAL_INPUT(prompt)
    if response is not None and response.strip().lower() in {"b", "back", "go back"}:
        raise GoBack()
    return response


def handle_go_back(message: str = "[INFO] Returning to previous menu.") -> Callable:
    """Decorator to catch GoBack and show a message."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except GoBack:
                print("\n" + message)
        return wrapper
    return decorator


def split_choice_input(raw: str):
    """Split user input by commas or whitespace."""
    if not raw:
        return []
    return [token.strip() for token in re.split(r"[,\s]+", raw.strip()) if token.strip()]


def ask_yes_no(prompt: str, default: Optional[bool] = True) -> bool:
    """Prompt for a yes/no answer, insisting on validity before proceeding."""
    while True:
        user_input = input(prompt).strip().lower()
        if not user_input:
            if default is not None:
                return default
            print("  [ERROR] Please enter Y or N.")
            continue
        if user_input in {"y", "yes"}:
            return True
        if user_input in {"n", "no"}:
            return False
        print("  [ERROR] Please enter Y or N.")


def get_pine_files(directory: Path = None):
    """Get all .pine files in the directory, excluding optimized ones."""
    current_dir = directory or Path.cwd()
    pine_files = list(current_dir.glob("*.pine"))
    pine_files = [f for f in pine_files if not f.name.startswith('optimised_')]
    return sorted(pine_files)


def display_pine_menu(pine_files):
    """Display numbered menu of available Pine Script files."""
    if not pine_files:
        print("\n[ERROR] No Pine Script files found in the current directory.")
        print("   Please ensure you have .pine files in:", Path.cwd())
        return None
    
    print("\n" + "="*70)
    print("  Available Pine Script Indicators")
    print("="*70)
    
    for i, file in enumerate(pine_files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"  [{i}] {file.name:<40} ({size_kb:.1f} KB)")
    
    print("="*70)
    return pine_files


def get_user_choice(pine_files):
    """Get user's file selection."""
    while True:
        try:
            choice = input(f"\nSelect indicator to optimize (1-{len(pine_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(pine_files):
                return pine_files[choice_num - 1]
            else:
                print(f"[ERROR] Please enter a number between 1 and {len(pine_files)}")
        except ValueError:
            print("[ERROR] Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n")
            return None


def display_data_status(dm: DataManager, interval: str = None):
    """Display current data status."""
    datasets = dm.get_available_datasets()
    
    print("\n" + "-"*70)
    print("  Current Data Status")
    print("-"*70)
    
    if not datasets:
        print("  No datasets found. Use 'Download new data' to get started.")
        return
    
    # Group by interval
    by_interval = {}
    for symbol, intv in datasets:
        if intv not in by_interval:
            by_interval[intv] = []
        by_interval[intv].append(symbol)
    
    for intv in sorted(by_interval.keys()):
        symbols = by_interval[intv]
        interval_name = INTERVAL_NAMES.get(intv, intv)
        marker = " <-- selected" if intv == interval else ""
        print(f"\n  {interval_name} ({intv}): {len(symbols)} symbols{marker}")
        display_symbols = symbols[:8]
        if len(symbols) > 8:
            print(f"    {', '.join(display_symbols)}, ... (+{len(symbols)-8} more)")
        else:
            print(f"    {', '.join(display_symbols)}")


def find_pine_directories(root: Path):
    """Return directories under root that contain .pine files with counts."""
    counts = {}
    for pine_file in root.rglob("*.pine"):
        if pine_file.is_file():
            parent = pine_file.parent
            counts[parent] = counts.get(parent, 0) + 1
    return sorted(counts.items(), key=lambda item: str(item[0]))


def display_indicator_catalog(pine_files, directory: Path = None, limit: int = 30):
    """Print a numbered list of indicators, optionally tied to a directory."""
    heading = f"Indicators in {directory}" if directory else "Indicators"
    print(f"\n{heading} ({len(pine_files)} total):")
    for idx, pine_file in enumerate(pine_files[:limit], 1):
        print(f"  [{idx}] {pine_file.name}")
    if len(pine_files) > limit:
        print(f"  ... and {len(pine_files) - limit} more")


def choose_indicator_directory() -> Path:
    """Prompt for indicator directory, defaulting to ./pinescripts if present."""
    default_dir = Path.cwd() / "pinescripts"
    if default_dir.exists():
        default_value = default_dir
    else:
        default_value = Path.cwd()

    available_dirs = find_pine_directories(Path.cwd())
    counts_map = {path: count for path, count in available_dirs}
    default_count = counts_map.get(default_value)
    if default_count is None:
        if default_value.exists():
            default_count = sum(1 for _ in default_value.rglob("*.pine"))
        else:
            default_count = 0
    directory_choices = [(default_value, default_count)]
    for path, count in available_dirs:
        if path == default_value:
            continue
        directory_choices.append((path, count))

    print("\n  Detected Pine Script directories:")
    for idx, (path, count) in enumerate(directory_choices, 1):
        print(f"    [{idx}] {path} ({count} .pine files)")
    print("    [C] Custom directory path")

    prompt = f"  Select directory (1-{len(directory_choices)}) or enter path [default: {default_value}]: "
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            return default_value
        if user_input.lower() in {"c", "custom"}:
            custom_input = input("  Enter custom directory path: ").strip()
            candidate = Path(custom_input)
        else:
            candidate = None
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(directory_choices):
                    return directory_choices[idx][0]
            candidate = Path(user_input)
        if candidate.exists() and candidate.is_dir():
            return candidate
        print("  [ERROR] Directory not found. Please enter a valid path.")
