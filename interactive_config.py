"""
Configuration management for the interactive optimizer.
"""

import os
from interactive_ui import handle_go_back

TRIAL_CONTROL_OPTIONS = {
    "max_trials": None,
    "min_runtime_seconds": None,
    "stall_seconds": None,
    "improvement_rate_floor": None,
    "n_jobs": None,
    "fast_evaluation": None,
}


def _get_trial_overrides():
    """Return the currently configured trial controls."""
    return {k: v for k, v in TRIAL_CONTROL_OPTIONS.items() if v is not None}


def apply_trial_overrides(target: dict):
    """Apply trial control defaults to a dictionary (preserving existing entries)."""
    for key, value in _get_trial_overrides().items():
        if key not in target or target[key] is None:
            target[key] = value


def _prompt_trial_setting(prompt: str, current, cast, validator, invalid_msg: str):
    """Helper to prompt for trial control values (int/float)."""
    while True:
        default = f"[{current}]" if current is not None else "[none]"
        entry = input(f"  {prompt} {default} (enter 'clear' to reset, blank to keep): ").strip()
        if not entry:
            return current
        if entry.lower() in {"clear", "none"}:
            return None
        try:
            value = cast(entry)
        except ValueError:
            print(f"  [ERROR] Invalid number: '{entry}'")
            continue
        if not validator(value):
            print(f"  [ERROR] {invalid_msg}")
            continue
        return value


@handle_go_back("[INFO] Returning to main menu.")
def configure_trial_controls():
    """Interactive menu to configure max/min/stall trial settings."""
    print("\n  Enter 'B' at any prompt to return to the previous menu.")
    print("\n" + "=" * 70)
    print("  Trial Control Settings")
    print("=" * 70)
    print(
        "  These settings are applied to subsequent optimizations "
        "and persist until cleared."
    )
    print("  Set 'max trials' to cap the search, 'min runtime' to prevent early exits,")
    print("  'stall timeout' to delay the no-improvement guard, and 'improvement floor'")
    print("  to tune how sensitive the optimizer is to slowing progress.")
    print("  Set 'n_jobs' for parallel trial execution (default: auto, min(4, cpu_count)).")
    print("  Set 'fast_evaluation' to use reduced forecast horizons for faster optimization.")

    TRIAL_CONTROL_OPTIONS["max_trials"] = _prompt_trial_setting(
        "Max trials",
        TRIAL_CONTROL_OPTIONS["max_trials"],
        int,
        lambda v: v > 0,
        "Enter a positive integer."
    )
    TRIAL_CONTROL_OPTIONS["min_runtime_seconds"] = _prompt_trial_setting(
        "Minimum runtime (seconds) before early-stop checks",
        TRIAL_CONTROL_OPTIONS["min_runtime_seconds"],
        int,
        lambda v: v >= 0,
        "Enter 0 or a positive number."
    )
    TRIAL_CONTROL_OPTIONS["stall_seconds"] = _prompt_trial_setting(
        "Stall timeout (seconds, 0 disables)",
        TRIAL_CONTROL_OPTIONS["stall_seconds"],
        int,
        lambda v: v >= 0,
        "Enter 0 or a positive number."
    )
    TRIAL_CONTROL_OPTIONS["improvement_rate_floor"] = _prompt_trial_setting(
        "Improvement rate floor (%/s)",
        TRIAL_CONTROL_OPTIONS["improvement_rate_floor"],
        float,
        lambda v: True,
        "Enter a number."
    )
    
    # n_jobs setting
    default_n_jobs = min(4, os.cpu_count() or 4)
    while True:
        default_str = f"[{default_n_jobs}]" if TRIAL_CONTROL_OPTIONS["n_jobs"] is None else f"[{TRIAL_CONTROL_OPTIONS['n_jobs']}]"
        entry = input(f"  Parallel jobs (n_jobs) {default_str} (enter 'clear' to reset, blank to keep): ").strip()
        if not entry:
            break
        if entry.lower() in {"clear", "none"}:
            TRIAL_CONTROL_OPTIONS["n_jobs"] = None
            break
        try:
            value = int(entry)
            if value > 0:
                TRIAL_CONTROL_OPTIONS["n_jobs"] = value
                break
            else:
                print("  [ERROR] Enter a positive integer.")
        except ValueError:
            print(f"  [ERROR] Invalid number: '{entry}'")
    
    # fast_evaluation setting
    while True:
        current = TRIAL_CONTROL_OPTIONS["fast_evaluation"]
        default_str = "[False]" if current is None else f"[{current}]"
        entry = input(f"  Fast evaluation (reduced horizons) {default_str} (y/n, 'clear' to reset, blank to keep): ").strip().lower()
        if not entry:
            break
        if entry in {"clear", "none"}:
            TRIAL_CONTROL_OPTIONS["fast_evaluation"] = None
            break
        if entry in {"y", "yes", "true", "1"}:
            TRIAL_CONTROL_OPTIONS["fast_evaluation"] = True
            break
        if entry in {"n", "no", "false", "0"}:
            TRIAL_CONTROL_OPTIONS["fast_evaluation"] = False
            break
        print("  [ERROR] Please enter Y or N.")

    print("\n  Trial controls updated:")
    for label, key in [
        ("Max trials", "max_trials"),
        ("Min runtime (sec)", "min_runtime_seconds"),
        ("Stall timeout (sec)", "stall_seconds"),
        ("Improvement rate floor", "improvement_rate_floor"),
        ("Parallel jobs (n_jobs)", "n_jobs"),
        ("Fast evaluation", "fast_evaluation"),
    ]:
        value = TRIAL_CONTROL_OPTIONS[key]
        if key == "fast_evaluation" and value is not None:
            print(f"    {label}: {value}")
        else:
            print(f"    {label}: {value if value is not None else 'none'}")
