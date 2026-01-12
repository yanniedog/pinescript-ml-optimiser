"""
Cache management for the interactive optimizer.
"""

import json
import shutil
from pathlib import Path


def _combo_key_from_row(row: dict):
    """Build a unique key for an indicator-symbol-interval combination."""
    return (row.get("file_name"), row.get("symbol"), row.get("interval"))


def _load_matrix_rows(summary_path: Path, source_root: Path):
    """Load cached matrix rows from a summary JSON and associate them with the source."""
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    matrix = payload.get("matrix") or []
    entries = {}
    for row in matrix:
        key = _combo_key_from_row(row)
        if not all(key):
            continue
        entries[key] = (row, source_root)
    return entries


def _collect_cached_combos():
    """Collect previously run combinations from current outputs and backups."""
    combos = {}
    summary_dir = Path("optimized_outputs") / "summary"
    combos.update(_load_matrix_rows(summary_dir / "unified_optimization_matrix.json", Path.cwd()))

    backup_root = Path("backup")
    if not backup_root.exists():
        return combos
    for backup_dir in sorted(p for p in backup_root.iterdir() if p.is_dir()):
        summary_path = backup_dir / "optimized_outputs" / "summary" / "unified_optimization_matrix.json"
        combos.update(_load_matrix_rows(summary_path, backup_dir))
    return combos


def _restore_cached_outputs(row: dict, source_root: Path) -> bool:
    """Copy cached output files from a known source to the current workspace."""
    success = True
    for key in ("output_pine", "output_report"):
        rel_path = row.get(key)
        if not rel_path:
            success = False
            continue
        src_path = (source_root / rel_path).resolve()
        dest_path = Path(rel_path)
        if not src_path.exists():
            print(f"[CACHE] Missing cached file for {rel_path} at {src_path}")
            success = False
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest_path.exists():
                src_stat = src_path.stat()
                dest_stat = dest_path.stat()
                if dest_stat.st_mtime >= src_stat.st_mtime:
                    continue
            shutil.copy2(src_path, dest_path)
        except Exception as exc:
            print(f"[CACHE] Failed to restore {rel_path}: {exc}")
            success = False
    return success
