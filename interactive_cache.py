"""
Cache management for the interactive optimizer.
Supports reading from both zip file backups (new format) and directory backups (legacy format).
"""

import datetime
import json
import shutil
import zipfile
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


def _load_matrix_rows_from_zip(zip_path: Path):
    """Load cached matrix rows from a backup zip file."""
    summary_internal_path = "optimized_outputs/summary/unified_optimization_matrix.json"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if summary_internal_path not in zf.namelist():
                return {}
            payload = json.loads(zf.read(summary_internal_path).decode("utf-8"))
    except Exception:
        return {}
    
    matrix = payload.get("matrix") or []
    entries = {}
    for row in matrix:
        key = _combo_key_from_row(row)
        if not all(key):
            continue
        # Use the zip path as the source (not a directory)
        entries[key] = (row, zip_path)
    return entries


def _collect_cached_combos():
    """Collect previously run combinations from current outputs and backups (both zip and directory)."""
    combos = {}
    summary_dir = Path("optimized_outputs") / "summary"
    combos.update(_load_matrix_rows(summary_dir / "unified_optimization_matrix.json", Path.cwd()))

    backup_root = Path("backup")
    if not backup_root.exists():
        return combos
    
    for item in sorted(backup_root.iterdir()):
        # Handle zip file backups (new format)
        if item.is_file() and item.suffix == ".zip":
            combos.update(_load_matrix_rows_from_zip(item))
        # Handle directory backups (legacy format)
        elif item.is_dir():
            summary_path = item / "optimized_outputs" / "summary" / "unified_optimization_matrix.json"
            combos.update(_load_matrix_rows(summary_path, item))
    
    return combos


def _restore_cached_outputs(row: dict, source_root: Path) -> bool:
    """Copy cached output files from a known source to the current workspace.
    
    source_root can be either:
    - A directory path (legacy format or current workspace)
    - A zip file path (new format)
    """
    success = True
    
    # Check if source is a zip file
    is_zip_source = source_root.is_file() and source_root.suffix == ".zip"
    
    for key in ("output_pine", "output_report"):
        rel_path = row.get(key)
        if not rel_path:
            success = False
            continue
        
        dest_path = Path(rel_path)
        
        if is_zip_source:
            # Extract from zip file
            try:
                with zipfile.ZipFile(source_root, 'r') as zf:
                    if rel_path not in zf.namelist():
                        print(f"[CACHE] Missing cached file for {rel_path} in {source_root}")
                        success = False
                        continue
                    
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check if we need to update
                    if dest_path.exists():
                        # Get zip file info for comparison
                        zip_info = zf.getinfo(rel_path)
                        dest_stat = dest_path.stat()
                        # ZipInfo date_time is a tuple (year, month, day, hour, min, sec)
                        zip_mtime = datetime.datetime(*zip_info.date_time).timestamp()
                        if dest_stat.st_mtime >= zip_mtime:
                            continue
                    
                    # Extract the file
                    with zf.open(rel_path) as src_file:
                        dest_path.write_bytes(src_file.read())
                        
            except Exception as exc:
                print(f"[CACHE] Failed to restore {rel_path} from zip: {exc}")
                success = False
        else:
            # Copy from directory (legacy format or current workspace)
            src_path = (source_root / rel_path).resolve()
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
