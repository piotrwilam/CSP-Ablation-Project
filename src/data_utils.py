"""
Dataset utilities: resolve paths from manifest, load by sprint/dataset.
"""
import os
from typing import Optional

import pandas as pd


def _get_data_dir(code_dir: str) -> str:
    """Data directory (inside repo)."""
    return os.path.join(code_dir, "data")


def load_dataset_manifest(code_dir: str) -> pd.DataFrame:
    """Load dataset_manifest.csv."""
    data_dir = _get_data_dir(code_dir)
    manifest_path = os.path.join(data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        return pd.DataFrame(columns=["dataset_id", "sprint", "path", "n_pairs", "description", "created"])
    return pd.read_csv(manifest_path)


def get_dataset_path(
    sprint_id: str,
    code_dir: str,
    data_dir_drive: Optional[str] = None,
) -> str:
    """
    Resolve path to minimal_pairs JSON for a sprint.

    Tries: manifest path (code_dir/data/{path}), then sprint folder, then legacy locations.
    """
    data_dir = _get_data_dir(code_dir)
    manifest = load_dataset_manifest(code_dir)
    row = manifest[manifest["sprint"] == sprint_id]
    if not row.empty and pd.notna(row.iloc[0].get("path")) and str(row.iloc[0]["path"]).strip():
        path = os.path.join(data_dir, str(row.iloc[0]["path"]).strip())
        if os.path.exists(path):
            return path

    # Fallback: sprint folder
    fallback = os.path.join(data_dir, sprint_id, "minimal_pairs_code.json")
    if os.path.exists(fallback):
        return fallback

    # Legacy fallbacks
    legacy = os.path.join(data_dir, "minimal_pairs_code.json")
    if os.path.exists(legacy):
        return legacy
    if data_dir_drive:
        drive_path = os.path.join(data_dir_drive, "minimal_pairs_code.json")
        if os.path.exists(drive_path):
            return drive_path

    raise FileNotFoundError(
        f"Dataset for sprint '{sprint_id}' not found. "
        f"Expected: {fallback} or {legacy}"
    )
