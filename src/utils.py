"""
Artifact utilities: Drive + Hugging Face Hub (dual save, load with HF fallback).
"""
import json
import os
import pickle
from typing import Any, Optional

import numpy as np

HF_REPO_ID = "piotrwilam/CSP-Ablation-Project"


def get_artifacts_dir(
    data_dir: str,
    sprint: str = "sprint1",
    version: str = "v1.0",
) -> str:
    """Return artifacts directory for a sprint/version."""
    path = os.path.join(data_dir, "artifacts", sprint, version)
    os.makedirs(path, exist_ok=True)
    return path


def save_to_hub(
    local_path: str,
    repo_path: str,
    repo_id: str = HF_REPO_ID,
) -> bool:
    """
    Upload a file to Hugging Face Hub.
    Returns True on success, False on error (e.g. no auth).
    """
    if not os.path.exists(local_path):
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=repo_id)
        print(f"  Pushed to HF: {repo_path}")
        return True
    except Exception as e:
        print(f"  HF upload skipped ({e})")
        return False


def load_artifact(
    relative_path: str,
    data_dir: str,
    repo_id: str = HF_REPO_ID,
    load_fn=None,
) -> Any:
    """
    Load artifact from Drive first, else HF Hub.
    relative_path: e.g. "artifacts/sprint1/v1.0/code_vuln_probe.pkl"
    load_fn: optional (path) -> obj; defaults to infer from extension.
    """
    local = os.path.join(data_dir, relative_path)
    if os.path.exists(local):
        return _do_load(local, load_fn)

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, filename=relative_path)
        return _do_load(downloaded, load_fn)
    except Exception as e:
        raise FileNotFoundError(f"Artifact not on Drive or HF: {relative_path} — {e}")


def _do_load(path: str, load_fn=None) -> Any:
    if load_fn is not None:
        return load_fn(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    if ext == ".csv":
        return path  # Caller uses pd.read_csv
    raise ValueError(f"No default loader for {ext}")


def save_artifact_and_push(
    obj: Any,
    local_path: str,
    repo_path: str,
    repo_id: str = HF_REPO_ID,
) -> None:
    """
    Save to local path and push to HF.
    obj: for .pkl use dict/sklearn obj; for .npy use ndarray.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    ext = os.path.splitext(local_path)[1].lower()
    if ext == ".pkl":
        with open(local_path, "wb") as f:
            pickle.dump(obj, f)
    elif ext == ".npy":
        np.save(local_path, obj)
    else:
        raise ValueError(f"Save not implemented for {ext}")
    save_to_hub(local_path, repo_path, repo_id)
