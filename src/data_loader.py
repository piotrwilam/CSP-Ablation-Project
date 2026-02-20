"""
Load and pre-process the minimal-pairs dataset for probing.
"""
import json
import os
from typing import Dict, List

from .config import LABEL_NAMES


def prompt_from_scenario(code_str: str) -> str:
    """
    Append a uniform suffix so the model gathers context into the same
    final token for every example, preventing tokenisation leakage.
    """
    return f"{code_str.strip()}\n# Security review:"


def load_minimal_pairs(pairs_path: str) -> List[Dict]:
    """
    Load ``minimal_pairs_code.json`` and return a flat list of
    ``{"code": ..., "label": 0|1}`` dicts (two per pair).
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"minimal_pairs_code.json not found at {pairs_path}.\n"
            "Create it with keys: clean, corrupted, clean_label, corrupted_label."
        )

    with open(pairs_path) as f:
        minimal_pairs = json.load(f)

    required = {"clean", "corrupted", "clean_label", "corrupted_label"}
    if not minimal_pairs or not required.issubset(minimal_pairs[0].keys()):
        raise ValueError(f"minimal_pairs_code.json must have keys {required}")

    probe_examples: List[Dict] = []
    for p in minimal_pairs:
        probe_examples.append({"code": p["clean"], "label": p["clean_label"]})
        probe_examples.append({"code": p["corrupted"], "label": p["corrupted_label"]})

    n_pairs = len(minimal_pairs)
    n_examples = len(probe_examples)
    print(f"Loaded {n_pairs} pairs → {n_examples} probe examples")
    if n_examples < 100:
        print(
            f"  WARNING: only {n_examples} examples — probe results will be noisy. "
            "200+ recommended."
        )

    return probe_examples
