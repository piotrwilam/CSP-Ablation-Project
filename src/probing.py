"""
Probing tournament: linear (LogReg) and non-linear (MLP) classifiers
trained per-layer to locate where the model encodes vulnerability info.
"""
import gc
import os
import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .config import (
    LABEL_NAMES,
    LOGREG_MAX_ITER,
    MLP_HIDDEN_LAYERS,
    MLP_MAX_ITER,
    PROBE_RANDOM_STATE,
    PROBE_TEST_SIZE,
)


# ── Per-layer linear sweep ──────────────────────────────────────────────────

def run_linear_sweep(
    all_layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict[int, float]:
    """
    Train a LogisticRegression probe at each layer; return ``{layer: accuracy}``.
    """
    n_layers = len(all_layer_data)
    accs: Dict[int, float] = {}

    print(f"Running linear probe at each of {n_layers} layers...")
    for L in range(n_layers):
        X, y = all_layer_data[L]
        if len(np.unique(y)) < 2:
            accs[L] = 0.5
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE, stratify=y,
        )
        sc = StandardScaler(with_std=False)
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        lr = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=PROBE_RANDOM_STATE)
        lr.fit(X_tr_s, y_tr)
        accs[L] = lr.score(X_te_s, y_te)

    best = max(accs, key=accs.get)
    print(f"Best linear layer: {best} ({accs[best]:.2%})")
    return accs


# ── Per-layer MLP sweep ─────────────────────────────────────────────────────

def run_mlp_sweep(
    all_layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict[int, float]:
    """
    Train an MLP probe at each layer; return ``{layer: accuracy}``.
    """
    n_layers = len(all_layer_data)
    accs: Dict[int, float] = {}

    print(f"MLP tournament at each of {n_layers} layers...")
    for L in range(n_layers):
        X, y = all_layer_data[L]
        if len(np.unique(y)) < 2:
            accs[L] = 0.5
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE, stratify=y,
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            max_iter=MLP_MAX_ITER,
            random_state=PROBE_RANDOM_STATE,
        )
        mlp.fit(X_tr, y_tr)
        accs[L] = mlp.score(X_te, y_te)
        if L % 4 == 0:
            print(f"  Layer {L:02d}: {accs[L]:.1%}")
        del mlp
        gc.collect()

    best = max(accs, key=accs.get)
    print(f"Best MLP layer: {best} ({accs[best]:.2%})")
    return accs


# ── Final probe at chosen layer ─────────────────────────────────────────────

def train_final_probe(
    all_layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    probe_layer: int,
    artifacts_dir: str,
) -> Tuple[LogisticRegression, StandardScaler, float]:
    """
    Train a LogReg probe at *probe_layer*, print a classification report,
    save the probe + scaler + raw arrays to *artifacts_dir*.
    """
    X, y = all_layer_data[probe_layer]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE, stratify=y,
    )
    scaler = StandardScaler(with_std=False)
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    probe = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=PROBE_RANDOM_STATE)
    probe.fit(X_tr_s, y_tr)
    preds = probe.predict(X_te_s)
    acc = accuracy_score(y_te, preds)

    print(f"Final probe at layer {probe_layer}:")
    print(classification_report(y_te, preds, target_names=list(LABEL_NAMES.values())))
    print(f"Accuracy: {acc:.2%}")

    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "code_vuln_probe.pkl"), "wb") as f:
        pickle.dump({"probe": probe, "scaler": scaler, "probe_layer": probe_layer}, f)
    np.save(os.path.join(artifacts_dir, "X_train.npy"), X)
    np.save(os.path.join(artifacts_dir, "y_train.npy"), y)
    print(f"Saved probe → {artifacts_dir}/code_vuln_probe.pkl")

    return probe, scaler, acc


# ── Plotting helpers ────────────────────────────────────────────────────────

def plot_linear_accuracy(
    linear_accs: Dict[int, float],
    artifacts_dir: str,
) -> None:
    """Save and show per-layer linear probe accuracy."""
    layers = sorted(linear_accs)
    best = max(linear_accs, key=linear_accs.get)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, [linear_accs[L] for L in layers], "o-", linewidth=2, markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.axvline(best, color="green", linestyle="--", alpha=0.6, label=f"best={best}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title("Per-layer linear probe (LogReg, 80/20 split)")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()

    os.makedirs(artifacts_dir, exist_ok=True)
    fig.savefig(os.path.join(artifacts_dir, "per_layer_linear_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.show()

    pd.DataFrame(
        [{"layer": L, "linear_accuracy": round(linear_accs[L], 4)} for L in layers]
    ).to_csv(os.path.join(artifacts_dir, "per_layer_linear_accuracy.csv"), index=False)


def plot_linear_vs_mlp(
    linear_accs: Dict[int, float],
    mlp_accs: Dict[int, float],
    artifacts_dir: str,
) -> None:
    """Save and show combined linear / MLP accuracy chart."""
    layers = sorted(linear_accs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, [linear_accs.get(L, 0.5) for L in layers], "o-", label="Linear", color="tab:blue")
    ax.plot(layers, [mlp_accs.get(L, 0.5) for L in layers], "s-", label="MLP", color="tab:orange")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear vs MLP probe accuracy")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()

    os.makedirs(artifacts_dir, exist_ok=True)
    fig.savefig(os.path.join(artifacts_dir, "per_layer_linear_vs_mlp.png"), dpi=150, bbox_inches="tight")
    plt.show()

    pd.DataFrame(
        [{"layer": L, "linear": linear_accs.get(L, 0.5), "mlp": mlp_accs.get(L, 0.5)} for L in layers]
    ).to_csv(os.path.join(artifacts_dir, "per_layer_accuracy_comparison.csv"), index=False)
