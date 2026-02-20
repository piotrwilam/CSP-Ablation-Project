# CSP-Ablation-Project

Mechanistic interpretability of code vulnerability detection in OpenAI's **circuit-sparsity** (CSP) model.

## Overview

This project investigates how a weight-sparse transformer (419M params, 8 layers) internally represents the distinction between *secure* and *insecure* Python code. The work is organized in two phases:

| Phase | Notebook | Description |
|-------|----------|-------------|
| **1 — Finding Ethical Circuits** | `notebooks/phase1_probing.ipynb` | Probing tournament (linear + MLP) across all layers to locate where security information is encoded |
| **2 — Ablation** | `notebooks/phase2_ablation.ipynb` | Targeted ablation experiments on the identified circuits *(planned)* |

## Project Structure

```
CSP-Ablation-Project/
├── README.md
├── EXPERIMENT_LOG.md
├── requirements.txt
├── .gitignore
├── data/
│   └── minimal_pairs_code.json   # not tracked — add locally or via Drive
├── src/
│   ├── __init__.py
│   ├── config.py                 # paths, model ID, constants
│   ├── model_loader.py           # load CSP model + tokenizer
│   ├── data_loader.py            # load & flatten minimal-pairs dataset
│   ├── hidden_states.py          # hook-based hidden-state extraction
│   └── probing.py                # linear / MLP sweeps, final probe, plots
└── notebooks/
    ├── phase1_probing.ipynb      # Phase 1 — probing tournament
    └── phase2_ablation.ipynb     # Phase 2 — ablation (placeholder)
```

## Google Drive Layout

When running in Colab, the project expects this Drive structure:

```
My Drive/
├── CODE/
│   └── CSP-Ablation-Project/     # git clone of this repo
└── DATA/
    └── CSP-Ablation-Project/
        ├── minimal_pairs_code.json  # (fallback location)
        └── artifacts/               # probes, plots, CSVs
```

## Setup

1. **Colab:** Open `notebooks/phase1_probing.ipynb` in Google Colab.
2. Set runtime to **GPU** (T4 or better).
3. Run the install cell once, restart the runtime, then run all remaining cells.

## Key Results (Phase 1)

| Metric | Value |
|--------|-------|
| Best linear probe layer | 4 (87.0%) |
| Best MLP probe layer | 6 (84.4%) |
| Dataset | 192 minimal pairs → 384 examples |

## Dataset

`minimal_pairs_code.json` — pairs of Python code snippets where each pair contains a *clean* (secure) and *corrupted* (insecure) variant. Keys: `clean`, `corrupted`, `clean_label` (0), `corrupted_label` (1).

## Model

[openai/circuit-sparsity](https://huggingface.co/openai/circuit-sparsity) — a weight-sparse GPT-style transformer with 419M parameters and 8 transformer layers.
