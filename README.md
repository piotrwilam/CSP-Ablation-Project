# CSP-Ablation-Project

Mechanistic interpretability of code vulnerability detection in OpenAI's **circuit-sparsity** (CSP) model.

## Overview

This project investigates how a weight-sparse transformer (419M params, 8 layers) internally represents the distinction between *secure* and *insecure* Python code. The work is organized in three sprints with different minimal-pair datasets. **Sprint 1** (Phase 1 + Phase 2) is complete.

| Phase | Notebook | Description |
|-------|----------|-------------|
| **1 — Finding Ethical Circuits** | `notebooks/phase1_probing.ipynb` | Probing tournament (linear + MLP) across all layers to locate where security information is encoded |
| **2 — Ablation** | `notebooks/phase2_ablation.ipynb` | T1–T5: feature ranking, activation profiling, ablation engine, threshold sweep, generation sanity check |

## Project Structure

```
CSP-Ablation-Project/
├── README.md
├── EXPERIMENT_LOG.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── sprint1/
│   │   └── minimal_pairs_code.json
│   ├── sprint2/                  # placeholder for Sprint 2
│   ├── sprint3/                  # placeholder for Sprint 3
│   └── dataset_manifest.csv      # registry of datasets
├── src/
│   ├── config.py                 # paths, HF_REPO_ID, artifacts_dir()
│   ├── data_loader.py            # load minimal-pairs
│   ├── data_utils.py             # get_dataset_path(), manifest
│   ├── model_loader.py            # load CSP model + tokenizer
│   ├── hidden_states.py          # hook-based hidden-state extraction
│   ├── probing.py                # linear / MLP sweeps, final probe, plots
│   ├── ablation.py               # CircuitAblator (zero + mean ablation)
│   └── utils.py                  # save_to_hub(), load_artifact()
├── notebooks/
│   ├── phase1_probing.ipynb      # Phase 1 — probing tournament
│   └── phase2_ablation.ipynb     # Phase 2 — T1–T5 feature analysis & ablation
└── experiments/                  # timestamped run logs
```

## Google Drive & Hugging Face

**Drive layout** (Colab):
```
My Drive/
├── CODE/
│   └── CSP-Ablation-Project/     # git clone
└── DATA/
    └── CSP-Ablation-Project/
        └── artifacts/
            ├── sprint1/
            │   └── v1.0/         # probes, CSVs, plots (versioned)
            └── ...
```

**Hugging Face**: [piotrwilam/CSP-Ablation-Project](https://huggingface.co/piotrwilam/CSP-Ablation-Project)  
Artifacts are auto-pushed after each run (Drive + HF). Load: Drive first, HF fallback.  
Sprint 1 artifacts: [artifacts/sprint1/v1.0](https://huggingface.co/piotrwilam/CSP-Ablation-Project/tree/main/artifacts/sprint1/v1.0)

## Setup

1. **Colab:** Open `notebooks/phase1_probing.ipynb` in Google Colab.
2. Set runtime to **GPU** (T4 or better).
3. Run the install cell once, restart the runtime, then run all remaining cells.

## Key Results — Sprint 1 Complete

**Phase 1 (Probing)**
| Metric | Value |
|--------|-------|
| Best linear probe layer | 4 (87.0%) |
| Best MLP probe layer | 6 (84.4%) |
| Dataset | 192 minimal pairs → 384 examples |

**Phase 2 (Ablation)**
| Metric | Value |
|--------|-------|
| Baseline probe accuracy | 86.7% |
| Accuracy at k=20 ablation | ~50% (chance) |
| Signal concentration | Compact — top ~20 features carry most signal |
| Artifacts | [sprint1/v1.0 on HF](https://huggingface.co/piotrwilam/CSP-Ablation-Project/tree/main/artifacts/sprint1/v1.0) |

## Dataset

`minimal_pairs_code.json` — pairs of Python code snippets where each pair contains a *clean* (secure) and *corrupted* (insecure) variant. Keys: `clean`, `corrupted`, `clean_label` (0), `corrupted_label` (1).

## Model

[openai/circuit-sparsity](https://huggingface.co/openai/circuit-sparsity) — a weight-sparse GPT-style transformer with 419M parameters and 8 transformer layers.
