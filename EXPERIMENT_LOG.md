# Experiment Log

## Phase 1 — Finding Ethical Circuits

### Experiment 1.1: Probing Tournament (NLP04_1)

**Date:** 2026-02-17  
**Notebook:** `notebooks/phase1_probing.ipynb` (refactored from `NLP04_1.ipynb`)  
**Runtime:** Colab T4 GPU  

**Setup:**
- Model: `openai/circuit-sparsity` (419M params, 8 layers)
- Dataset: 192 minimal pairs → 384 probe examples
- Probes: LogisticRegression (linear) and MLP (128, 64) per layer
- Split: 80/20 stratified, seed 42

**Results:**

| Layer | Linear Accuracy | MLP Accuracy |
|-------|----------------|-------------|
| 0     | —              | 72.7%       |
| 4     | **87.0%**      | 79.2%       |
| 6     | —              | **84.4%**   |

- Best linear readout: **layer 4** (87.0%)
- Best MLP readout: **layer 6** (84.4%)
- The linear probe outperforms the MLP at layer 4, suggesting the security signal is largely linearly separable in the mid-network residual stream.

**Artifacts saved:**
- `code_vuln_probe.pkl` — best linear probe + scaler
- `X_train.npy`, `y_train.npy` — full hidden-state arrays at probe layer
- `per_layer_linear_accuracy.csv`, `per_layer_accuracy_comparison.csv`
- `per_layer_linear_accuracy.png`, `per_layer_linear_vs_mlp.png`
- `analysis_results.json`

---

## Phase 2 — Ablation

### T1–T5 Implementation (Sprint 1)

**Notebook:** `notebooks/phase2_ablation.ipynb`  
**Prerequisite:** Phase 1 artifacts in `DATA/CSP-Ablation-Project/artifacts/`

**Pipeline:**
- **T1** Probe weight extraction & feature ranking → `top_features.json`, weight histogram
- **T2** Feature activation profiling → per-feature AUC, overlapping histograms for top 5
- **T3** Ablation engine (`src/ablation.py`) — zero + mean strategies, tested on 3–5 examples
- **T4** Ablation threshold sweep (k=1,3,5,10,20,50,100,200) → ablation curve, `ablation_sweep_results.json`
- **T5** Generation sanity check — 40 prompts (20 insecure, 20 secure), original vs ablated output comparison

**Artifacts:**
- `top_features.json`, `weight_distribution_histogram.png`
- `per_feature_auc.csv`, `activation_histograms_top5.png`
- `ablation_sweep_results.json`, `ablation_curve.png`
- `generation_comparison.csv`
