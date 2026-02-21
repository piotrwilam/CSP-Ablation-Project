"""
Project-wide configuration: model IDs, paths, constants.
"""
import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "openai/circuit-sparsity"
MAX_SEQ_LENGTH = 512

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "SECURE", 1: "INSECURE"}

# ── Hugging Face ─────────────────────────────────────────────────────────────
HF_REPO_ID = "piotrwilam/CSP-Ablation-Project"

# ── Google Drive paths (Colab) ───────────────────────────────────────────────
# CODE lives inside the cloned repo; DATA lives in a sibling directory on Drive.
DRIVE_MOUNT = "/content/drive"
DRIVE_ROOT = os.path.join(DRIVE_MOUNT, "MyDrive")
CODE_DIR = os.path.join(DRIVE_ROOT, "CODE", "CSP-Ablation-Project")
DATA_DIR = os.path.join(DRIVE_ROOT, "DATA", "CSP-Ablation-Project")


def artifacts_dir(sprint: str = "sprint1", version: str = "v1.0") -> str:
    """Artifacts path for sprint/version: DATA/artifacts/{sprint}/{version}/"""
    path = os.path.join(DATA_DIR, "artifacts", sprint, version)
    os.makedirs(path, exist_ok=True)
    return path


# Legacy: flat artifacts
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

# ── Probing defaults ────────────────────────────────────────────────────────
PROBE_TEST_SIZE = 0.2
PROBE_RANDOM_STATE = 42
MAX_PROBE_SAMPLES = 400
MLP_HIDDEN_LAYERS = (128, 64)
MLP_MAX_ITER = 500
LOGREG_MAX_ITER = 500
