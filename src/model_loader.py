"""
Load the OpenAI circuit-sparsity model and tokenizer.

The HF repo's config.py does `from circuit_sparsity.gpt import GPTConfig`,
so we first build a tiny package at /tmp, then monkey-patch GPTConfig to
silently drop the extra kwargs that its dataclass doesn't define.
"""
import dataclasses
import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID


def _ensure_circuit_sparsity_package() -> None:
    """Download gpt.py / hook_utils.py from HF and expose as a Python package."""
    pkg_dir = "/tmp/circuit_sparsity"
    if not os.path.exists(os.path.join(pkg_dir, "gpt.py")):
        os.makedirs(pkg_dir, exist_ok=True)
        for fname in ["gpt.py", "hook_utils.py"]:
            src = hf_hub_download(MODEL_ID, fname)
            shutil.copy2(src, os.path.join(pkg_dir, fname))
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")
        print(f"Built circuit_sparsity package at {pkg_dir}")
    if "/tmp" not in sys.path:
        sys.path.insert(0, "/tmp")


def _patch_gptconfig() -> None:
    """Wrap GPTConfig.__init__ to ignore unknown kwargs (HF config passes extras)."""
    import circuit_sparsity.gpt  # noqa: E402

    _GPTConfig = circuit_sparsity.gpt.GPTConfig
    _valid_fields = {f.name for f in dataclasses.fields(_GPTConfig)}
    _original_init = _GPTConfig.__init__

    def _patched_init(self, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in _valid_fields}
        _original_init(self, **filtered)

    _GPTConfig.__init__ = _patched_init
    print(f"Patched GPTConfig (accepts {len(_valid_fields)} fields)")


def get_layers(model):
    """Return the list of transformer blocks. Handles several common layouts."""
    if hasattr(model, "circuit_model"):
        return model.circuit_model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "h"):
        return model.model.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers in model.")


def load_model_and_tokenizer(device_map: str = "auto"):
    """
    Return (model, tokenizer, layers) ready for inference.

    The model is in eval mode with gradients disabled for all params.
    """
    assert torch.cuda.is_available(), "GPU required — Runtime → Change runtime type → GPU."

    _ensure_circuit_sparsity_package()
    _patch_gptconfig()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    n_layers = len(layers)
    print(f"Model: {MODEL_ID} | {n_layers} layers")

    return model, tokenizer, layers
