"""
Extract residual-stream hidden states at the final token position
using forward hooks (CSP does not support ``output_hidden_states``).
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import MAX_SEQ_LENGTH, MAX_PROBE_SAMPLES
from .data_loader import prompt_from_scenario


def collect_resid_all_layers(
    examples: List[dict],
    model,
    tokenizer,
    layers,
    max_samples: int = MAX_PROBE_SAMPLES,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Pass each example through the model and capture the post-block hidden
    state at the **last token** of every layer using hooks.

    Returns ``{layer_idx: (X, y)}`` where *X* is ``(n_samples, d_model)``
    and *y* is ``(n_samples,)`` with 0/1 labels.
    """
    device = next(model.parameters()).device
    n_layers = len(layers)
    all_vecs: Dict[int, list] = {L: [] for L in range(n_layers)}
    all_labels: list = []
    hidden_buf = [None] * (n_layers + 1)

    def _make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden_buf[idx + 1] = h.detach().clone()
        return hook

    handles = [layers[i].register_forward_hook(_make_hook(i)) for i in range(n_layers)]

    n_collect = min(max_samples, len(examples))
    print(f"Collecting hidden states at all {n_layers} layers (hook-based)...")

    for i in range(n_collect):
        ex = examples[i]
        code = prompt_from_scenario(ex["code"])
        enc = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False,
        ).to(device)
        ids = enc["input_ids"]
        pos = ids.shape[1] - 1

        with torch.no_grad():
            model(input_ids=ids)

        for L in range(n_layers):
            vec = hidden_buf[L + 1][0, pos, :].float().cpu().numpy()
            all_vecs[L].append(vec)
        all_labels.append(ex["label"])

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_collect} examples")

    for h in handles:
        h.remove()

    labels_arr = np.array(all_labels)
    result = {L: (np.stack(all_vecs[L]), labels_arr) for L in range(n_layers)}
    print(f"Done. {n_layers} layers, {result[0][0].shape[0]} examples each.")
    return result
