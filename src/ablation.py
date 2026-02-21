"""
Circuit ablation engine for CSP model.

Supports zero ablation and mean ablation of specific feature indices
in the residual stream at a given layer.
"""
from typing import List, Optional

import numpy as np
import torch


class CircuitAblator:
    """
    Register/remove a forward hook on a transformer layer to ablate
    specific feature indices in the residual stream.

    Strategies:
    - zero: set ablated features to 0
    - mean: replace with dataset mean for those features
    """

    def __init__(
        self,
        layer_module,
        layer_idx: int,
        indices: List[int],
        strategy: str = "zero",
        mean_values: Optional[np.ndarray] = None,
    ):
        """
        Args:
            layer_module: The transformer block (e.g. model.transformer.h[i])
            layer_idx: Layer index (for reference)
            indices: Feature indices to ablate (0..d_model-1)
            strategy: "zero" or "mean"
            mean_values: (d_model,) array of per-feature means; required if strategy="mean"
        """
        self.layer_module = layer_module
        self.layer_idx = layer_idx
        self.indices = list(indices)
        self.strategy = strategy
        self.mean_values = mean_values
        self._handle = None
        self._enabled = False

    def _make_hook(self):
        indices = self.indices
        strategy = self.strategy
        mean_values = self.mean_values

        def hook(module, inp, out):
            if not indices:
                return out
            out_is_tuple = isinstance(out, tuple)
            h = out[0] if out_is_tuple else out
            h = h.clone()
            if strategy == "zero":
                h[..., indices] = 0.0
            elif strategy == "mean" and mean_values is not None:
                dev = h.device
                mv = torch.tensor(mean_values[indices], dtype=h.dtype, device=dev)
                h[..., indices] = mv
            if out_is_tuple:
                return (h,) + out[1:]
            return h

        return hook

    def enable(self) -> None:
        """Register the ablation hook."""
        if self._handle is not None:
            return
        self._handle = self.layer_module.register_forward_hook(self._make_hook())
        self._enabled = True

    def disable(self) -> None:
        """Remove the ablation hook."""
        if self._handle is None:
            return
        self._handle.remove()
        self._handle = None
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def update_indices(self, indices: List[int]) -> None:
        """Change which features to ablate (must disable first if currently enabled)."""
        self.indices = list(indices)
