from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CondBatch:
    """Container for condition inputs.

    Currently supports:
    - tabular: Tensor (B, tabular_dim)

    Future extension points:
    - text: precomputed embeddings or token ids
    """

    tabular: Optional[torch.Tensor] = None


class CondEncoder(nn.Module):
    """Condition encoder with CFG dropout support.

    Hard constraints:
    - If condition is dropped (CFG), y_c MUST become unconditional (zeros or null embedding).
    - Inference/feature extraction should default to unconditional (avoid label leakage).

    Notes:
    - This module is intentionally minimal; it provides a stable interface for future metadata/text.
    """

    def __init__(
        self,
        *,
        d_cond: int,
        tabular_dim: int = 0,
        use_null_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.d_cond = int(d_cond)
        self.tabular_dim = int(tabular_dim)
        self.use_null_embedding = bool(use_null_embedding)

        self.tabular_net: Optional[nn.Module]
        if self.tabular_dim > 0:
            self.tabular_net = nn.Sequential(
                nn.Linear(self.tabular_dim, self.d_cond),
                nn.SiLU(),
                nn.Linear(self.d_cond, self.d_cond),
            )
        else:
            self.tabular_net = None

        if self.use_null_embedding:
            self.null = nn.Parameter(torch.zeros(self.d_cond))
        else:
            self.null = None

    def unconditional(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.null is None:
            return torch.zeros((batch_size, self.d_cond), device=device, dtype=dtype)
        return self.null.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

    def cfg_drop(self, y_c: torch.Tensor, *, drop_prob: float) -> torch.Tensor:
        """Apply classifier-free guidance drop to a condition embedding.

        With probability drop_prob per-sample, replaces y_c with unconditional embedding.
        """
        if y_c.ndim != 2 or y_c.shape[1] != self.d_cond:
            raise ValueError(f"y_c must be (B,d_cond), got {tuple(y_c.shape)}")
        p = float(drop_prob)
        if p <= 0.0:
            return y_c
        if p > 1.0:
            raise ValueError("drop_prob must be in [0,1]")
        b = int(y_c.shape[0])
        mask = (torch.rand((b,), device=y_c.device) < p)
        if not mask.any():
            return y_c
        y = y_c.clone()
        y0 = self.unconditional(b, device=y_c.device, dtype=y_c.dtype)
        y[mask] = y0[mask]
        return y

    def encode(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cond: Optional[CondBatch] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Encode condition to y_c (B, d_cond).

        - If cond is None (or force_uncond=True): returns unconditional y_c.
        - If cond.tabular is provided: encodes tabular features.

        This helper exists so training/inference can remain unconditional by default.
        """
        b = int(batch_size)
        if force_uncond or cond is None:
            return self.unconditional(b, device=device, dtype=dtype)

        if cond.tabular is not None:
            if self.tabular_net is None:
                raise ValueError("tabular provided but tabular_dim=0 in CondEncoder")
            if cond.tabular.shape[0] != b:
                raise ValueError("CondBatch.tabular batch size mismatch")
            return self.tabular_net(cond.tabular.to(device=device, dtype=dtype))

        # No known condition types present.
        return self.unconditional(b, device=device, dtype=dtype)

    def forward(
        self,
        cond: Optional[CondBatch],
        *,
        drop_prob: float = 0.0,
        force_uncond: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode condition.

        Args:
            cond: optional condition batch.
            drop_prob: CFG drop probability. Applied per-sample.
            force_uncond: if True, always return unconditional y_c.

        Returns:
            y_c: (B, d_cond)
            drop_mask: (B,) bool mask indicating dropped samples (or None)
        """
        if cond is None or force_uncond:
            raise ValueError("CondEncoder.forward requires a CondBatch; use encode(..., cond=None) for unconditional")

        # Determine batch size
        if cond.tabular is not None:
            b = int(cond.tabular.shape[0])
            device = cond.tabular.device
            dtype = cond.tabular.dtype
        else:
            raise ValueError("CondBatch must contain at least one condition tensor (e.g., tabular)")

        y = self.unconditional(b, device=device, dtype=dtype)
        if cond.tabular is not None:
            if self.tabular_net is None:
                raise ValueError("tabular provided but tabular_dim=0 in CondEncoder")
            y = self.tabular_net(cond.tabular)

        drop_mask = None
        if drop_prob > 0.0:
            p = float(drop_prob)
            if p < 0.0 or p > 1.0:
                raise ValueError("drop_prob must be in [0,1]")
            drop_mask = torch.rand((b,), device=device) < p
            if drop_mask.any():
                y = y.clone()
                y0 = self.unconditional(b, device=device, dtype=dtype)
                y[drop_mask] = y0[drop_mask]

        return y, drop_mask


def make_unconditional_cond(batch_size: int, device: torch.device) -> CondBatch:
    # A simple helper for unconditional path when no metadata exists.
    return CondBatch(tabular=torch.zeros((batch_size, 1), device=device))
