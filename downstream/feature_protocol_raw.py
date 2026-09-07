from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from downstream._path import ensure_repo_root
from downstream.noise_utils import NoiseMode, noise_for_batch

ensure_repo_root()

from core.diffusion.cond_encoder import CondEncoder
from core.diffusion.diffusion_schedule import DiffusionSchedule
from core.diffusion.model_dit import DiTDenoiser


def _unwrap_ddp(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def resolve_capture_layers(depth: int, capture_layers: Sequence[int]) -> List[int]:
    d = int(depth)
    out: List[int] = []
    for lid in capture_layers:
        i = int(lid)
        if i < 0:
            i = d + i
        if i < 0 or i >= d:
            raise ValueError(f"capture_layer out of range: got {lid} -> {i}, depth={d}")
        out.append(i)
    return out


def _to_x0(x: torch.Tensor) -> torch.Tensor:
    # (B,T,N) -> (B,1,N,T)
    return x.permute(0, 2, 1).unsqueeze(1).contiguous()


@dataclass(frozen=True)
class FeatureProtocolRawConfig:
    timestep: int
    capture_layers: List[int]
    noise_mode: NoiseMode = "per_subject"
    noise_seed: int = 0


@dataclass(frozen=True)
class FeatureProtocolOutput:
    tokens_list: List[torch.Tensor]
    meta: Dict[str, Any]


class FeatureProtocolRaw:
    """x -> x0(raw) -> add noise at fixed timestep -> DiT forward -> capture multi-layer tokens."""

    def __init__(
        self,
        *,
        dit: DiTDenoiser,
        cond_encoder: CondEncoder,
        schedule: DiffusionSchedule,
        cfg: FeatureProtocolRawConfig,
        device: torch.device,
    ) -> None:
        self.dit = dit
        self.cond = cond_encoder
        self.schedule = schedule
        self.cfg = cfg
        self.device = device

    def tokens_from_batch(
        self,
        x: torch.Tensor,
        *,
        subjects: Sequence[str],
        sample_indices: Optional[Sequence[int]] = None,
        enable_grad: bool = False,
        timestep: Optional[int] = None,
    ) -> FeatureProtocolOutput:
        x = x.to(device=self.device)
        x0 = _to_x0(x)

        b = int(x0.shape[0])
        t = int(self.cfg.timestep if timestep is None else timestep)
        t_vec = torch.full((b,), t, device=self.device, dtype=torch.int64)

        alpha, sigma = self.schedule.alpha_sigma(t_vec)
        while alpha.ndim < x0.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        if len(subjects) != b:
            raise ValueError(f"subjects length must match batch size: got {len(subjects)} vs {b}")

        idx_list = None
        if sample_indices is not None:
            if len(sample_indices) != b:
                raise ValueError("sample_indices length must match batch size")
            idx_list = [int(i) for i in sample_indices]

        eps = noise_for_batch(
            mode=self.cfg.noise_mode,
            subjects=[str(s) for s in subjects],
            sample_indices=idx_list,
            global_seed=int(self.cfg.noise_seed),
            shape_per_sample=tuple(x0.shape[1:]),
            device=self.device,
            dtype=x0.dtype,
        )

        xt = alpha * x0 + sigma * eps

        with torch.no_grad():
            y_c = self.cond.encode(batch_size=b, device=self.device, dtype=xt.dtype, cond=None)

        dit_core = _unwrap_ddp(self.dit)
        layers = resolve_capture_layers(int(getattr(dit_core, "depth")), self.cfg.capture_layers)

        if enable_grad:
            out = self.dit(xt, t_vec, y_c, return_hiddens=True, capture_layers=layers)
        else:
            with torch.no_grad():
                out = self.dit(xt, t_vec, y_c, return_hiddens=True, capture_layers=layers)

        hiddens = out.get("hiddens", None) if isinstance(out, dict) else getattr(out, "hiddens", None)
        if hiddens is None:
            raise RuntimeError("Expected hiddens when return_hiddens=True")

        tokens_list: List[torch.Tensor] = []
        for lid in layers:
            if lid not in hiddens:
                raise RuntimeError(f"Requested hidden layer not captured: {lid}")
            tokens_list.append(hiddens[lid])

        meta: Dict[str, Any] = {
            "timestep": int(t),
            "capture_layers_resolved": layers,
            "noise_mode": str(self.cfg.noise_mode),
            "noise_seed": int(self.cfg.noise_seed),
        }
        return FeatureProtocolOutput(tokens_list=tokens_list, meta=meta)


__all__ = [
    "FeatureProtocolOutput",
    "FeatureProtocolRaw",
    "FeatureProtocolRawConfig",
    "resolve_capture_layers",
]
