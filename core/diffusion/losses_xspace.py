from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch

from core.data.signal_losses import acf_loss as stage0_acf_loss
from core.data.signal_losses import fft_magnitude_loss as stage0_fft_magnitude_loss


@dataclass
class XSpaceLossCfg:
    use_xspace_loss: bool
    lambda_fc: float
    lambda_fft: float
    lambda_acf: float
    xspace_loss_every_k_steps: int
    xspace_loss_subsample_ratio: float
    xspace_start_step: int
    fft_half_spectrum: bool
    acf_lags: int


def _corrcoef_time(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute ROI×ROI correlation for each sample.

    Args:
        x: (B,T,N)
    Returns:
        corr: (B,N,N)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x as (B,T,N), got {tuple(x.shape)}")
    b, t, n = x.shape
    x = x - x.mean(dim=1, keepdim=True)
    # std can be tiny; clamp for stability
    x = x / (x.std(dim=1, keepdim=True).clamp_min(eps))
    # corr = (X^T X)/(T-1)
    xt = x.transpose(1, 2)  # (B,N,T)
    corr = torch.matmul(xt, x) / float(max(1, t - 1))
    return corr


def _autocast_off(x: torch.Tensor):
    if x.is_cuda:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return contextlib.nullcontext()


def _finite_or_zero(loss: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(loss):
        return loss
    return torch.zeros((), device=loss.device, dtype=loss.dtype)


def fc_loss_fisherz(x: torch.Tensor, x_hat: torch.Tensor, clamp: float = 0.999_999) -> torch.Tensor:
    """Functional connectivity loss with Fisher Z transform.

    Computes per-sample FC (corr matrix across ROIs), FisherZ, then MSE.
    """
    # NOTE: Under AMP, clamp constants can be cast to fp16 and round to 1.0,
    # making atanh(1)=inf. Force fp32 + autocast off for stability.
    with _autocast_off(x):
        x32 = x.float()
        xh32 = x_hat.float()
        c1 = _corrcoef_time(x32)
        c2 = _corrcoef_time(xh32)
        c1 = c1.clamp(-float(clamp), float(clamp))
        c2 = c2.clamp(-float(clamp), float(clamp))
        z1 = torch.atanh(c1)
        z2 = torch.atanh(c2)
        loss = torch.mean((z1 - z2) ** 2)
        return _finite_or_zero(loss)


def fft_loss(x: torch.Tensor, x_hat: torch.Tensor, *, half_spectrum: bool) -> torch.Tensor:
    with _autocast_off(x):
        loss = stage0_fft_magnitude_loss(x.float(), x_hat.float(), half_spectrum=half_spectrum)
        return _finite_or_zero(loss)


def acf_loss(x: torch.Tensor, x_hat: torch.Tensor, *, lags: int) -> torch.Tensor:
    with _autocast_off(x):
        loss = stage0_acf_loss(x.float(), x_hat.float(), lags=lags)
        return _finite_or_zero(loss)


@torch.no_grad()
def choose_subsample_mask(batch_size: int, ratio: float, device: torch.device) -> Optional[torch.Tensor]:
    """Return boolean mask (B,) selecting a subset, or None meaning select all."""
    r = float(ratio)
    if r >= 1.0:
        return None
    if r <= 0.0:
        return torch.zeros((batch_size,), device=device, dtype=torch.bool)
    return (torch.rand((batch_size,), device=device) < r)
