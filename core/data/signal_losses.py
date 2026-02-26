from __future__ import annotations

import contextlib
from typing import Literal, Tuple

import torch


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor, lambda2: float) -> torch.Tensor:
    # L1 + lambda2 * L2^2
    l1 = torch.mean(torch.abs(x - x_hat))
    l2 = torch.mean((x - x_hat) ** 2)
    return l1 + lambda2 * l2


def kl_loss_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    reduction: Literal["mean", "sum"] = "mean",
    logvar_clamp_min: float = -20.0,
    logvar_clamp_max: float = 20.0,
    debug_asserts: bool = False,
) -> torch.Tensor:
    """KL divergence to N(0, I) supporting vector or feature-map latents.

    mu/logvar can be (B, D) or (B, C, n, t) etc.

    reduction:
        - "mean": mean over non-batch dims per sample, then mean over batch (stable scale)
        - "sum":  sum over non-batch dims per sample, then mean over batch
    """
    if debug_asserts:
        if mu.shape != logvar.shape:
            raise ValueError(f"mu/logvar shape mismatch: {tuple(mu.shape)} vs {tuple(logvar.shape)}")
        if mu.ndim < 2:
            raise ValueError(f"Expected mu/logvar with batch dim, got mu.ndim={mu.ndim}")
        if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
            raise ValueError("Non-finite values in mu/logvar")

    logvar = logvar.clamp(float(logvar_clamp_min), float(logvar_clamp_max))

    # kl_per_elem = 0.5 * (mu^2 + exp(logvar) - 1 - logvar)
    kl_per_elem = 0.5 * (mu.pow(2) + torch.exp(logvar) - 1.0 - logvar)

    # Reduce over all non-batch dims.
    dims = tuple(range(1, kl_per_elem.ndim))
    if reduction == "mean":
        kl_per_sample = kl_per_elem.mean(dim=dims)
    elif reduction == "sum":
        kl_per_sample = kl_per_elem.sum(dim=dims)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    kl = kl_per_sample.mean()
    return kl


def fft_magnitude_loss(x: torch.Tensor, x_hat: torch.Tensor, half_spectrum: bool = True) -> torch.Tensor:
    # x: (B, T, ROI) FFT over time dim
    # NOTE: cuFFT has limitations for half precision on non-power-of-two lengths.
    # Run FFT in fp32 with autocast disabled (CUDA only) to avoid runtime errors.
    if x.is_cuda:
        autocast_off = torch.amp.autocast(device_type="cuda", enabled=False)
    else:
        autocast_off = contextlib.nullcontext()

    with autocast_off:
        x32 = x.float()
        xh32 = x_hat.float()

        if half_spectrum:
            fx = torch.fft.rfft(x32, dim=1)
            fxh = torch.fft.rfft(xh32, dim=1)
        else:
            fx = torch.fft.fft(x32, dim=1)
            fxh = torch.fft.fft(xh32, dim=1)

        mag_x = torch.abs(fx)
        mag_xh = torch.abs(fxh)
        return torch.mean((mag_x - mag_xh) ** 2)


def _acf(x: torch.Tensor, lags: int, eps: float = 1e-6) -> torch.Tensor:
    """Compute ACF for each ROI with given lags.

    Args:
        x: (B, T, ROI)
    Returns:
        acf: (B, lags, ROI) for lag=1..lags
    """
    b, t, r = x.shape
    x = x - x.mean(dim=1, keepdim=True)
    denom = torch.mean(x * x, dim=1, keepdim=False) + eps  # (B, ROI)

    acfs = []
    max_lag = min(lags, t - 1)
    for lag in range(1, max_lag + 1):
        x0 = x[:, :-lag, :]
        x1 = x[:, lag:, :]
        num = torch.mean(x0 * x1, dim=1)  # (B, ROI)
        acfs.append(num / denom)

    if len(acfs) == 0:
        return torch.zeros((b, 0, r), device=x.device, dtype=x.dtype)

    return torch.stack(acfs, dim=1)


def acf_loss(x: torch.Tensor, x_hat: torch.Tensor, lags: int) -> torch.Tensor:
    ax = _acf(x, lags)
    axh = _acf(x_hat, lags)
    # if T is small, ax could have <lags entries; match by min length
    m = min(ax.shape[1], axh.shape[1])
    if m == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.mean((ax[:, :m, :] - axh[:, :m, :]) ** 2)


def beta_warmup(global_step: int, beta_max: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return float(beta_max)
    frac = min(1.0, float(global_step) / float(warmup_steps))
    return float(beta_max) * frac
