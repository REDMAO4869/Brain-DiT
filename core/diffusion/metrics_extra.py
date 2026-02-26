from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def compute_recon_metrics(x: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, float]:
    """Compute reconstruction metrics between X and X_hat.

    Shapes:
      x, x_hat: (B, T, N)

    Metrics (per spec):
      - mse: mean((x - x_hat)^2) over all elements
      - mae: mean(|x - x_hat|)
      - corr_global: per-sample Pearson corr on flattened vectors, then mean over batch
      - corr_roi_mean: per-sample mean Pearson corr over ROIs (corr along time dim), then mean over batch

    Constant ROI handling:
      - If a ROI has std==0 for either x or x_hat within a sample, that ROI correlation is ignored (NaN skipped).
      - Returns corr_roi_num_valid: average number of valid ROIs per sample.
    """
    if x.ndim != 3 or x_hat.ndim != 3:
        raise ValueError(f"Expected (B,T,N) tensors, got x={tuple(x.shape)} x_hat={tuple(x_hat.shape)}")
    if x.shape != x_hat.shape:
        raise ValueError(f"Shape mismatch: x={tuple(x.shape)} x_hat={tuple(x_hat.shape)}")

    diff = x - x_hat
    mse = torch.mean(diff * diff)
    mae = torch.mean(torch.abs(diff))

    # corr_global: flatten per sample
    b = x.shape[0]
    xf = x.reshape(b, -1)
    yf = x_hat.reshape(b, -1)
    xf = xf - xf.mean(dim=1, keepdim=True)
    yf = yf - yf.mean(dim=1, keepdim=True)
    xstd = xf.std(dim=1, unbiased=False)
    ystd = yf.std(dim=1, unbiased=False)
    denom = xstd * ystd
    corr_g = torch.full((b,), float("nan"), device=x.device, dtype=x.dtype)
    ok = denom > 0
    if ok.any():
        corr_g[ok] = torch.sum(xf[ok] * yf[ok], dim=1) / (xf.shape[1] * denom[ok])
    corr_global = torch.nanmean(corr_g)

    # corr_roi_mean: per ROI correlation along T
    # x: (B,T,N) -> center along T
    x0 = x - x.mean(dim=1, keepdim=True)
    y0 = x_hat - x_hat.mean(dim=1, keepdim=True)
    xstd_roi = x0.std(dim=1, unbiased=False)  # (B,N)
    ystd_roi = y0.std(dim=1, unbiased=False)  # (B,N)
    denom_roi = xstd_roi * ystd_roi
    corr_roi = torch.full_like(denom_roi, float("nan"))
    ok_roi = denom_roi > 0
    if ok_roi.any():
        # cov along T divided by T
        cov = torch.mean(x0 * y0, dim=1)  # (B,N)
        corr_roi[ok_roi] = cov[ok_roi] / denom_roi[ok_roi]

    # Per-sample mean over ROIs, then mean over batch
    corr_roi_per_sample = torch.nanmean(corr_roi, dim=1)  # (B,)
    corr_roi_mean = torch.nanmean(corr_roi_per_sample)

    num_valid = torch.sum(ok_roi, dim=1).to(torch.float32)  # (B,)
    corr_roi_num_valid = torch.mean(num_valid)

    return {
        "recon_mse": float(mse.item()),
        "recon_mae": float(mae.item()),
        "recon_corr_global": float(corr_global.item()) if torch.isfinite(corr_global) else float("nan"),
        "recon_corr_roi_mean": float(corr_roi_mean.item()) if torch.isfinite(corr_roi_mean) else float("nan"),
        "recon_corr_roi_num_valid": float(corr_roi_num_valid.item()),
    }
