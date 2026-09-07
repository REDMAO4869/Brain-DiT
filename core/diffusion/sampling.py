from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from core.diffusion.diffusion_schedule import DiffusionSchedule, PredType

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@torch.no_grad()
def _compute_betas_from_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    """Compute betas (1..S) from alpha_bar (0..S).

    alpha_bar: (S+1,) with alpha_bar[0]=1.
    returns betas: (S+1,) with betas[0]=0 (unused) and betas[t] for t>=1.
    """
    if alpha_bar.ndim != 1 or alpha_bar.shape[0] < 2:
        raise ValueError("alpha_bar must be 1D (S+1,)")
    ab = alpha_bar.clamp(1e-6, 1.0)
    betas = torch.zeros_like(ab)
    betas[1:] = 1.0 - (ab[1:] / ab[:-1])
    return betas.clamp(1e-8, 0.999)


@torch.no_grad()
def sample_ddpm(
    *,
    denoiser: torch.nn.Module,
    schedule: DiffusionSchedule,
    pred_type: PredType,
    shape: Tuple[int, int, int, int],
    device: torch.device,
    cond: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    progress_every: int = 0,
    progress_desc: Optional[str] = None,
    progress_position: int = 0,
    progress_leave: bool = False,
) -> torch.Tensor:
    """Full DDPM reverse sampling from z_T ~ N(0,I) to z_0.

    - Unconditional by default: provide cond=None or a zero cond embedding.
    - pred_type: 'v' or 'eps'

    Returns:
      z0_sample: (B,C,n,t)
    """
    b, c, n, t = shape
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        z = torch.randn((b, c, n, t), device=device, generator=gen)
    else:
        z = torch.randn((b, c, n, t), device=device)

    # Prepare schedule terms
    ab = schedule.alpha_bar.to(device=device, dtype=z.dtype)
    betas = _compute_betas_from_alpha_bar(ab)

    # For unconditional sanity check, use y_c=0 if cond not provided
    if cond is None:
        # Infer d_cond if possible from module attribute, else require caller.
        d_cond = getattr(denoiser, "d_cond", None)
        if d_cond is None:
            raise ValueError("cond is None but denoiser has no d_cond attribute; pass cond explicitly")
        cond = torch.zeros((b, int(d_cond)), device=device, dtype=z.dtype)
    else:
        cond = cond.to(device=device, dtype=z.dtype)

    steps = range(schedule.num_steps, 0, -1)
    if progress_every > 0 and tqdm is not None:
        steps = tqdm(
            steps,
            desc=progress_desc or "ddpm",
            position=progress_position,
            leave=progress_leave,
            miniters=max(1, int(progress_every)),
        )

    for ti in steps:
        t_int = torch.full((b,), ti, device=device, dtype=torch.int64)
        out = denoiser(z, t_int, cond)
        pred = out.pred

        if pred_type == "v":
            eps_hat = schedule.eps_from_xt_v(z, t_int, pred)
        else:
            eps_hat = pred

        # DDPM posterior mean
        beta_t = betas[ti]
        alpha_t = 1.0 - beta_t
        ab_t = ab[ti]
        ab_prev = ab[ti - 1]

        # mu = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-ab_t) * eps_hat)
        mu = (z - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_hat) / torch.sqrt(alpha_t)

        if ti == 1:
            z = mu
        else:
            # posterior variance
            var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
            noise = torch.randn_like(z)
            z = mu + torch.sqrt(var) * noise

    return z


@torch.no_grad()
def compute_sample_stats(x: torch.Tensor) -> Dict[str, float]:
    """Compute simple plausibility stats for generated samples.

    x: (B,T,N)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x as (B,T,N), got {tuple(x.shape)}")

    # Basic stats
    x_min = torch.min(x)
    x_max = torch.max(x)
    x_std = torch.std(x)

    # FC stats: correlation over time per sample, stats over off-diagonal entries
    b, t, n = x.shape
    x0 = x - x.mean(dim=1, keepdim=True)
    x0 = x0 / (x0.std(dim=1, keepdim=True) + 1e-6)
    xt = x0.transpose(1, 2)  # (B,N,T)
    corr = torch.matmul(xt, x0) / float(max(1, t - 1))  # (B,N,N)

    tri = torch.triu(torch.ones((n, n), device=x.device, dtype=torch.bool), diagonal=1)
    vals = corr[:, tri]
    fc_mean = torch.mean(vals)
    fc_var = torch.var(vals, unbiased=False)

    # FFT energy band ratios along time dimension
    # Define bands by fraction of Nyquist (0..0.5 cycles/sample).
    # low: [0, 0.1], mid: (0.1, 0.3], high: (0.3, 0.5]
    xf = torch.fft.rfft(x.to(torch.float32), dim=1)  # (B, F, N)
    mag2 = (xf.real**2 + xf.imag**2)  # power
    power = torch.mean(mag2, dim=2)  # (B,F)

    f = power.shape[1]
    i_low = max(1, int(0.1 * f))
    i_mid = max(i_low + 1, int(0.3 * f))

    e_total = torch.sum(power, dim=1) + 1e-12
    e_low = torch.sum(power[:, :i_low], dim=1)
    e_mid = torch.sum(power[:, i_low:i_mid], dim=1)
    e_high = torch.sum(power[:, i_mid:], dim=1)

    r_low = torch.mean(e_low / e_total)
    r_mid = torch.mean(e_mid / e_total)
    r_high = torch.mean(e_high / e_total)

    return {
        "sample_x_min": float(x_min.item()),
        "sample_x_max": float(x_max.item()),
        "sample_x_std": float(x_std.item()),
        "sample_fc_mean": float(fc_mean.item()),
        "sample_fc_var": float(fc_var.item()),
        "sample_fft_energy_low": float(r_low.item()),
        "sample_fft_energy_mid": float(r_mid.item()),
        "sample_fft_energy_high": float(r_high.item()),
    }
