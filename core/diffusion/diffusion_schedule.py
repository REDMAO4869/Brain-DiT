from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import torch


PredType = Literal["v", "eps"]
ScheduleType = Literal["cosine", "linear"]


@dataclass(frozen=True)
class DiffusionSchedule:
    """Precomputes a simple DDPM-style schedule in (alpha, sigma) parameterization.

    We use the common forward process:
        x_t = alpha_t * x0 + sigma_t * eps
    where alpha_t = sqrt(alpha_bar_t), sigma_t = sqrt(1 - alpha_bar_t).

    v-parameterization (fixed convention used here):
        v = alpha_t * eps - sigma_t * x0

    Inversion formulas used by Stage-1:
        x0 from eps: x0 = (x_t - sigma_t * eps_hat) / alpha_t
        x0 from v  : x0 = alpha_t * x_t - sigma_t * v_hat

    NOTE:
    - Timesteps are integers in [1..S]. We also keep index 0 for convenience.
    """

    num_steps: int
    schedule: ScheduleType
    alpha_bar: torch.Tensor  # (S+1,)

    @staticmethod
    def create(num_steps: int, schedule: ScheduleType = "cosine", *, device=None, dtype=torch.float32) -> "DiffusionSchedule":
        s = int(num_steps)
        if s <= 0:
            raise ValueError("num_steps must be > 0")

        if schedule == "cosine":
            alpha_bar = _cosine_alpha_bar(s, device=device, dtype=dtype)
        elif schedule == "linear":
            alpha_bar = _linear_alpha_bar(s, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Make sure numeric range is safe.
        alpha_bar = alpha_bar.clamp(1e-6, 1.0)
        return DiffusionSchedule(num_steps=s, schedule=schedule, alpha_bar=alpha_bar)

    def to(self, *, device=None, dtype=None) -> "DiffusionSchedule":
        return DiffusionSchedule(
            num_steps=self.num_steps,
            schedule=self.schedule,
            alpha_bar=self.alpha_bar.to(device=device, dtype=dtype) if (device is not None or dtype is not None) else self.alpha_bar,
        )

    def alpha_sigma(self, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (alpha_t, sigma_t) broadcastable to latent map (B,C,n,t)."""
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be 1D (B,), got {tuple(timesteps.shape)}")
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise ValueError("timesteps must be integer tensor")

        t = timesteps.clamp(0, self.num_steps)
        ab = self.alpha_bar.index_select(0, t)
        alpha = torch.sqrt(ab)
        sigma = torch.sqrt(1.0 - ab)
        return alpha, sigma

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(timesteps)
        while alpha.ndim < x0.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * x0 + sigma * eps

    def v_from_x0_eps(self, x0: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(timesteps)
        while alpha.ndim < x0.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return alpha * eps - sigma * x0

    def eps_from_xt_v(self, xt: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(timesteps)
        while alpha.ndim < xt.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        # eps = sigma * x_t + alpha * v
        return sigma * xt + alpha * v

    def x0_from_xt_eps(self, xt: torch.Tensor, timesteps: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(timesteps)
        while alpha.ndim < xt.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        return (xt - sigma * eps) / alpha

    def x0_from_xt_v(self, xt: torch.Tensor, timesteps: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(timesteps)
        while alpha.ndim < xt.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        # x0 = alpha * x_t - sigma * v
        return alpha * xt - sigma * v


def _cosine_alpha_bar(num_steps: int, *, device=None, dtype=torch.float32, s: float = 0.008) -> torch.Tensor:
    # Nichol & Dhariwal cosine schedule (alpha_bar).
    steps = torch.arange(0, num_steps + 1, device=device, dtype=dtype)
    t = steps / float(num_steps)
    f = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    # normalize so alpha_bar[0] = 1
    return f / f[0]


def _linear_alpha_bar(num_steps: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    # Simple linear beta schedule -> compute alpha_bar.
    # betas linearly from 1e-4 to 2e-2 (common default).
    beta_start = 1e-4
    beta_end = 2e-2
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bar = torch.empty((num_steps + 1,), device=device, dtype=dtype)
    alpha_bar[0] = 1.0
    alpha_bar[1:] = torch.cumprod(alphas, dim=0)
    return alpha_bar
