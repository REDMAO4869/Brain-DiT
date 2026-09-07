from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _sincos_1d(pos: torch.Tensor, dim: int, *, temperature: float = 10_000.0) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("dim must be even for sin-cos embedding")
    half = dim // 2
    omega = torch.arange(half, device=pos.device, dtype=torch.float32) / float(half)
    omega = 1.0 / (temperature**omega)
    out = pos.float().unsqueeze(1) * omega.unsqueeze(0)  # (L, half)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (L, dim)


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, *, device: torch.device) -> torch.Tensor:
    """2D sin-cos positional embedding.

    Returns:
        pos: (1, L, dim) where L = h*w
    """
    if dim % 2 != 0:
        raise ValueError("pos_embed dim must be even")
    dim_h = dim // 2
    dim_w = dim - dim_h

    grid_y = torch.arange(h, device=device)
    grid_x = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(-1)  # (L,)
    xx = xx.reshape(-1)

    emb_y = _sincos_1d(yy, dim_h)
    emb_x = _sincos_1d(xx, dim_w)
    emb = torch.cat([emb_y, emb_x], dim=1)  # (L, dim)
    return emb.unsqueeze(0)


class PatchEmbed2D(nn.Module):
    """Conv2d patchify embedding for latent map (B,C,n,t)."""

    def __init__(self, in_ch: int, d_model: int, patch_size: int) -> None:
        super().__init__()
        p = int(patch_size)
        self.patch_size = p
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=p, stride=p)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # z: (B,C,n,t)
        x = self.proj(z)  # (B,d,H,W)
        b, d, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, d)  # (B,L,d)
        return tokens, h, w


class TimestepEmbedder(nn.Module):
    def __init__(self, d_cond: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        self.d_cond = int(d_cond)
        h = int(hidden) if hidden is not None else int(d_cond * 4)
        self.mlp = nn.Sequential(
            nn.Linear(d_cond, h),
            nn.SiLU(),
            nn.Linear(h, d_cond),
        )

    @staticmethod
    def sincos(t: torch.Tensor, dim: int) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError("t must be (B,)")
        half = dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / float(half)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = self.sincos(t, self.d_cond)
        return self.mlp(base)


class AdaLNZero(nn.Module):
    """AdaLN-Zero modulation.

    Produces shift/scale and residual gates for attention and MLP.

    Initialization: last Linear weights and bias are zero -> starts as identity.
    """

    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_cond = int(d_cond)
        hidden = int(d_cond * 4)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.d_cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6 * self.d_model),
        )
        # zero-init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # y: (B,d_cond)
        out = self.net(y)
        b = out.shape[0]
        out = out.view(b, 6, self.d_model)
        shift1, scale1, gate1, shift2, scale2, gate2 = out.unbind(dim=1)
        return shift1, scale1, gate1, shift2, scale2, gate2


class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        d_cond: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.mod = AdaLNZero(d_model=d_model, d_cond=d_cond)

    @staticmethod
    def _adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d), shift/scale: (B,d)
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d_model), y: (B,d_cond)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(y)

        h = self.ln1(x)
        h = self._adaln(h, shift1, scale1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out) * gate1.unsqueeze(1)

        h = self.ln2(x)
        h = self._adaln(h, shift2, scale2)
        h = self.mlp(h)
        x = x + self.drop(h) * gate2.unsqueeze(1)
        return x


@dataclass
class DiTOutput:
    pred: torch.Tensor
    hiddens: Optional[Dict[int, torch.Tensor]] = None


class DiTDenoiser(nn.Module):
    """DiT denoiser on latent maps (B,C,n,t).

    Hard constraints:
    - Input/Output latent maps are always (B,C,n,t)
    - patchify/unpatchify must be exactly invertible when n,t divisible by patch_size
    - supports feature capture of intermediate tokens
    """

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int,
        d_model: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        d_cond: int,
        pos_embed: str = "sincos",  # "sincos"|"learned"
        max_h: int = 256,
        max_w: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.d_cond = int(d_cond)
        self.pos_embed_type = str(pos_embed)

        self.patch = PatchEmbed2D(self.in_channels, self.d_model, self.patch_size)

        if self.pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_h * max_w, self.d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.max_h = int(max_h)
            self.max_w = int(max_w)
        elif self.pos_embed_type == "sincos":
            self.pos_embed = None
            self.max_h = 0
            self.max_w = 0
        else:
            raise ValueError("pos_embed must be 'sincos' or 'learned'")

        self.time = TimestepEmbedder(d_cond=self.d_cond)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    d_cond=self.d_cond,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.in_channels * self.patch_size * self.patch_size)

    def _pos_embed(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if self.pos_embed_type == "learned":
            if h > self.max_h or w > self.max_w:
                raise ValueError(
                    f"learned pos_embed max grid exceeded: got (h,w)=({h},{w}) but max=({self.max_h},{self.max_w})"
                )
            l = h * w
            return self.pos_embed[:, :l, :].to(device=device)
        # sincos
        return build_2d_sincos_pos_embed(h, w, self.d_model, device=device)

    def unpatchify(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # tokens: (B, L, C*p*p)
        b, l, d = tokens.shape
        p = self.patch_size
        c = self.in_channels
        if l != h * w:
            raise ValueError(f"L mismatch: L={l} but h*w={h*w}")
        if d != c * p * p:
            raise ValueError(f"token dim mismatch: got {d}, expected {c*p*p}")

        x = tokens.view(b, h, w, c, p, p)
        # (B, h, w, c, p, p) -> (B, c, h*p, w*p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h * p, w * p)
        return x

    def forward(
        self,
        z: torch.Tensor,
        timesteps: torch.Tensor,
        y_c: torch.Tensor,
        *,
        return_hiddens: bool = False,
        capture_layers: Optional[Sequence[int]] = None,
    ) -> DiTOutput:
        if z.ndim != 4:
            raise ValueError(f"Expected z as (B,C,n,t), got {tuple(z.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != z.shape[0]:
            raise ValueError("timesteps must be (B,)")
        if y_c.ndim != 2 or y_c.shape[0] != z.shape[0] or y_c.shape[1] != self.d_cond:
            raise ValueError("y_c must be (B,d_cond)")

        b, c, n, t = z.shape
        p = self.patch_size
        if (n % p) != 0 or (t % p) != 0:
            raise ValueError(
                f"latent map size (n,t)=({n},{t}) must be divisible by patch_size={p} for invertible patchify/unpatchify"
            )

        x, h, w = self.patch(z)  # (B,L,d)
        x = x + self._pos_embed(h, w, device=z.device)

        y_t = self.time(timesteps)
        y = y_t + y_c

        cap = set(int(i) for i in (capture_layers or []))
        hiddens: Optional[Dict[int, torch.Tensor]] = {} if return_hiddens else None

        for i, blk in enumerate(self.blocks):
            x = blk(x, y)
            if hiddens is not None and i in cap:
                hiddens[i] = x

        x = self.final_ln(x)
        out = self.head(x)  # (B,L,C*p*p)
        z_hat = self.unpatchify(out, h, w)  # (B,C,n,t)
        return DiTOutput(pred=z_hat, hiddens=hiddens)
