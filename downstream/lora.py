from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._path import ensure_repo_root

ensure_repo_root()


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRAMultiheadAttention(nn.Module):
    """Wrap nn.MultiheadAttention and add LoRA updates to Q and V projections."""

    def __init__(self, base: nn.MultiheadAttention, cfg: LoRAConfig) -> None:
        super().__init__()
        if not base.batch_first:
            raise ValueError("Expected base MultiheadAttention with batch_first=True")

        self.embed_dim = int(base.embed_dim)
        self.num_heads = int(base.num_heads)
        self.dropout_p = float(base.dropout)

        self.in_proj_weight = base.in_proj_weight
        self.in_proj_bias = base.in_proj_bias
        self.out_proj = base.out_proj

        self._base = base
        for p in self._base.parameters():
            p.requires_grad = False

        r = int(cfg.r)
        if r <= 0:
            raise ValueError("LoRA r must be > 0")
        self.r = r
        self.alpha = float(cfg.alpha)
        self.scaling = self.alpha / float(self.r)
        self.lora_dropout = nn.Dropout(float(cfg.dropout))

        if self.in_proj_weight is None:
            raise RuntimeError("Expected base attention to have in_proj_weight")
        dev = self.in_proj_weight.device
        dt = self.in_proj_weight.dtype

        self.q_A = nn.Parameter(torch.zeros((self.r, self.embed_dim), device=dev, dtype=dt))
        self.q_B = nn.Parameter(torch.zeros((self.embed_dim, self.r), device=dev, dtype=dt))
        self.v_A = nn.Parameter(torch.zeros((self.r, self.embed_dim), device=dev, dtype=dt))
        self.v_B = nn.Parameter(torch.zeros((self.embed_dim, self.r), device=dev, dtype=dt))

        nn.init.kaiming_uniform_(self.q_A, a=5**0.5)
        nn.init.zeros_(self.q_B)
        nn.init.kaiming_uniform_(self.v_A, a=5**0.5)
        nn.init.zeros_(self.v_B)

    def _delta(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (B @ A) * A.new_tensor(self.scaling)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, need_weights: bool = False):
        if query is not key or key is not value:
            raise ValueError("This wrapper supports self-attention only (query==key==value)")

        E = self.embed_dim
        w = self.in_proj_weight
        if w is None:
            raise RuntimeError("Expected in_proj_weight")

        w_q = w[0:E, :]
        w_k = w[E : 2 * E, :]
        w_v = w[2 * E : 3 * E, :]

        dq = self._delta(self.q_A, self.q_B)
        dv = self._delta(self.v_A, self.v_B)

        w_eff = torch.cat([w_q + dq, w_k, w_v + dv], dim=0)

        q = query.transpose(0, 1)
        k = key.transpose(0, 1)
        v = value.transpose(0, 1)

        attn_out, attn_w = F.multi_head_attention_forward(
            query=q,
            key=k,
            value=v,
            embed_dim_to_check=E,
            num_heads=self.num_heads,
            in_proj_weight=w_eff,
            in_proj_bias=self.in_proj_bias,
            bias_k=self._base.bias_k,
            bias_v=self._base.bias_v,
            add_zero_attn=self._base.add_zero_attn,
            dropout_p=self.dropout_p if self.training else 0.0,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=need_weights,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=False,
            is_causal=False,
        )

        out = attn_out.transpose(0, 1)
        return out, attn_w


def apply_lora_to_last_k_blocks(dit, *, last_k: int, cfg: LoRAConfig) -> int:
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise ValueError("Expected DiT model with .blocks")

    depth = len(blocks)
    k = int(last_k)
    if k <= 0:
        return 0

    start = max(0, depth - k)
    replaced = 0
    for i in range(start, depth):
        blk = blocks[i]
        base_attn = blk.attn
        blk.attn = LoRAMultiheadAttention(base_attn, cfg)
        replaced += 1
    return replaced
