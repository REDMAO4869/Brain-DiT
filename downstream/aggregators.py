from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AggregatorOutput:
    emb: torch.Tensor
    weights: torch.Tensor


def _check_E(E: torch.Tensor, num_layers: int, d_model: int) -> None:
    if E.ndim != 3:
        raise ValueError(f"Expected E as (B,L,D), got shape={tuple(E.shape)}")
    if E.shape[1] != int(num_layers):
        raise ValueError(f"E layer dim mismatch: expected L={num_layers}, got {int(E.shape[1])}")
    if E.shape[2] != int(d_model):
        raise ValueError(f"E d_model mismatch: expected D={d_model}, got {int(E.shape[2])}")


class TokenAttentionPool(nn.Module):
    """Cross-attention over a token sequence using a learnable query.

    Intended usage: concatenate tokens from multiple captured layers into one sequence and
    attend over the sequence to produce a single embedding.
    """

    def __init__(
        self,
        *,
        d_model: int,
        use_layernorm: bool = True,
        q_len: int = 1,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.q_len = int(q_len)
        if self.q_len <= 0:
            raise ValueError("q_len must be >= 1")
        self.temperature = float(temperature)
        self.ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        self.out_ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        self.query = nn.Parameter(torch.zeros((1, self.q_len, self.d_model)))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.mha = nn.MultiheadAttention(self.d_model, int(attn_heads), dropout=float(attn_dropout), batch_first=True)

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # E is a token sequence: (B,N,D). N can vary; we do not enforce N.
        if E.ndim != 3:
            raise ValueError(f"Expected E as (B,N,D), got shape={tuple(E.shape)}")
        if int(E.shape[2]) != self.d_model:
            raise ValueError(f"E d_model mismatch: expected D={self.d_model}, got {int(E.shape[2])}")
        if self.ln is not None:
            E = self.ln(E)

        B = int(E.shape[0])
        q = self.query.expand(B, self.q_len, self.d_model)
        if self.temperature != 1.0:
            q = q / float(self.temperature)
        out, attn = self.mha(q, E, E, need_weights=True)
        if self.out_ln is not None:
            out = self.out_ln(out)
        if self.q_len == 1:
            emb = out.squeeze(1)
        else:
            emb = out.mean(dim=1)
        weights = attn.squeeze(1) if (self.q_len == 1 and attn.ndim == 3) else attn
        return emb, weights


class LWSScalar(nn.Module):
    """Layer-Weighted Sum with scalar weight per layer."""

    def __init__(self, *, d_model: int, num_layers: int, use_layernorm: bool = True, temperature: float = 1.0) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.temperature = float(temperature)
        self.ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        self.a = nn.Parameter(torch.zeros((self.num_layers,), dtype=torch.float32))

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_E(E, self.num_layers, self.d_model)
        if self.ln is not None:
            E = self.ln(E)
        logits = self.a
        if self.temperature != 1.0:
            logits = logits / float(self.temperature)
        w = torch.softmax(logits, dim=0)  # (L,)
        emb = torch.sum(E * w.view(1, -1, 1), dim=1)
        return emb, w


class LWSPerDim(nn.Module):
    """Per-dimension Layer-Weighted Sum with weight per (layer, dim)."""

    def __init__(self, *, d_model: int, num_layers: int, use_layernorm: bool = True, temperature: float = 1.0) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.temperature = float(temperature)
        self.ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        self.A = nn.Parameter(torch.zeros((self.num_layers, self.d_model), dtype=torch.float32))

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_E(E, self.num_layers, self.d_model)
        if self.ln is not None:
            E = self.ln(E)
        logits = self.A
        if self.temperature != 1.0:
            logits = logits / float(self.temperature)
        W = torch.softmax(logits, dim=0)  # (L,D)
        emb = torch.sum(E * W.unsqueeze(0), dim=1)
        w_mean = torch.mean(W, dim=1)  # (L,)
        return emb, w_mean


class GateAggregator(nn.Module):
    """Per-sample gating over layers using a small MLP on concatenated pooled features."""

    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        use_layernorm: bool = True,
        gate_hidden: int = 128,
        gate_dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.temperature = float(temperature)
        self.ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        in_dim = self.num_layers * self.d_model
        h = int(gate_hidden)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Dropout(float(gate_dropout)),
            nn.Linear(h, self.num_layers),
        )

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_E(E, self.num_layers, self.d_model)
        if self.ln is not None:
            E = self.ln(E)
        B = int(E.shape[0])
        logits = self.gate(E.reshape(B, self.num_layers * self.d_model))
        if self.temperature != 1.0:
            logits = logits / float(self.temperature)
        w = torch.softmax(logits, dim=-1)  # (B,L)
        emb = torch.sum(E * w.unsqueeze(-1), dim=1)
        return emb, w


class LayerAttention(nn.Module):
    """Cross-attention over layer tokens using a learnable query."""

    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        use_layernorm: bool = True,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.temperature = float(temperature)
        self.ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        self.out_ln = nn.LayerNorm(self.d_model) if bool(use_layernorm) else None
        self.query = nn.Parameter(torch.zeros((1, 1, self.d_model)))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.mha = nn.MultiheadAttention(self.d_model, int(attn_heads), dropout=float(attn_dropout), batch_first=True)

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_E(E, self.num_layers, self.d_model)
        if self.ln is not None:
            E = self.ln(E)
        B = int(E.shape[0])
        q = self.query.expand(B, 1, self.d_model)
        if self.temperature != 1.0:
            q = q / float(self.temperature)
        out, attn = self.mha(q, E, E, need_weights=True)
        emb = out.squeeze(1)
        if self.out_ln is not None:
            emb = self.out_ln(emb)
        if attn.ndim == 3:
            weights = attn.squeeze(1)
        else:
            weights = attn
        return emb, weights


def build_layer_aggregator(
    cfg: Dict[str, object],
    *,
    d_model: int,
    num_layers: int,
) -> nn.Module:
    agg_type = str(cfg.get("type", "lws_scalar"))
    use_ln = bool(cfg.get("use_layernorm", True))
    temperature = float(cfg.get("temperature", 1.0))

    if agg_type == "lws_scalar":
        return LWSScalar(d_model=d_model, num_layers=num_layers, use_layernorm=use_ln, temperature=temperature)
    if agg_type == "lws_per_dim":
        return LWSPerDim(d_model=d_model, num_layers=num_layers, use_layernorm=use_ln, temperature=temperature)
    if agg_type == "gate":
        return GateAggregator(
            d_model=d_model,
            num_layers=num_layers,
            use_layernorm=use_ln,
            gate_hidden=int(cfg.get("gate_hidden", 128)),
            gate_dropout=float(cfg.get("gate_dropout", 0.1)),
            temperature=temperature,
        )
    if agg_type == "layer_attn":
        return LayerAttention(
            d_model=d_model,
            num_layers=num_layers,
            use_layernorm=use_ln,
            attn_heads=int(cfg.get("attn_heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.1)),
            temperature=temperature,
        )
    if agg_type == "token_attn":
        return TokenAttentionPool(
            d_model=d_model,
            use_layernorm=use_ln,
            q_len=int(cfg.get("q_len", 1)),
            attn_heads=int(cfg.get("attn_heads", 4)),
            attn_dropout=float(cfg.get("attn_dropout", 0.1)),
            temperature=temperature,
        )
    raise ValueError("aggregator.type must be one of: lws_scalar|lws_per_dim|gate|layer_attn|token_attn")
