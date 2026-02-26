from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import os

import torch
from matplotlib import pyplot as plt

from ._path import ensure_repo_root
from .noise_utils import noise_for_batch

ensure_repo_root()

from core.diffusion.cond_encoder import CondEncoder
from core.diffusion.diffusion_schedule import DiffusionSchedule
from core.diffusion.model_dit import DiTDenoiser


def load_stage1_raw(stage1_ckpt_path: str, device: torch.device) -> Tuple[DiTDenoiser, CondEncoder, DiffusionSchedule, Dict[str, Any]]:
    ckpt = torch.load(stage1_ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "dit" not in ckpt or "cond_encoder" not in ckpt or "config" not in ckpt:
        raise ValueError("Stage-1 raw checkpoint must contain keys: 'dit', 'cond_encoder', 'config'")
    cfg1 = ckpt["config"]

    diff_cfg = cfg1.get("diffusion", {}) if isinstance(cfg1.get("diffusion", {}), dict) else {}
    num_steps = int(diff_cfg.get("num_steps", 1000))
    schedule = str(diff_cfg.get("schedule", "cosine"))
    pred_type = str(diff_cfg.get("pred_type", "v"))
    if pred_type not in ("v", "eps"):
        raise ValueError("stage1 diffusion.pred_type must be 'v' or 'eps'")

    schedule_obj = DiffusionSchedule.create(num_steps=num_steps, schedule=schedule, device=device)

    dit_cfg = cfg1.get("dit", {}) if isinstance(cfg1.get("dit", {}), dict) else {}
    cond_cfg = cfg1.get("condition", {}) if isinstance(cfg1.get("condition", {}), dict) else {}

    head_w = ckpt["dit"]["head.weight"]
    d_model = int(head_w.shape[1])
    patch_size = int(dit_cfg.get("patch_size", 4))
    in_channels = int(head_w.shape[0] // (patch_size * patch_size))

    dit = DiTDenoiser(
        in_channels=in_channels,
        patch_size=patch_size,
        d_model=int(dit_cfg.get("d_model", d_model)),
        depth=int(dit_cfg.get("depth", 8)),
        num_heads=int(dit_cfg.get("num_heads", 8)),
        mlp_ratio=float(dit_cfg.get("mlp_ratio", 4.0)),
        dropout=float(dit_cfg.get("dropout", 0.0)),
        d_cond=int(cond_cfg.get("cond_dim", dit_cfg.get("d_cond", 256))),
        pos_embed=str(dit_cfg.get("pos_embed", "sincos")),
        max_h=int(dit_cfg.get("max_h", 256)),
        max_w=int(dit_cfg.get("max_w", 256)),
    ).to(device)
    dit.load_state_dict(ckpt["dit"], strict=True)

    # Some older stage1 checkpoints save config.condition.tabular_dim=0 even when
    # cond_encoder state_dict contains tabular_net.* weights.
    tabular_dim = int(cond_cfg.get("tabular_dim", 0))
    cond_state = ckpt.get("cond_encoder", {})
    if tabular_dim <= 0 and isinstance(cond_state, dict) and "tabular_net.0.weight" in cond_state:
        try:
            tabular_dim = int(cond_state["tabular_net.0.weight"].shape[1])
            print(f"[info] inferred condition.tabular_dim={tabular_dim} from checkpoint weights")
        except Exception:
            pass

    cond_encoder = CondEncoder(
        d_cond=int(cond_cfg.get("cond_dim", dit_cfg.get("d_cond", 256))),
        tabular_dim=tabular_dim,
        use_null_embedding=bool(cond_cfg.get("use_null_embedding", True)),
    ).to(device)
    cond_encoder.load_state_dict(ckpt["cond_encoder"], strict=True)

    dit.eval()
    cond_encoder.eval()
    for p in dit.parameters():
        p.requires_grad = False
    for p in cond_encoder.parameters():
        p.requires_grad = False

    return dit, cond_encoder, schedule_obj, cfg1


def resolve_capture_layer(dit: DiTDenoiser, capture_layer: int) -> int:
    depth = int(getattr(dit, "depth"))
    cl = int(capture_layer)
    if cl < 0:
        cl = depth + cl
    if cl < 0 or cl >= depth:
        raise ValueError(f"capture_layer out of range: got {capture_layer}, depth={depth}")
    return cl


@dataclass
class EmbedConfig:
    timestep: int
    capture_layer: int
    pool: str = "mean"


class Stage3RawFeatureExtractor:
    def __init__(
        self,
        *,
        dit: DiTDenoiser,
        cond_encoder: CondEncoder,
        schedule: DiffusionSchedule,
        embed_cfg: EmbedConfig,
        device: torch.device,
        noise_seed: int = 0,
        pred_type: str = "v",
    ) -> None:
        self.dit = dit
        self.cond_encoder = cond_encoder
        self.schedule = schedule
        self.embed_cfg = embed_cfg
        self.device = device

        self.noise_seed = int(noise_seed)

        pred_type = str(pred_type)
        if pred_type not in ("v", "eps"):
            raise ValueError("pred_type must be 'v' or 'eps'")
        self.pred_type = pred_type

        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(self.noise_seed)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.embed_cfg.pool == "mean":
            return torch.mean(tokens, dim=1)
        raise ValueError(f"Unknown pool: {self.embed_cfg.pool}")

    @staticmethod
    def _to_x0(x: torch.Tensor) -> torch.Tensor:
        # (B,T,N) -> (B,1,N,T)
        return x.permute(0, 2, 1).unsqueeze(1).contiguous()

    @staticmethod
    def _to_x(x0: torch.Tensor) -> torch.Tensor:
        # (B,1,N,T) -> (B,T,N)
        return x0.squeeze(1).permute(0, 2, 1).contiguous()

    def extract_features(
        self,
        x: torch.Tensor,
        *,
        enable_grad: bool = False,
        debug: bool = False,
        debug_outdir: Optional[str] = None,
        noise_ids: Optional[Sequence[str]] = None,
        noise_mode: str = "per_subject",
    ) -> torch.Tensor:
        """Extract pooled token features from raw fMRI inputs."""

        x = x.to(device=self.device)
        x0 = self._to_x0(x)

        b = int(x0.shape[0])
        t = int(self.embed_cfg.timestep)
        t_vec = torch.full((b,), t, device=self.device, dtype=torch.int64)

        if noise_ids is None:
            eps = torch.randn(x0.shape, device=self.device, dtype=x0.dtype, generator=self._gen)
        else:
            if len(noise_ids) != b:
                raise ValueError(f"noise_ids length must match batch size: got {len(noise_ids)} vs {b}")
            eps = noise_for_batch(
                mode=str(noise_mode),
                subjects=[str(s) for s in noise_ids],
                sample_indices=None,
                global_seed=self.noise_seed,
                shape_per_sample=tuple(x0.shape[1:]),
                device=self.device,
                dtype=x0.dtype,
            )

        xt = self.schedule.q_sample(x0, t_vec, eps)

        y_c = self.cond_encoder.encode(batch_size=b, device=self.device, dtype=xt.dtype, cond=None)

        if enable_grad:
            out = self.dit(xt, t_vec, y_c, return_hiddens=True, capture_layers=[int(self.embed_cfg.capture_layer)])
        else:
            with torch.no_grad():
                out = self.dit(xt, t_vec, y_c, return_hiddens=True, capture_layers=[int(self.embed_cfg.capture_layer)])

        pred_hat = out.pred
        if self.pred_type == "v":
            x0_hat = self.schedule.x0_from_xt_v(xt, t_vec, pred_hat)
        else:
            x0_hat = self.schedule.x0_from_xt_eps(xt, t_vec, pred_hat)

        if debug and debug_outdir is not None:
            import numpy as _np

            os.makedirs(debug_outdir, exist_ok=True)
            try:
                _np.save(os.path.join(debug_outdir, "x_gt.npy"), x.detach().cpu().numpy())
                _np.save(os.path.join(debug_outdir, "x0.npy"), x0.detach().cpu().numpy())
                _np.save(os.path.join(debug_outdir, "xt.npy"), xt.detach().cpu().numpy())
                _np.save(os.path.join(debug_outdir, "x0_hat.npy"), x0_hat.detach().cpu().numpy())
            except Exception as e:
                print(f"[debug][warn] failed saving npy: {e}")

            try:
                idx = 0
                x_gt = x.detach().cpu().numpy()[idx]
                x_noisy = self._to_x(xt).detach().cpu().numpy()[idx]
                x_dn = self._to_x(x0_hat).detach().cpu().numpy()[idx]

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(x_gt.T, aspect="auto")
                axs[0].set_title("x (ground truth)")
                axs[1].imshow(x_noisy.T, aspect="auto")
                axs[1].set_title("x_t (noisy)")
                axs[2].imshow(x_dn.T, aspect="auto")
                axs[2].set_title("x0_hat (denoised)")
                plt.tight_layout()
                out_png = os.path.join(debug_outdir, "recon_heatmaps.png")
                plt.savefig(out_png, dpi=150)
                plt.close(fig)
                print(f"[debug] saved {out_png}")
            except Exception as e:
                print(f"[debug][warn] failed plotting heatmaps: {e}")

        if out.hiddens is None or int(self.embed_cfg.capture_layer) not in out.hiddens:
            raise RuntimeError("Requested hidden layer not captured")
        tokens = out.hiddens[int(self.embed_cfg.capture_layer)]
        feat = self._pool_tokens(tokens)
        return feat


__all__ = [
    "EmbedConfig",
    "Stage3RawFeatureExtractor",
    "load_stage1_raw",
    "resolve_capture_layer",
]
