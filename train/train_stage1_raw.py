from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from core.data.splits import read_split_csv
from core.data.dataset import (
    CSVTimeSeriesDataset,
    build_split_datasets,
    build_split_datasets_from_csv_splits,
    combine_datasets_concat,
    compute_sample_weights_for_concat_dataset,
    format_dataset_summary,
    resolve_data_cfg,
)
from core.data.utils import copy_config_used, ensure_dir, resolve_device, set_seed

from core.diffusion.cond_encoder import CondBatch, CondEncoder
from core.diffusion.diffusion_schedule import DiffusionSchedule
from core.diffusion.losses_xspace import XSpaceLossCfg, acf_loss, choose_subsample_mask, fc_loss_fisherz, fft_loss
from core.diffusion.metrics_extra import compute_recon_metrics
from core.diffusion.model_dit import DiTDenoiser
from core.diffusion.sampling import compute_sample_stats, sample_ddpm
from core.diffusion.tabular_condition import TabularConditioner
from core.diffusion.viz_stage1 import save_latent_stats, save_x_recon_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 raw diffusion (DiT/DDPM) on fMRI time-series")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--ddp", action="store_true", help="Enable DDP (launch with torchrun)")
    p.add_argument("--resume_ckpt", default=None, help="Optional path to a checkpoint (.pt) to resume training from")
    return p.parse_args()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or dist.get_rank() == 0


def _unwrap_ddp(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if isinstance(m, DDP) else m


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    return cfg


def _resolve_output_dirs(output_dir: str) -> Tuple[str, str, str]:
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    viz_dir = os.path.join(output_dir, "viz")
    ensure_dir(output_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(viz_dir)
    return ckpt_dir, log_dir, viz_dir


def _write_csv_row(csv_path: str, header: list, row: Dict[str, Any]) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _seed_worker_fn(base_seed: int):
    def _fn(worker_id: int) -> None:
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return _fn


def _maybe_subset(ds, k, split_name: str):
    if k is None:
        return ds
    k = int(k)
    if k <= 0:
        return ds
    n = len(ds)
    kk = min(k, n)
    print(f"[debug] Subsetting {split_name} dataset: {kk}/{n} samples")
    return Subset(ds, list(range(kk)))


def _build_csv_datasets_with_labels(
    *,
    csv_splits: Dict[str, Dict[str, str]],
    split: str,
    seq_len: Optional[int],
    crop_mode: str,
    pad_mode: str,
    roi_dim: Optional[int],
    dataset_names: Optional[Sequence[str]],
    path_prefix: Optional[str],
    subject_col: str,
    path_col: str,
    label_table: TabularConditioner,
    strict_seq_len: bool = False,
    drop_missing_labels: bool = True,
) -> Tuple[Dict[str, CSVTimeSeriesDataset], Dict[str, int], Dict[str, int]]:
    split_key = str(split).strip().lower()
    if split_key == "valid":
        split_key = "val"
    if split_key not in csv_splits:
        raise ValueError(f"CSV splits missing key: {split_key}. Available: {list(csv_splits.keys())}")

    mapping = csv_splits[split_key]
    names = list(dataset_names) if dataset_names is not None else sorted(mapping.keys())

    per_ds: Dict[str, CSVTimeSeriesDataset] = {}
    counts: Dict[str, int] = {}
    dropped: Dict[str, int] = {}

    for ds_name in names:
        if ds_name not in mapping:
            continue
        csv_path = str(mapping[ds_name])
        rows = read_split_csv(csv_path, subject_col=str(subject_col), path_col=str(path_col))
        kept = []
        miss = 0
        if drop_missing_labels:
            for r in rows:
                if label_table.has_subject(r.subject):
                    kept.append(r)
                else:
                    miss += 1
        else:
            kept = list(rows)
            for r in rows:
                if not label_table.has_subject(r.subject):
                    miss += 1
        if len(kept) == 0:
            if _is_main_process():
                if drop_missing_labels:
                    print(f"[warn] split={split_key} dataset={ds_name} has 0 samples after label-table filtering")
                else:
                    print(f"[warn] split={split_key} dataset={ds_name} has 0 samples")
            continue
        ds = CSVTimeSeriesDataset(
            rows=kept,
            seq_len=seq_len,
            crop_mode=crop_mode,
            pad_mode=pad_mode,
            roi_dim=roi_dim,
            dataset_name=str(ds_name),
            path_prefix=path_prefix,
            strict_seq_len=bool(strict_seq_len),
        )
        per_ds[str(ds_name)] = ds
        counts[str(ds_name)] = len(kept)
        dropped[str(ds_name)] = miss

    if len(per_ds) == 0:
        raise ValueError(f"No datasets built for split={split_key} after label-table filtering")
    return per_ds, counts, dropped


def _build_split_dataset(
    *,
    split: str,
    data_cfg: Dict[str, Any],
    seq_len: int,
    crop_mode: str,
    pad_mode: str,
    roi_dim: int,
    seed: int,
    label_table: Optional[TabularConditioner],
    drop_missing_labels: bool = True,
) -> Tuple[torch.utils.data.Dataset, Dict[str, int], Optional[Dict[str, int]]]:
    csv_splits = data_cfg.get("csv_splits")
    if label_table is not None:
        if csv_splits is None:
            raise ValueError("Conditional training requires CSV splits with subject IDs")
        per_ds, counts, dropped = _build_csv_datasets_with_labels(
            csv_splits=csv_splits,
            split=split,
            seq_len=seq_len,
            crop_mode=crop_mode,
            pad_mode=pad_mode,
            roi_dim=roi_dim,
            dataset_names=data_cfg.get("csv_datasets") or data_cfg.get("datasets"),
            path_prefix=data_cfg.get("path_prefix"),
            subject_col=data_cfg.get("subject_col", "Subject"),
            path_col=data_cfg.get("path_col", "Path"),
            label_table=label_table,
            strict_seq_len=bool(data_cfg.get("strict_seq_len", False)),
            drop_missing_labels=bool(drop_missing_labels),
        )
    elif csv_splits is not None:
        per_ds, counts = build_split_datasets_from_csv_splits(
            csv_splits=csv_splits,
            split=split,
            seq_len=seq_len,
            crop_mode=crop_mode,
            pad_mode=pad_mode,
            roi_dim=roi_dim,
            dataset_names=data_cfg.get("csv_datasets") or data_cfg.get("datasets"),
            path_prefix=data_cfg.get("path_prefix"),
            subject_col=data_cfg.get("subject_col", "Subject"),
            path_col=data_cfg.get("path_col", "Path"),
            strict_seq_len=bool(data_cfg.get("strict_seq_len", False)),
        )
        dropped = None
    else:
        per_ds, counts = build_split_datasets(
            data_root=data_cfg["data_root"],
            datasets=data_cfg["datasets"],
            atlas=data_cfg["atlas"],
            split=data_cfg["splits"][split],
            seq_len=seq_len,
            crop_mode=crop_mode,
            pad_mode=pad_mode,
            roi_dim=roi_dim,
            seed=seed,
            scan_recursive=data_cfg.get("scan_recursive", True),
            strict_seq_len=bool(data_cfg.get("strict_seq_len", False)),
        )
        dropped = None

    ds = combine_datasets_concat(per_ds)
    return ds, counts, dropped


def _to_x0(x: torch.Tensor) -> torch.Tensor:
    # (B,T,N) -> (B,1,N,T)
    return x.permute(0, 2, 1).unsqueeze(1).contiguous()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))

    use_ddp = bool(cfg.get("use_ddp", False)) or bool(args.ddp)
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if use_ddp and world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_available():
            raise SystemExit("torch.distributed is not available, cannot use DDP")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://", timeout=datetime.timedelta(minutes=30))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        if _is_main_process():
            print(f"[ddp] enabled: world_size={world_size} backend={backend}")
    else:
        use_ddp = False

    output_dir = os.path.abspath(str(cfg.get("output_dir", "outputs/train")))
    ckpt_dir, log_dir, viz_dir = _resolve_output_dirs(output_dir)
    if _is_main_process():
        copy_config_used(args.config, output_dir)

    # ---- Condition / CFG ----
    cond_cfg = cfg.get("condition", {}) if isinstance(cfg.get("condition", {}), dict) else {}
    cond_enable = bool(cond_cfg.get("enable", False))
    if bool(cond_cfg.get("use_text_cond", False)):
        raise ValueError("Text conditioning is not implemented in train")

    label_table: Optional[TabularConditioner] = None
    missing_label_policy = "disabled"
    drop_missing_labels = True
    if cond_enable:
        label_table_path = str(cond_cfg.get("label_table", ""))
        if not label_table_path:
            raise ValueError("condition.enable=true requires condition.label_table")
        label_table = TabularConditioner(
            label_table_path=label_table_path,
            subject_col=str(cond_cfg.get("label_subject_col", "Subject")),
            age_col=str(cond_cfg.get("label_age_col", "age")),
            gender_col=str(cond_cfg.get("label_gender_col", "Gender")),
            dx_cols=cond_cfg.get("label_dx_cols", None),
        )
        missing_label_policy = str(cond_cfg.get("missing_label_policy", "unconditional")).strip().lower()
        if missing_label_policy not in {"drop", "unconditional"}:
            raise ValueError("condition.missing_label_policy must be 'drop' or 'unconditional'")
        drop_missing_labels = missing_label_policy == "drop"

    tabular_dim = int(cond_cfg.get("tabular_dim", 0))
    if cond_enable:
        if tabular_dim <= 0:
            tabular_dim = int(label_table.tabular_dim if label_table is not None else 0)
    else:
        tabular_dim = 0

    cond_dim = int(cond_cfg.get("cond_dim", 256))
    cond_drop_prob = float(cond_cfg.get("cond_drop_prob", 0.1)) if cond_enable else 0.0
    use_null_embedding = bool(cond_cfg.get("use_null_embedding", False))

    # ---- Data ----
    seq_len = int(cfg.get("seq_len"))
    roi_dim = int(cfg.get("roi_dim"))
    crop_mode = str(cfg.get("crop_mode", "center"))
    pad_mode = str(cfg.get("pad_mode", "zeros"))

    data_cfg = resolve_data_cfg(cfg)
    data_root = str(data_cfg["data_root"])
    datasets = list(data_cfg["datasets"])
    atlas = str(data_cfg["atlas"])
    splits = data_cfg.get("splits", cfg.get("splits", {"train": "train", "val": "val", "test": "test"}))

    train_ds, train_counts, train_dropped = _build_split_dataset(
        split="train",
        data_cfg={**data_cfg, "splits": splits},
        seq_len=seq_len,
        crop_mode=crop_mode,
        pad_mode=pad_mode,
        roi_dim=roi_dim,
        seed=seed,
        label_table=label_table,
        drop_missing_labels=drop_missing_labels,
    )
    val_ds, val_counts, val_dropped = _build_split_dataset(
        split="val",
        data_cfg={**data_cfg, "splits": splits},
        seq_len=seq_len,
        crop_mode=crop_mode,
        pad_mode=pad_mode,
        roi_dim=roi_dim,
        seed=seed,
        label_table=label_table,
        drop_missing_labels=drop_missing_labels,
    )

    train_ds = _maybe_subset(train_ds, cfg.get("max_train_samples"), "train")
    val_ds = _maybe_subset(val_ds, cfg.get("max_val_samples"), "val")

    split_counts = {"train": train_counts, "val": val_counts}
    if _is_main_process():
        summary = format_dataset_summary(
            data_root=data_root,
            datasets=datasets,
            atlas=atlas,
            splits=splits,
            split_counts=split_counts,
        )
        if train_dropped is not None or val_dropped is not None:
            stat_key = "label_table_dropped" if drop_missing_labels else "label_table_missing"
            summary += f"\n{stat_key}: "
            parts = []
            if train_dropped is not None:
                parts.append(f"train={train_dropped}")
            if val_dropped is not None:
                parts.append(f"val={val_dropped}")
            summary += ", ".join(parts)
        print(summary)
        with open(os.path.join(log_dir, "dataset_summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)

    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampling = str(data_cfg.get("dataset_sampling", "concat")).lower()
    dataset_weights = data_cfg.get("dataset_weights", None)
    if dataset_weights is not None and not isinstance(dataset_weights, dict):
        raise ValueError("cfg['dataset_weights'] must be a mapping like {HCP: 1.0, ABCD: 0.5}")

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    elif train_sampling in {"balanced", "weighted"} and not isinstance(train_ds, Subset):
        weights, _ = compute_sample_weights_for_concat_dataset(
            per_dataset_counts=split_counts["train"],
            dataset_weights=dataset_weights,
        )
        gen = torch.Generator()
        gen.manual_seed(seed)
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True, generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        worker_init_fn=_seed_worker_fn(seed),
        generator=generator,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        worker_init_fn=_seed_worker_fn(seed),
    )

    # ---- Diffusion schedule ----
    diff_cfg = cfg.get("diffusion", {})
    num_steps = int(diff_cfg.get("num_steps", 1000))
    schedule = str(diff_cfg.get("schedule", "cosine"))
    pred_type = str(diff_cfg.get("pred_type", "v"))
    if pred_type not in ("v", "eps"):
        raise ValueError("diffusion.pred_type must be 'v' or 'eps'")

    schedule_obj = DiffusionSchedule.create(num_steps=num_steps, schedule=schedule, device=device)

    # ---- DiT ----
    dit_cfg = cfg.get("dit", {})
    patch_size = int(dit_cfg.get("patch_size", 4))
    d_model = int(dit_cfg.get("d_model", 512))
    depth = int(dit_cfg.get("depth", 8))
    num_heads = int(dit_cfg.get("num_heads", 8))
    mlp_ratio = float(dit_cfg.get("mlp_ratio", 4.0))
    dropout = float(dit_cfg.get("dropout", 0.0))
    d_cond = int(dit_cfg.get("d_cond", cond_dim))
    pos_embed = str(dit_cfg.get("pos_embed", "sincos"))
    max_h = int(dit_cfg.get("max_h", 256))
    max_w = int(dit_cfg.get("max_w", 256))
    if d_cond != cond_dim:
        raise ValueError("dit.d_cond must match condition.cond_dim")
    if (roi_dim % patch_size) != 0 or (seq_len % patch_size) != 0:
        raise ValueError(
            f"patch_size={patch_size} must divide roi_dim={roi_dim} and seq_len={seq_len} for raw diffusion"
        )

    dit = DiTDenoiser(
        in_channels=1,
        patch_size=patch_size,
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        d_cond=d_cond,
        pos_embed=pos_embed,
        max_h=max_h,
        max_w=max_w,
    ).to(device)

    cond_encoder = CondEncoder(
        d_cond=d_cond,
        tabular_dim=tabular_dim,
        use_null_embedding=use_null_embedding,
    ).to(device)

    if use_ddp:
        dit = DDP(dit, device_ids=[local_rank] if device.type == "cuda" else None)
        cond_encoder = DDP(cond_encoder, device_ids=[local_rank] if device.type == "cuda" else None)

    # ---- Optim ----
    optim_cfg = cfg.get("optim", {}) if isinstance(cfg.get("optim", {}), dict) else {}
    lr = float(optim_cfg.get("lr", cfg.get("lr", 1e-4)))
    weight_decay = float(optim_cfg.get("weight_decay", cfg.get("weight_decay", 1e-4)))
    grad_clip_norm = optim_cfg.get("grad_clip_norm", cfg.get("grad_clip_norm", None))
    amp = bool(optim_cfg.get("amp", cfg.get("amp", True)))

    optimizer = torch.optim.AdamW(list(dit.parameters()) + list(cond_encoder.parameters()), lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ---- Loss config (x-space) ----
    losses_cfg = cfg.get("losses", {}) if isinstance(cfg.get("losses", {}), dict) else {}
    xcfg = XSpaceLossCfg(
        use_xspace_loss=bool(losses_cfg.get("use_xspace_loss", False)),
        lambda_fc=float(losses_cfg.get("lambda_fc", 1.0)),
        lambda_fft=float(losses_cfg.get("lambda_fft", 1.0)),
        lambda_acf=float(losses_cfg.get("lambda_acf", 0.0)),
        xspace_loss_every_k_steps=int(losses_cfg.get("xspace_loss_every_k_steps", 10)),
        xspace_loss_subsample_ratio=float(losses_cfg.get("xspace_loss_subsample_ratio", 0.25)),
        xspace_start_step=int(losses_cfg.get("xspace_start_step", 0)),
        fft_half_spectrum=bool(losses_cfg.get("fft_half_spectrum", True)),
        acf_lags=int(losses_cfg.get("acf_lags", 20)),
    )

    # ---- Logging ----
    tb_writer = None
    if _is_main_process():
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = os.path.join(log_dir, "tensorboard")
            ensure_dir(tb_dir)
            tb_writer = SummaryWriter(log_dir=tb_dir)
        except Exception as e:
            print(f"[warn] TensorBoard disabled: {e}")

    metrics_csv = os.path.join(log_dir, "metrics.csv")
    header = [
        "time",
        "split",
        "epoch",
        "global_step",
        "loss_total",
        "loss_diff",
        "loss_fc",
        "loss_fft",
        "loss_acf",
        "recon_mse",
        "recon_mae",
        "recon_corr_global",
        "recon_corr_roi_mean",
        "recon_corr_roi_num_valid",
        "sample_x_min",
        "sample_x_max",
        "sample_x_std",
        "sample_fc_mean",
        "sample_fc_var",
        "sample_fft_energy_low",
        "sample_fft_energy_mid",
        "sample_fft_energy_high",
        "lr",
    ]

    # Save run metadata
    if _is_main_process():
        meta_path = os.path.join(output_dir, "metrics.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "device": str(device),
                    "seq_len": seq_len,
                    "roi_dim": roi_dim,
                    "diffusion": {"num_steps": num_steps, "schedule": schedule, "pred_type": pred_type},
                    "dit": {
                        "patch_size": patch_size,
                        "d_model": d_model,
                        "depth": depth,
                        "num_heads": num_heads,
                        "mlp_ratio": mlp_ratio,
                        "dropout": dropout,
                        "pos_embed": pos_embed,
                        "d_cond": d_cond,
                    },
                    "condition": {
                        "enabled": cond_enable,
                        "cond_dim": d_cond,
                        "cond_drop_prob": cond_drop_prob,
                        "use_null_embedding": use_null_embedding,
                        "tabular_dim": tabular_dim,
                        "missing_label_policy": missing_label_policy,
                        "label_table": str(cond_cfg.get("label_table")) if cond_enable else None,
                        "label_dx_cols": cond_cfg.get("label_dx_cols", None),
                        "label_stats": label_table.stats.__dict__ if label_table is not None else None,
                    },
                    "losses": {
                        "use_xspace_loss": xcfg.use_xspace_loss,
                        "lambda_fc": xcfg.lambda_fc,
                        "lambda_fft": xcfg.lambda_fft,
                        "lambda_acf": xcfg.lambda_acf,
                        "xspace_loss_every_k_steps": xcfg.xspace_loss_every_k_steps,
                        "xspace_loss_subsample_ratio": xcfg.xspace_loss_subsample_ratio,
                        "xspace_start_step": xcfg.xspace_start_step,
                    },
                },
                f,
                indent=2,
                sort_keys=True,
            )

    epochs = int(cfg.get("epochs", 50))
    log_every = int(cfg.get("log_every_n_steps", 10))
    save_every_epochs = int(cfg.get("save_every_n_epochs", 1))

    global_step = 0
    best_val = float("inf")
    start_epoch = 1

    ckpt_best_name = str(cfg.get("ckpt_best_name", "best.pt"))
    ckpt_last_name = str(cfg.get("ckpt_last_name", "last.pt"))

    resume_ckpt = args.resume_ckpt if args.resume_ckpt else cfg.get("resume_ckpt", None)
    if resume_ckpt is not None:
        resume_ckpt = os.path.abspath(str(resume_ckpt))
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume_ckpt not found: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Invalid checkpoint payload (expected dict): {resume_ckpt}")

        if "dit" not in ckpt or "cond_encoder" not in ckpt:
            raise ValueError(f"Resume checkpoint missing required keys 'dit'/'cond_encoder': {resume_ckpt}")

        _unwrap_ddp(dit).load_state_dict(ckpt["dit"], strict=True)
        _unwrap_ddp(cond_encoder).load_state_dict(ckpt["cond_encoder"], strict=True)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and isinstance(ckpt["scaler"], dict):
            scaler.load_state_dict(ckpt["scaler"])

        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if start_epoch < 1:
            start_epoch = 1

        if _is_main_process():
            print(
                f"[resume] loaded: {resume_ckpt} | "
                f"start_epoch={start_epoch} global_step={global_step} best_val={best_val:.6f}"
            )

    def save_ckpt(path: str, epoch: int, is_best: bool) -> None:
        if not _is_main_process():
            return
        dit_raw = _unwrap_ddp(dit)
        cond_raw = _unwrap_ddp(cond_encoder)
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "dit": dit_raw.state_dict(),
            "cond_encoder": cond_raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
            "is_best": is_best,
        }
        torch.save(payload, path)

    def _cond_from_paths(paths: Sequence[object]) -> Tuple[Optional[CondBatch], Optional[torch.Tensor]]:
        if label_table is None:
            return None, None

        b = len(paths)
        miss_mask = torch.zeros((b,), dtype=torch.bool)
        tab = torch.zeros((b, tabular_dim), dtype=torch.float32)
        known_subjects: List[str] = []
        known_indices: List[int] = []
        for i, p in enumerate(paths):
            sid = getattr(p, "subject_id", None)
            sid_str = str(sid).strip() if sid is not None else ""
            if sid_str == "" or (not label_table.has_subject(sid_str)):
                miss_mask[i] = True
            else:
                known_subjects.append(sid_str)
                known_indices.append(i)

        if len(known_subjects) > 0:
            known_tab = label_table.encode_subjects(known_subjects)
            tab[known_indices] = known_tab

        return CondBatch(tabular=tab), miss_mask

    def forward_batch(
        x: torch.Tensor,
        *,
        train: bool,
        cond: Optional[CondBatch] = None,
        cond_missing_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        nonlocal global_step
        b = x.shape[0]
        x0 = _to_x0(x)

        # sample timesteps and noise
        t_int = torch.randint(1, num_steps + 1, (b,), device=device, dtype=torch.int64)
        eps = torch.randn_like(x0)
        xt = schedule_obj.q_sample(x0, t_int, eps)

        # condition (unconditional by default)
        cond_raw = _unwrap_ddp(cond_encoder)
        y_c = cond_raw.encode(batch_size=b, device=device, dtype=xt.dtype, cond=cond if cond_enable else None)
        if train and cond_drop_prob > 0.0:
            y_c = cond_raw.cfg_drop(y_c, drop_prob=cond_drop_prob)
        if cond_missing_mask is not None and bool(cond_missing_mask.any()):
            miss = cond_missing_mask.to(device=y_c.device, dtype=torch.bool)
            y0 = cond_raw.unconditional(b, device=y_c.device, dtype=y_c.dtype)
            y_c = y_c.clone()
            y_c[miss] = y0[miss]

        # Sanity/metrics config
        log_recon_metrics = bool(losses_cfg.get("log_recon_metrics", False))
        use_recon_metrics_in_loss = bool(losses_cfg.get("use_recon_metrics_in_loss", False))
        lambda_mse = float(losses_cfg.get("lambda_mse", 0.0))
        lambda_mae = float(losses_cfg.get("lambda_mae", 0.0))

        with torch.cuda.amp.autocast(enabled=amp):
            out = dit(xt, t_int, y_c)
            pred = out.pred

            if pred_type == "v":
                target = schedule_obj.v_from_x0_eps(x0, t_int, eps)
            else:
                target = eps
            loss_diff = torch.mean((pred - target) ** 2)

            loss_fc = torch.zeros((), device=device)
            loss_fft_v = torch.zeros((), device=device)
            loss_acf_v = torch.zeros((), device=device)

            # Reconstruct x0_hat for x-space losses and metrics.
            if pred_type == "v":
                x0_hat_full = schedule_obj.x0_from_xt_v(xt, t_int, pred)
            else:
                x0_hat_full = schedule_obj.x0_from_xt_eps(xt, t_int, pred)

            do_xspace = False
            if xcfg.use_xspace_loss and (global_step >= xcfg.xspace_start_step):
                do_xspace = True
                if xcfg.xspace_loss_every_k_steps > 1:
                    do_xspace = (global_step % xcfg.xspace_loss_every_k_steps) == 0

            need_xhat = bool(use_recon_metrics_in_loss) or bool(do_xspace) or (bool(log_recon_metrics) and (not train))
            x_hat_full = None
            if need_xhat:
                x_hat_full = x0_hat_full.squeeze(1).permute(0, 2, 1).contiguous()

            loss_total = loss_diff

            if do_xspace:
                mask = choose_subsample_mask(b, xcfg.xspace_loss_subsample_ratio, device=device)
                if mask is None:
                    xs = x
                    xh = x_hat_full
                else:
                    if not mask.any():
                        xs = None
                        xh = None
                    else:
                        xs = x[mask]
                        xh = x_hat_full[mask] if x_hat_full is not None else None

                if xs is not None and xh is not None:
                    loss_fc = fc_loss_fisherz(xs, xh)
                    loss_fft_v = fft_loss(xs, xh, half_spectrum=xcfg.fft_half_spectrum)
                    if xcfg.lambda_acf > 0.0:
                        loss_acf_v = acf_loss(xs, xh, lags=xcfg.acf_lags)

                    loss_total = loss_total + xcfg.lambda_fc * loss_fc + xcfg.lambda_fft * loss_fft_v + xcfg.lambda_acf * loss_acf_v

            recon_metrics = {}
            if log_recon_metrics and (not train) and (x_hat_full is not None):
                recon_metrics = compute_recon_metrics(x, x_hat_full)
                if use_recon_metrics_in_loss and (lambda_mse > 0.0 or lambda_mae > 0.0):
                    diff2 = torch.mean((x - x_hat_full) ** 2)
                    diff1 = torch.mean(torch.abs(x - x_hat_full))
                    loss_total = loss_total + lambda_mse * diff2 + lambda_mae * diff1

        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_diff": float(loss_diff.detach().item()),
            "loss_fc": float(loss_fc.detach().item()),
            "loss_fft": float(loss_fft_v.detach().item()),
            "loss_acf": float(loss_acf_v.detach().item()),
        }
        for k, v in recon_metrics.items():
            metrics[k] = float(v)
        return loss_total, metrics

    for epoch in range(start_epoch, epochs + 1):
        dit.train()
        cond_encoder.train()

        if use_ddp and train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        is_main = _is_main_process()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}") if is_main else train_loader
        for x, _paths in pbar:
            x = x.to(device, non_blocking=True)
            if not torch.isfinite(x).all():
                raise SystemExit("Non-finite input detected")

            optimizer.zero_grad(set_to_none=True)
            cond_batch, cond_missing_mask = _cond_from_paths(_paths)
            loss_total, m = forward_batch(x, train=True, cond=cond_batch, cond_missing_mask=cond_missing_mask)

            if not torch.isfinite(loss_total):
                raise SystemExit(f"Non-finite loss at step={global_step}")

            scaler.scale(loss_total).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(dit.parameters()) + list(cond_encoder.parameters()), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            if is_main and hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"step": global_step, "loss": f"{m['loss_total']:.4f}"})

            if is_main and global_step % log_every == 0:
                now = time.time()
                _write_csv_row(
                    metrics_csv,
                    header,
                    {
                        "time": now,
                        "split": "train",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_total": m.get("loss_total", float("nan")),
                        "loss_diff": m.get("loss_diff", float("nan")),
                        "loss_fc": m.get("loss_fc", float("nan")),
                        "loss_fft": m.get("loss_fft", float("nan")),
                        "loss_acf": m.get("loss_acf", float("nan")),
                        "recon_mse": m.get("recon_mse", None),
                        "recon_mae": m.get("recon_mae", None),
                        "recon_corr_global": m.get("recon_corr_global", None),
                        "recon_corr_roi_mean": m.get("recon_corr_roi_mean", None),
                        "recon_corr_roi_num_valid": m.get("recon_corr_roi_num_valid", None),
                        "sample_x_min": None,
                        "sample_x_max": None,
                        "sample_x_std": None,
                        "sample_fc_mean": None,
                        "sample_fc_var": None,
                        "sample_fft_energy_low": None,
                        "sample_fft_energy_mid": None,
                        "sample_fft_energy_high": None,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss_total", m["loss_total"], global_step)
                    tb_writer.add_scalar("train/loss_diff", m["loss_diff"], global_step)
                    tb_writer.add_scalar("train/loss_fc", m["loss_fc"], global_step)
                    tb_writer.add_scalar("train/loss_fft", m["loss_fft"], global_step)
                    tb_writer.add_scalar("train/loss_acf", m["loss_acf"], global_step)
                    if "recon_mse" in m:
                        tb_writer.add_scalar("train/recon_mse", m["recon_mse"], global_step)
                    if "recon_mae" in m:
                        tb_writer.add_scalar("train/recon_mae", m["recon_mae"], global_step)
                    if "recon_corr_global" in m:
                        tb_writer.add_scalar("train/recon_corr_global", m["recon_corr_global"], global_step)
                    if "recon_corr_roi_mean" in m:
                        tb_writer.add_scalar("train/recon_corr_roi_mean", m["recon_corr_roi_mean"], global_step)

            global_step += 1

        # ---- Validation ----
        if use_ddp and dist.is_initialized():
            dist.barrier()

        if _is_main_process():
            dit.eval()
            cond_encoder.eval()
            val_sums = {"loss_total": 0.0, "loss_diff": 0.0, "loss_fc": 0.0, "loss_fft": 0.0, "loss_acf": 0.0}
            val_sums.update(
                {
                    "recon_mse": 0.0,
                    "recon_mae": 0.0,
                    "recon_corr_global": 0.0,
                    "recon_corr_roi_mean": 0.0,
                    "recon_corr_roi_num_valid": 0.0,
                }
            )
            n_batches = 0

            sanity = cfg.get("sanity", {}) if isinstance(cfg.get("sanity", {}), dict) else {}
            save_val_artifacts = bool(sanity.get("save_val_artifacts", False))
            val_artifact_every_epochs = int(sanity.get("val_artifact_every_epochs", 1))
            val_artifact_batch_idx = int(sanity.get("val_artifact_batch_idx", 0))
            val_artifact_num_samples = int(sanity.get("val_artifact_num_samples", 1))
            save_latents = bool(sanity.get("save_latents", False))
            save_timeseries = bool(sanity.get("save_timeseries", True))
            roi_plot_indices = sanity.get("roi_plot_indices", [0, 100, 200])

            do_full_sampling = bool(sanity.get("do_full_sampling", False))
            sample_every_epochs = int(sanity.get("sample_every_epochs", 5))
            sample_batch_size = int(sanity.get("sample_batch_size", 1))
            sample_seed = int(sanity.get("sample_seed", 123))
            save_samples = bool(sanity.get("save_samples", True))

            val_art_dir = os.path.join(output_dir, "val_artifacts")
            samp_dir = os.path.join(output_dir, "samples")
            ensure_dir(val_art_dir)
            ensure_dir(samp_dir)

            with torch.no_grad():
                for batch_i, (x, _paths) in enumerate(tqdm(val_loader, desc=f"val epoch {epoch}/{epochs}")):
                    x = x.to(device, non_blocking=True)

                    cond_batch, cond_missing_mask = _cond_from_paths(_paths)
                    _loss_total, m = forward_batch(x, train=False, cond=cond_batch, cond_missing_mask=cond_missing_mask)
                    for k in val_sums:
                        if k in m and m[k] is not None and (not (isinstance(m[k], float) and (m[k] != m[k]))):
                            val_sums[k] += float(m[k])
                    n_batches += 1

                    do_art = (
                        save_val_artifacts
                        and (val_artifact_every_epochs > 0)
                        and (epoch % val_artifact_every_epochs == 0)
                        and (batch_i == val_artifact_batch_idx)
                    )
                    if do_art:
                        bsz = x.shape[0]
                        kk = max(1, min(int(val_artifact_num_samples), bsz))

                        x0 = _to_x0(x[:kk])
                        t_int = torch.ones((kk,), device=device, dtype=torch.int64)
                        eps = torch.randn_like(x0)
                        xt = schedule_obj.q_sample(x0, t_int, eps)
                        y0 = torch.zeros((kk, int(d_cond)), device=device, dtype=xt.dtype)
                        out = dit(xt, t_int, y0)
                        pred = out.pred
                        if pred_type == "v":
                            x0_hat = schedule_obj.x0_from_xt_v(xt, t_int, pred)
                        else:
                            x0_hat = schedule_obj.x0_from_xt_eps(xt, t_int, pred)
                        x_hat = x0_hat.squeeze(1).permute(0, 2, 1).contiguous()

                        for sid in range(kk):
                            prefix = os.path.join(val_art_dir, f"epoch{epoch:04d}_step{global_step:08d}_id{sid:02d}")
                            save_latent_stats(x0[sid], x0_hat[sid], prefix)
                            if save_latents:
                                import numpy as _np

                                _np.save(prefix + "_x0.npy", x0[sid].detach().cpu().float().numpy())
                                _np.save(prefix + "_x0_hat.npy", x0_hat[sid].detach().cpu().float().numpy())

                            save_x_recon_plots(
                                x[sid].detach().cpu(),
                                x_hat[sid].detach().cpu(),
                                prefix,
                                roi_plot_indices=roi_plot_indices,
                                save_timeseries=save_timeseries,
                            )

            val_avg = {k: (v / max(1, n_batches)) for k, v in val_sums.items()}
            now = time.time()
            _write_csv_row(
                metrics_csv,
                header,
                {
                    "time": now,
                    "split": "val",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss_total": val_avg["loss_total"],
                    "loss_diff": val_avg["loss_diff"],
                    "loss_fc": val_avg["loss_fc"],
                    "loss_fft": val_avg["loss_fft"],
                    "loss_acf": val_avg["loss_acf"],
                    "recon_mse": val_avg.get("recon_mse", None),
                    "recon_mae": val_avg.get("recon_mae", None),
                    "recon_corr_global": val_avg.get("recon_corr_global", None),
                    "recon_corr_roi_mean": val_avg.get("recon_corr_roi_mean", None),
                    "recon_corr_roi_num_valid": val_avg.get("recon_corr_roi_num_valid", None),
                    "sample_x_min": None,
                    "sample_x_max": None,
                    "sample_x_std": None,
                    "sample_fc_mean": None,
                    "sample_fc_var": None,
                    "sample_fft_energy_low": None,
                    "sample_fft_energy_mid": None,
                    "sample_fft_energy_high": None,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
            )
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss_total", val_avg["loss_total"], epoch)
                tb_writer.add_scalar("val/loss_diff", val_avg["loss_diff"], epoch)
                tb_writer.add_scalar("val/loss_fc", val_avg["loss_fc"], epoch)
                tb_writer.add_scalar("val/loss_fft", val_avg["loss_fft"], epoch)
                tb_writer.add_scalar("val/loss_acf", val_avg["loss_acf"], epoch)
                if "recon_mse" in val_avg:
                    tb_writer.add_scalar("val/recon_mse", val_avg["recon_mse"], epoch)
                if "recon_mae" in val_avg:
                    tb_writer.add_scalar("val/recon_mae", val_avg["recon_mae"], epoch)
                if "recon_corr_global" in val_avg:
                    tb_writer.add_scalar("val/recon_corr_global", val_avg["recon_corr_global"], epoch)
                if "recon_corr_roi_mean" in val_avg:
                    tb_writer.add_scalar("val/recon_corr_roi_mean", val_avg["recon_corr_roi_mean"], epoch)

            if do_full_sampling and (sample_every_epochs > 0) and (epoch % sample_every_epochs == 0):
                cond_raw = _unwrap_ddp(cond_encoder)
                y0 = cond_raw.unconditional(sample_batch_size, device=device, dtype=torch.float32)
                x0_samp = sample_ddpm(
                    denoiser=dit,
                    schedule=schedule_obj,
                    pred_type=pred_type,
                    shape=(sample_batch_size, 1, roi_dim, seq_len),
                    device=device,
                    cond=y0,
                    seed=sample_seed,
                )
                x_samp = x0_samp.squeeze(1).permute(0, 2, 1).contiguous()

                prefix = os.path.join(samp_dir, f"epoch{epoch:04d}_step{global_step:08d}_id00")
                save_x_recon_plots(
                    x_samp[0].detach().cpu(),
                    x_samp[0].detach().cpu(),
                    prefix + "_sample",
                    roi_plot_indices=None,
                    save_timeseries=False,
                )

                if save_samples:
                    import numpy as _np

                    _np.save(prefix + "_x_sample.npy", x_samp.detach().cpu().float().numpy())

                samp_stats = compute_sample_stats(x_samp)
                _write_csv_row(
                    metrics_csv,
                    header,
                    {
                        "time": time.time(),
                        "split": "sample",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_total": None,
                        "loss_diff": None,
                        "loss_fc": None,
                        "loss_fft": None,
                        "loss_acf": None,
                        "recon_mse": None,
                        "recon_mae": None,
                        "recon_corr_global": None,
                        "recon_corr_roi_mean": None,
                        "recon_corr_roi_num_valid": None,
                        "sample_x_min": samp_stats.get("sample_x_min", None),
                        "sample_x_max": samp_stats.get("sample_x_max", None),
                        "sample_x_std": samp_stats.get("sample_x_std", None),
                        "sample_fc_mean": samp_stats.get("sample_fc_mean", None),
                        "sample_fc_var": samp_stats.get("sample_fc_var", None),
                        "sample_fft_energy_low": samp_stats.get("sample_fft_energy_low", None),
                        "sample_fft_energy_mid": samp_stats.get("sample_fft_energy_mid", None),
                        "sample_fft_energy_high": samp_stats.get("sample_fft_energy_high", None),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("sample/x_std", samp_stats["sample_x_std"], epoch)
                    tb_writer.add_scalar("sample/fc_mean", samp_stats["sample_fc_mean"], epoch)
                    tb_writer.add_scalar("sample/fft_energy_low", samp_stats["sample_fft_energy_low"], epoch)
                    tb_writer.add_scalar("sample/fft_energy_mid", samp_stats["sample_fft_energy_mid"], epoch)
                    tb_writer.add_scalar("sample/fft_energy_high", samp_stats["sample_fft_energy_high"], epoch)

            if epoch % save_every_epochs == 0:
                save_ckpt(os.path.join(ckpt_dir, ckpt_last_name), epoch, is_best=False)
            if val_avg["loss_total"] < best_val:
                best_val = val_avg["loss_total"]
                save_ckpt(os.path.join(ckpt_dir, ckpt_best_name), epoch, is_best=True)

        if use_ddp and dist.is_initialized():
            dist.barrier()

    if tb_writer is not None:
        tb_writer.close()

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

    print(f"Done. Output saved under: {output_dir}")


if __name__ == "__main__":
    main()
