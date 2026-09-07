from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from downstream._path import ensure_repo_root

ensure_repo_root()

from downstream.datasets import (
    DownstreamCSVDataset,
    join_split_with_labels,
    read_labels_csv,
    read_split_csv,
    write_missing_labels_csv,
)
from downstream.aggregators import build_layer_aggregator
from downstream.feature_extractor_raw import load_stage1_raw
from downstream.feature_protocol_raw import FeatureProtocolRaw, FeatureProtocolRawConfig, resolve_capture_layers
from downstream.heads import HeadSpec, build_head
from downstream.lora import LoRAConfig, apply_lora_to_last_k_blocks
from downstream.metrics import confusion_matrix, evaluate_classification, evaluate_regression
from downstream.plots import (
    compute_roc_curve_binary,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_regression_scatter,
    plot_roc_curve,
)
from downstream.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 raw general: unified downstream training")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--ddp", action="store_true", help="Enable DDP (launch with torchrun)")
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


def _rank() -> int:
    return dist.get_rank() if _is_dist_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist_initialized() else 1


def _is_main_process() -> bool:
    return (not _is_dist_initialized()) or _rank() == 0


def _unwrap_module(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


class _DiTForwardDictWrapper(nn.Module):
    """Make DiT forward return DDP-traversable outputs."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core
        if hasattr(core, "depth"):
            self.depth = getattr(core, "depth")
        if hasattr(core, "d_model"):
            self.d_model = getattr(core, "d_model")

    def forward(self, *args, **kwargs):
        out = self.core(*args, **kwargs)
        pred = getattr(out, "pred", None)
        hiddens = getattr(out, "hiddens", None)
        if kwargs.get("return_hiddens", False):
            return {"pred": pred, "hiddens": (hiddens if hiddens is not None else {})}
        return {"pred": pred}


class _IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        item["index"] = int(idx)
        return item


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_dataset(
    cfg: Dict[str, Any],
    split: str,
    *,
    label_stats: Optional[Tuple[float, float]] = None,
) -> Tuple[Dataset, List[Any]]:
    data_cfg = cfg["data"]
    label_cfg = cfg["label"]
    splits_cfg = data_cfg.get("splits", {})
    if split not in splits_cfg or "csv" not in splits_cfg[split]:
        raise ValueError(f"data.splits.{split}.csv is required")
    split_csv = str(splits_cfg[split]["csv"])

    split_rows = read_split_csv(
        split_csv,
        subject_col=str(data_cfg.get("subject_col", "Subject")),
        path_col=str(data_cfg.get("path_col", "Path")),
    )

    labels = read_labels_csv(
        str(label_cfg["label_csv_path"]),
        subject_col=str(label_cfg.get("subject_col", "Subject")),
        label_col=str(label_cfg["target_col"]),
        label_map=label_cfg.get("label_map", None),
    )
    if label_stats is not None:
        mean, std = label_stats
        labels = {k: (float(v) - mean) / std for k, v in labels.items()}

    keep, missing = join_split_with_labels(split_rows, labels, split_name=split)

    ds = DownstreamCSVDataset(
        rows=keep,
        labels=labels,
        seq_len=int(data_cfg.get("seq_len", 200)) if data_cfg.get("seq_len") is not None else None,
        crop_mode=str(data_cfg.get("crop_mode", "center")),
        pad_mode=str(data_cfg.get("pad_mode", "zeros")),
        roi_dim=int(data_cfg.get("roi_dim")) if data_cfg.get("roi_dim") is not None else None,
        path_prefix=str(data_cfg.get("path_prefix")) if data_cfg.get("path_prefix") is not None else None,
        strict_seq_len=bool(data_cfg.get("strict_seq_len", False)),
    )
    return _IndexedDataset(ds), missing


def _compute_label_zscore_stats(cfg: Dict[str, Any]) -> Tuple[float, float]:
    data_cfg = cfg["data"]
    label_cfg = cfg["label"]
    splits_cfg = data_cfg.get("splits", {})
    if "train" not in splits_cfg or "csv" not in splits_cfg["train"]:
        raise ValueError("data.splits.train.csv is required for label z-score")
    split_csv = str(splits_cfg["train"]["csv"])

    split_rows = read_split_csv(
        split_csv,
        subject_col=str(data_cfg.get("subject_col", "Subject")),
        path_col=str(data_cfg.get("path_col", "Path")),
    )
    labels = read_labels_csv(
        str(label_cfg["label_csv_path"]),
        subject_col=str(label_cfg.get("subject_col", "Subject")),
        label_col=str(label_cfg["target_col"]),
        label_map=label_cfg.get("label_map", None),
    )
    keep, _missing = join_split_with_labels(split_rows, labels, split_name="train")
    if len(keep) == 0:
        raise ValueError("No labels found for train split; cannot compute z-score")
    values = np.array([labels[r.subject] for r in keep], dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Non-finite label stats for z-score")
    if std < 1.0e-8:
        std = 1.0
    return mean, std


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    num_classes: int


def _task_from_cfg(cfg: Dict[str, Any]) -> TaskSpec:
    label_cfg = cfg["label"]
    task_type = str(label_cfg.get("task_type", "classification"))
    if task_type not in ("classification", "regression"):
        raise ValueError("label.task_type must be classification or regression")

    if task_type == "regression":
        return TaskSpec(task_type=task_type, num_classes=1)

    num_classes = int(label_cfg.get("num_classes", 0))
    if num_classes <= 1:
        raise ValueError("label.num_classes must be set >= 2 for classification")
    return TaskSpec(task_type=task_type, num_classes=num_classes)


def _count_trainable(params: Sequence[torch.nn.Parameter]) -> Tuple[int, float]:
    n = int(sum(p.numel() for p in params if p.requires_grad))
    mb = (n * 4) / (1024.0**2)
    return n, mb


def _primary_metric(task: TaskSpec, best_metric_name: str, metrics: Dict[str, float]) -> float:
    name = str(best_metric_name)
    if name in metrics:
        return float(metrics[name])
    if task.task_type == "regression":
        return float(metrics.get("pearson", 0.0))
    return float(metrics.get("f1_weighted", 0.0))


def _weights_for_log(w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if w is None:
        return None
    if w.ndim == 2:
        return w.mean(dim=0)
    if w.ndim == 1:
        return w
    if w.ndim == 3:
        return w.mean(dim=(0, 1))
    raise ValueError(f"Unexpected weights shape: {tuple(w.shape)}")


def _format_weights_for_tqdm(w: Optional[torch.Tensor], *, max_len: int = 6) -> str:
    w1 = _weights_for_log(w)
    if w1 is None:
        return "[]"
    ww = w1.detach().cpu().float().tolist()
    s = ",".join(f"{x:.2f}" for x in ww[:max_len])
    if len(ww) > max_len:
        s += ",..."
    return f"[{s}]"


def _parse_t_list(emb_cfg: Dict[str, Any]) -> List[int]:
    t_list_cfg = emb_cfg.get("t_list", None)
    if t_list_cfg is None:
        t_list = [int(emb_cfg.get("timestep", 10))]
    elif isinstance(t_list_cfg, str):
        t_list = [int(x) for x in t_list_cfg.split(",") if x.strip() != ""]
    elif isinstance(t_list_cfg, (int, float)):
        t_list = [int(t_list_cfg)]
    else:
        t_list = [int(x) for x in list(t_list_cfg)]
    if len(t_list) == 0:
        t_list = [int(emb_cfg.get("timestep", 10))]
    return t_list


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    # ---- Optional DDP ----
    use_ddp = bool(cfg.get("train", {}).get("use_ddp", False)) or bool(args.ddp)
    env_world = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    if env_world > 1:
        use_ddp = True

    if use_ddp and env_world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_available():
            raise SystemExit("torch.distributed is not available, cannot use DDP")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        atexit.register(lambda: dist.destroy_process_group() if dist.is_initialized() else None)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    mode = str(cfg.get("mode", "linear_probe"))
    if mode not in ("linear_probe", "lora", "full_finetune"):
        raise ValueError("mode must be one of: linear_probe|lora|full_finetune")

    base_seed = int(cfg.get("train", {}).get("seed", 0))
    _set_seed(base_seed + int(_rank()))

    # Device
    device_str = str(cfg.get("train", {}).get("device", cfg.get("device", "auto")))
    if use_ddp and torch.cuda.is_available() and env_world > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        if device.type == "cuda":
            n_cuda = int(torch.cuda.device_count())
            idx = device.index if device.index is not None else 0
            if n_cuda > 0 and int(idx) >= n_cuda:
                if _is_main_process():
                    print(f"[warn] device {device} invalid with cuda.device_count()={n_cuda}; remap to cuda:0")
                device = torch.device("cuda:0")

    # Output dirs
    out_cfg = cfg.get("output", {})
    out_root = os.path.abspath(str(out_cfg.get("out_root", "checkpoints_raw_general")))
    data_cfg = cfg["data"]
    label_cfg = cfg["label"]
    dataset_name = str(data_cfg.get("dataset", data_cfg.get("dataset_name", "DATA")))
    task_name = str(label_cfg.get("target_col", "task"))
    fold = int(data_cfg.get("fold", data_cfg.get("fold_id", 0)))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if use_ddp and _is_dist_initialized() and _world_size() > 1:
        obj = [run_id if _is_main_process() else None]
        dist.broadcast_object_list(obj, src=0)
        run_id = str(obj[0])
    out_dir = os.path.join(out_root, dataset_name, task_name, mode, f"fold{fold}", run_id)
    if _is_main_process():
        ensure_dir(out_dir)
        ensure_dir(os.path.join(out_dir, "plots"))
    if use_ddp and _is_dist_initialized():
        dist.barrier()

    # Models
    ckpt_cfg = cfg.get("ckpt", cfg.get("checkpoints", {}))
    stage1_ckpt = ckpt_cfg.get("stage1_ckpt", ckpt_cfg.get("stage1_ckpt_path", None))
    if stage1_ckpt is None:
        raise KeyError("ckpt.stage1_ckpt (or stage1_ckpt_path) is required")
    dit, cond, schedule, cfg1 = load_stage1_raw(str(stage1_ckpt), device=device)

    # unconditional cond
    cond.eval()
    for p in cond.parameters():
        p.requires_grad = False

    # Mode-specific trainability
    if mode == "linear_probe":
        dit.eval()
        for p in dit.parameters():
            p.requires_grad = False
    elif mode == "lora":
        lora_cfg = cfg.get("lora", {})
        if not bool(lora_cfg.get("enable", True)):
            raise ValueError("mode=lora requires lora.enable=true")
        lc = LoRAConfig(
            r=int(lora_cfg.get("r", 8)),
            alpha=float(lora_cfg.get("alpha", 16.0)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
        )
        replaced = apply_lora_to_last_k_blocks(dit, last_k=int(lora_cfg.get("target_blocks", 2)), cfg=lc)
        print(f"[lora] replaced blocks={replaced}")

        for p in dit.parameters():
            p.requires_grad = False
        for name, p in dit.named_parameters():
            if "q_A" in name or "q_B" in name or "v_A" in name or "v_B" in name:
                p.requires_grad = True
        dit.train()
    else:
        dit.train()
        for p in dit.parameters():
            p.requires_grad = True

    # Task + head
    task = _task_from_cfg(cfg)
    head_spec = HeadSpec(task_type=task.task_type, num_classes=int(task.num_classes))
    d_model = int(getattr(dit, "d_model"))
    head_cfg = cfg.get("head", {})
    hidden_dim = None
    dropout = float(head_cfg.get("mlp_dropout", 0.1))
    if mode == "full_finetune":
        hidden_dim = int(head_cfg.get("mlp_hidden", d_model))
    head = build_head(d_model=d_model, task=head_spec, hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Feature protocol (with t_list)
    emb_cfg = cfg.get("embedding", {})
    t_list = _parse_t_list(emb_cfg)
    timestep_default = int(emb_cfg.get("timestep", 10))
    capture_layers_raw = list(emb_cfg.get("capture_layers", [-1]))
    capture_layers = resolve_capture_layers(int(getattr(dit, "depth")), capture_layers_raw)
    fp_cfg = FeatureProtocolRawConfig(
        timestep=timestep_default,
        capture_layers=capture_layers,
        noise_mode=str(emb_cfg.get("noise_mode", "per_subject")),  # type: ignore[arg-type]
        noise_seed=int(emb_cfg.get("noise_seed", 0)),
    )
    protocol = FeatureProtocolRaw(dit=dit, cond_encoder=cond, schedule=schedule, cfg=fp_cfg, device=device)

    # Aggregator
    agg_cfg = cfg.get("aggregator", {})
    agg_type = str(agg_cfg.get("type", "lws_scalar"))
    num_layers_for_agg = len(capture_layers) * len(t_list)
    aggregator = build_layer_aggregator(agg_cfg, d_model=d_model, num_layers=num_layers_for_agg).to(device)

    # ---- DDP wrap ----
    if use_ddp and _is_dist_initialized() and _world_size() > 1:
        ddp_kwargs = {"device_ids": [local_rank] if device.type == "cuda" else None}
        if any(p.requires_grad for p in dit.parameters()):
            dit = _DiTForwardDictWrapper(dit)
            dit = DDP(dit, **ddp_kwargs, find_unused_parameters=True)
            protocol.dit = dit
        aggregator = DDP(aggregator, **ddp_kwargs, find_unused_parameters=False)
        head = DDP(head, **ddp_kwargs, find_unused_parameters=False)

    # Data
    label_stats = None
    if bool(label_cfg.get("zscore", False)):
        if task.task_type != "regression":
            if _is_main_process():
                print("[warn] label.zscore is only for regression; ignoring.")
        else:
            label_stats = _compute_label_zscore_stats(cfg)
            if _is_main_process():
                mean, std = label_stats
                print(f"[label] zscore using train split: mean={mean:.6f} std={std:.6f}")

    ds_tr, miss_tr = _build_dataset(cfg, "train", label_stats=label_stats)
    ds_va, miss_va = _build_dataset(cfg, "valid", label_stats=label_stats)
    ds_te, miss_te = _build_dataset(cfg, "test", label_stats=label_stats)

    all_missing = list(miss_tr) + list(miss_va) + list(miss_te)
    if len(all_missing) > 0:
        miss_path = os.path.join(out_dir, "missing_labels.csv")
        if _is_main_process():
            write_missing_labels_csv(all_missing, miss_path)
            print(f"[warn] missing labels: {len(all_missing)} -> {miss_path}")
        if use_ddp and _is_dist_initialized():
            dist.barrier()

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_sampler = None
    valid_sampler = None
    test_sampler = None
    if use_ddp and _is_dist_initialized() and _world_size() > 1:
        train_sampler = DistributedSampler(ds_tr, num_replicas=_world_size(), rank=_rank(), shuffle=True, drop_last=False)
        valid_sampler = DistributedSampler(ds_va, num_replicas=_world_size(), rank=_rank(), shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(ds_te, num_replicas=_world_size(), rank=_rank(), shuffle=False, drop_last=False)

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Loss
    if task.task_type == "regression":
        loss_fn = nn.MSELoss()
    else:
        tr_cfg = cfg.get("train", {})
        use_class_weights = bool(tr_cfg.get("use_class_weights", True))
        if use_class_weights:
            counts = torch.zeros((int(task.num_classes),), dtype=torch.float32)
            for b in tqdm(train_eval_loader, desc="count_classes", ncols=120):
                yy = b["y"].long().view(-1)
                counts += torch.bincount(yy.cpu(), minlength=int(task.num_classes)).float()
            total = float(torch.sum(counts).item())
            denom = (counts * float(task.num_classes)).clamp_min(1.0)
            w = total / denom
            w[counts == 0] = 0.0
            class_weights = w.to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    opt_cfg = cfg.get("optim", {})
    lr_dit = float(opt_cfg.get("lr_dit", 1e-5))
    lr_head = float(opt_cfg.get("lr_head", 1e-4))
    lr_agg = float(opt_cfg.get("lr_agg", 1e-4))
    wd_dit = float(opt_cfg.get("weight_decay_dit", 1e-4))
    wd_head = float(opt_cfg.get("weight_decay_head", 0.0))
    wd_agg = float(opt_cfg.get("weight_decay_agg", 0.0))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    params: List[Dict[str, Any]] = []
    dit_params = [p for p in dit.parameters() if p.requires_grad]
    if len(dit_params) > 0:
        params.append({"params": dit_params, "lr": lr_dit, "weight_decay": wd_dit})
    params.append({"params": aggregator.parameters(), "lr": lr_agg, "weight_decay": wd_agg})
    params.append({"params": head.parameters(), "lr": lr_head, "weight_decay": wd_head})

    optimizer = torch.optim.AdamW(params, betas=betas, eps=eps)

    # Training knobs
    tr_cfg = cfg.get("train", {})
    use_amp = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = float(tr_cfg.get("grad_clip_norm", 0.0))
    epochs = int(tr_cfg.get("epochs", 20))
    patience = int(tr_cfg.get("patience", 10))
    warmup_ratio = float(tr_cfg.get("warmup_ratio", 0.05))
    best_metric_name = str(tr_cfg.get("best_metric", "f1_weighted" if task.task_type == "classification" else "pearson"))

    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)

    def _resolve_eval_after(val: Any) -> Tuple[bool, bool, bool]:
        if val is None:
            return True, True, True
        if isinstance(val, dict):
            train_on = bool(val.get("train", True))
            valid_on = bool(val.get("valid", val.get("val", True)))
            test_on = bool(val.get("test", True))
            return train_on, valid_on, test_on
        if isinstance(val, str):
            items = [v.strip().lower() for v in val.split(",") if v.strip()]
            names = set(items)
        elif isinstance(val, (list, tuple, set)):
            names = {str(v).strip().lower() for v in val if str(v).strip()}
        else:
            return True, True, True
        return "train" in names, ("valid" in names or "val" in names), "test" in names

    eval_train_after, eval_valid_after, eval_test_after = _resolve_eval_after(cfg.get("eval_after", None))
    epoch_eval_cfg = cfg.get("epoch_eval", {})
    if not isinstance(epoch_eval_cfg, dict):
        epoch_eval_cfg = {}
    eval_test_each_epoch = bool(epoch_eval_cfg.get("test_each_epoch", False))

    def _metric_delta(
        loaded_metrics: Dict[str, float],
        epoch_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in sorted(set(loaded_metrics.keys()) & set(epoch_metrics.keys())):
            try:
                out[k] = float(loaded_metrics[k]) - float(epoch_metrics[k])
            except Exception:
                continue
        return out

    def _gather_and_dedupe_eval(
        yt: torch.Tensor,
        ot: torch.Tensor,
        it: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (use_ddp and _is_dist_initialized() and _world_size() > 1):
            return yt, ot

        payload = {
            "idx": it.detach().cpu().numpy(),
            "y": yt.detach().cpu().numpy(),
            "o": ot.detach().cpu().numpy(),
        }
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(_world_size())]
        dist.all_gather_object(gathered, payload)

        idx_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        o_parts: List[np.ndarray] = []
        for g in gathered:
            if not isinstance(g, dict):
                continue
            gi = g.get("idx", None)
            gy = g.get("y", None)
            go = g.get("o", None)
            if gi is None or gy is None or go is None:
                continue
            idx_parts.append(np.asarray(gi).reshape(-1))
            y_parts.append(np.asarray(gy))
            o_parts.append(np.asarray(go))

        if len(idx_parts) == 0:
            return yt, ot

        idx_all = np.concatenate(idx_parts, axis=0).astype(np.int64, copy=False)
        y_all = np.concatenate(y_parts, axis=0)
        o_all = np.concatenate(o_parts, axis=0)

        # DistributedSampler may pad by repeating samples; keep one row per original dataset index.
        first_pos: Dict[int, int] = {}
        for pos, idx in enumerate(idx_all.tolist()):
            if idx not in first_pos:
                first_pos[idx] = pos
        keep_pos = [first_pos[k] for k in sorted(first_pos.keys())]

        y_keep = y_all[keep_pos]
        o_keep = o_all[keep_pos]
        yt_out = torch.from_numpy(np.asarray(y_keep))
        ot_out = torch.from_numpy(np.asarray(o_keep))
        return yt_out, ot_out

    def _lr_scale(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    def _apply_lr_scale(step: int) -> None:
        scale = _lr_scale(step)
        for g in optimizer.param_groups:
            base_lr = g.get("initial_lr", None)
            if base_lr is None:
                g["initial_lr"] = float(g["lr"])
                base_lr = g["initial_lr"]
            g["lr"] = float(base_lr) * scale

    # Logging
    logs_path = os.path.join(out_dir, "logs.csv")
    log_fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_f1_weighted",
        "val_pearson",
        "val_r2",
    ]
    if eval_test_each_epoch:
        log_fields.extend(
            [
                "test_loss",
                "test_acc",
                "test_f1_weighted",
                "test_auroc",
                "test_mse",
                "test_mae",
                "test_pearson",
                "test_r2",
            ]
        )
    if _is_main_process():
        with open(logs_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=log_fields,
            )
            w.writeheader()

    history = {"train_loss": [], "val_metric": []}
    best_val_loss: Optional[float] = None
    best_state: Optional[Dict[str, Any]] = None
    bad = 0
    global_step = 0
    last_w: Optional[torch.Tensor] = None
    last_val_metrics: Optional[Dict[str, float]] = None
    last_val_loss: Optional[float] = None
    last_val_y: Optional[torch.Tensor] = None
    last_val_o: Optional[torch.Tensor] = None
    best_val_epoch: Optional[int] = None
    test_at_best_val_epoch_loss: Optional[float] = None
    test_at_best_val_epoch_metrics: Optional[Dict[str, float]] = None

    noise_mode = str(emb_cfg.get("noise_mode", "per_subject"))

    def _forward_batch(batch: Dict[str, Any], *, enable_grad: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x = batch["x"].to(device=device)
        y = batch["y"].to(device=device)
        subjects = list(batch["subject"])

        indices = None
        if "index" in batch:
            indices = [int(i) for i in batch["index"]]

        tokens_list: List[torch.Tensor] = []
        for t in t_list:
            out = protocol.tokens_from_batch(
                x,
                subjects=subjects,
                sample_indices=indices if noise_mode == "per_sample_index" else None,
                enable_grad=enable_grad,
                timestep=int(t),
            )
            tokens_list.extend(out.tokens_list)

        if agg_type == "token_attn":
            E = torch.cat(tokens_list, dim=1)
        else:
            e_list = [tok.mean(dim=1) for tok in tokens_list]
            E = torch.stack(e_list, dim=1)

        emb2, weights = aggregator(E)
        pred = head(emb2)
        return y, pred, weights

    def _eval_loader(
        loader: DataLoader,
        *,
        desc: Optional[str] = None,
    ) -> Tuple[Dict[str, float], float, float, torch.Tensor, torch.Tensor]:
        dit.eval()
        aggregator.eval()
        head.eval()
        all_y: List[torch.Tensor] = []
        all_out: List[torch.Tensor] = []
        all_idx: List[torch.Tensor] = []
        loss_sum = 0.0
        n_obs = 0
        idx_fallback_cursor = 0
        with torch.no_grad():
            it = loader
            if desc is not None:
                it = tqdm(loader, desc=desc, leave=False, ncols=120, disable=not _is_main_process())
            for batch in it:
                y, out, _w = _forward_batch(batch, enable_grad=False)
                if task.task_type == "regression":
                    loss = loss_fn(out, y.view(-1, 1))
                else:
                    loss = loss_fn(out, y.long().view(-1))
                bsz = int(y.shape[0])
                loss_sum += float(loss.item()) * max(1, bsz)
                n_obs += bsz
                all_y.append(y)
                all_out.append(out)
                if "index" in batch:
                    all_idx.append(torch.as_tensor(batch["index"]).view(-1).detach().cpu().long())
                else:
                    all_idx.append(torch.arange(idx_fallback_cursor, idx_fallback_cursor + bsz, dtype=torch.long))
                    idx_fallback_cursor += bsz

        yt = torch.cat(all_y, dim=0)
        ot = torch.cat(all_out, dim=0)
        it_idx = torch.cat(all_idx, dim=0)
        yt, ot = _gather_and_dedupe_eval(yt, ot, it_idx)

        if use_ddp and _is_dist_initialized() and _world_size() > 1:
            stats = torch.tensor([loss_sum, float(n_obs)], device=device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            loss_mean = float((stats[0] / stats[1].clamp_min(1.0)).item())
        else:
            loss_mean = float(loss_sum / max(1, n_obs))

        if task.task_type == "regression":
            metrics = evaluate_regression(yt, ot)
        else:
            metrics = evaluate_classification(yt, ot, num_classes=task.num_classes)
        primary = _primary_metric(task, best_metric_name, metrics)
        return metrics, float(primary), float(loss_mean), yt, ot

    # Save run meta
    save_json(
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "out_dir": out_dir,
            "device": str(device),
            "t_list": t_list,
            "capture_layers": capture_layers,
            "stage1_ckpt": str(stage1_ckpt),
        },
        os.path.join(out_dir, "run_meta.json"),
    )

    epoch_pbar = tqdm(range(1, epochs + 1), desc="epochs", ncols=120, disable=not _is_main_process())
    for epoch in epoch_pbar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        dit.train()
        aggregator.train()
        head.train()
        losses = []

        running = {"loss_sum": 0.0, "n": 0, "correct": 0}
        train_iter = tqdm(train_loader, desc=f"train[{epoch}/{epochs}]", leave=False, ncols=120, disable=not _is_main_process())
        for step, batch in enumerate(train_iter, start=1):
            optimizer.zero_grad(set_to_none=True)
            _apply_lr_scale(global_step)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y, out, w = _forward_batch(batch, enable_grad=(mode != "linear_probe"))
                last_w = w.detach() if isinstance(w, torch.Tensor) else None
                if task.task_type == "regression":
                    loss = loss_fn(out, y.view(-1, 1))
                else:
                    loss = loss_fn(out, y.long().view(-1))

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dit.parameters() if p.requires_grad]
                    + [p for p in aggregator.parameters() if p.requires_grad]
                    + [p for p in head.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()

            li = float(loss.item())
            losses.append(li)
            running["loss_sum"] += li
            running["n"] += int(y.shape[0])

            if _is_main_process():
                if task.task_type == "classification":
                    pred = torch.argmax(out.detach(), dim=-1)
                    yy = y.long().view(-1)
                    running["correct"] += int((pred == yy).sum().item())
                    acc_run = running["correct"] / max(1, running["n"])
                    train_iter.set_postfix(loss=f"{running['loss_sum']/max(1, step):.4f}", acc=f"{acc_run:.3f}", w=_format_weights_for_tqdm(last_w))
                else:
                    mse_b = torch.mean((out.detach().view(-1) - y.view(-1)) ** 2).item()
                    train_iter.set_postfix(loss=f"{running['loss_sum']/max(1, step):.4f}", mse=f"{mse_b:.4f}", w=_format_weights_for_tqdm(last_w))

            global_step += 1

        val_m, primary, val_loss, _yv, _ov = _eval_loader(valid_loader, desc="eval[valid]")
        last_val_metrics = val_m
        last_val_loss = float(val_loss)
        last_val_y = _yv
        last_val_o = _ov
        epoch_test_metrics: Optional[Dict[str, float]] = None
        epoch_test_loss: Optional[float] = None
        if eval_test_each_epoch:
            t_m, _p, t_loss, _yt, _ot = _eval_loader(test_loader, desc="eval[test@epoch]")
            epoch_test_metrics = t_m
            epoch_test_loss = float(t_loss)
        train_loss = float(np.mean(losses) if losses else 0.0)
        history["train_loss"].append(train_loss)
        history["val_metric"].append(float(primary))

        if _is_main_process():
            if task.task_type == "regression":
                epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_r2=f"{val_m.get('r2', 0.0):.3f}", val_pearson=f"{val_m.get('pearson', 0.0):.3f}", w=_format_weights_for_tqdm(last_w))
            else:
                epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_m.get('acc', 0.0):.3f}", val_f1w=f"{val_m.get('f1_weighted', 0.0):.3f}", w=_format_weights_for_tqdm(last_w))

            with open(logs_path, "a", newline="", encoding="utf-8") as f:
                wcsv = csv.DictWriter(
                    f,
                    fieldnames=log_fields,
                )
                row = {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "val_loss": float(val_loss),
                    "val_acc": float(val_m.get("acc", 0.0)),
                    "val_f1_weighted": float(val_m.get("f1_weighted", 0.0)),
                    "val_pearson": float(val_m.get("pearson", 0.0)),
                    "val_r2": float(val_m.get("r2", 0.0)),
                }
                if eval_test_each_epoch:
                    row.update(
                        {
                            "test_loss": float(epoch_test_loss) if epoch_test_loss is not None else "",
                            "test_acc": float((epoch_test_metrics or {}).get("acc", 0.0)),
                            "test_f1_weighted": float((epoch_test_metrics or {}).get("f1_weighted", 0.0)),
                            "test_auroc": float((epoch_test_metrics or {}).get("auroc", 0.0)),
                            "test_mse": float((epoch_test_metrics or {}).get("mse", 0.0)),
                            "test_mae": float((epoch_test_metrics or {}).get("mae", 0.0)),
                            "test_pearson": float((epoch_test_metrics or {}).get("pearson", 0.0)),
                            "test_r2": float((epoch_test_metrics or {}).get("r2", 0.0)),
                        }
                    )
                wcsv.writerow(row)

        # Per-epoch weights
        if _is_main_process() and bool(out_cfg.get("log_weights_every_epoch", True)):
            w_epoch_t = _weights_for_log(last_w)
            w_epoch = w_epoch_t.detach().cpu().tolist() if w_epoch_t is not None else None
            if w_epoch is not None:
                with open(os.path.join(out_dir, f"layer_weights_epoch{int(epoch)}.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "epoch": int(epoch),
                            "weights": w_epoch,
                            "capture_layers": capture_layers,
                            "t_list": t_list,
                        },
                        f,
                        indent=2,
                    )

        stop = False
        if _is_main_process():
            improved = best_val_loss is None or float(val_loss) < best_val_loss
            if improved:
                best_val_loss = float(val_loss)
                best_val_epoch = int(epoch)
                if eval_test_each_epoch and epoch_test_metrics is not None and epoch_test_loss is not None:
                    test_at_best_val_epoch_loss = float(epoch_test_loss)
                    test_at_best_val_epoch_metrics = {k: float(v) for k, v in epoch_test_metrics.items()}
                else:
                    test_at_best_val_epoch_loss = None
                    test_at_best_val_epoch_metrics = None
                best_state = {
                    "dit": _unwrap_module(dit).state_dict(),
                    "aggregator": _unwrap_module(aggregator).state_dict(),
                    "head": _unwrap_module(head).state_dict(),
                    "cfg": cfg,
                    "stage1_cfg": cfg1,
                    "best_val_loss": float(best_val_loss),
                    "best_val_primary": float(primary),
                    "best_metric": str(best_metric_name),
                    "layer_weights": (_weights_for_log(last_w).detach().cpu().tolist() if last_w is not None else None),
                    "t_list": t_list,
                    "capture_layers": capture_layers,
                    "best_epoch": int(best_val_epoch),
                    "test_at_best_val_epoch_loss": test_at_best_val_epoch_loss,
                    "test_at_best_val_epoch_metrics": test_at_best_val_epoch_metrics,
                }
                torch.save(best_state, os.path.join(out_dir, "best.pt"))
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    stop = True

        if use_ddp and _is_dist_initialized() and _world_size() > 1:
            obj = [bool(stop) if _is_main_process() else None]
            dist.broadcast_object_list(obj, src=0)
            stop = bool(obj[0])
            dist.barrier()

        if stop:
            break

    if use_ddp and _is_dist_initialized() and _world_size() > 1:
        flag = [best_state is not None if _is_main_process() else None]
        dist.broadcast_object_list(flag, src=0)
        if not bool(flag[0]):
            raise RuntimeError("No best state captured")
    elif best_state is None:
        raise RuntimeError("No best state captured")

    # Load best
    best_path = os.path.join(out_dir, "best.pt")
    if use_ddp and _is_dist_initialized() and _world_size() > 1:
        dist.barrier()
        best_state = torch.load(best_path, map_location=device)
        _unwrap_module(dit).load_state_dict(best_state["dit"], strict=True)
        _unwrap_module(aggregator).load_state_dict(best_state["aggregator"], strict=True)
        _unwrap_module(head).load_state_dict(best_state["head"], strict=True)
    else:
        best_state = torch.load(best_path, map_location=device)
        dit.load_state_dict(best_state["dit"], strict=True)
        aggregator.load_state_dict(best_state["aggregator"], strict=True)
        head.load_state_dict(best_state["head"], strict=True)

    train_m: Dict[str, float] = {}
    ytr: Optional[torch.Tensor] = None
    otr: Optional[torch.Tensor] = None
    train_loss_eval = float(train_loss)
    if eval_train_after:
        train_m, _p, train_loss_eval, ytr, otr = _eval_loader(train_eval_loader, desc="eval[train]")

    valid_m: Dict[str, float] = {}
    valid_loss = float(last_val_loss) if last_val_loss is not None else float("nan")
    yv = last_val_y
    ov = last_val_o
    if eval_valid_after:
        valid_m, _p, valid_loss, yv, ov = _eval_loader(valid_loader, desc="eval[valid]")
    elif last_val_metrics is not None:
        valid_m = last_val_metrics

    test_m: Dict[str, float] = {}
    test_loss = float("nan")
    yt: Optional[torch.Tensor] = None
    ot: Optional[torch.Tensor] = None
    if eval_test_after:
        test_m, _p, test_loss, yt, ot = _eval_loader(test_loader, desc="eval[test]")

    test_at_best_val_epoch = None
    test_compare: Dict[str, Any] = {
        "enabled": bool(eval_test_each_epoch),
        "best_val_epoch": int(best_state.get("best_epoch", -1)),
    }
    logged_test_metrics = best_state.get("test_at_best_val_epoch_metrics", None)
    logged_test_loss = best_state.get("test_at_best_val_epoch_loss", None)
    if isinstance(logged_test_metrics, dict) and logged_test_loss is not None:
        test_at_best_val_epoch = {
            "epoch": int(best_state.get("best_epoch", -1)),
            "loss": float(logged_test_loss),
            "metrics": {k: float(v) for k, v in logged_test_metrics.items()},
        }
        if len(test_m) > 0 and np.isfinite(float(test_loss)):
            delta = _metric_delta(test_m, test_at_best_val_epoch["metrics"])
            test_compare["loaded_best_ckpt_minus_best_val_epoch"] = {
                "loss": float(test_loss) - float(test_at_best_val_epoch["loss"]),
                "metrics": delta,
            }
    elif eval_test_each_epoch:
        test_compare["note"] = "test_each_epoch enabled but no per-epoch test snapshot found"

    # Plots
    if _is_main_process() and bool(out_cfg.get("save_plots", True)):
        plot_learning_curves(history, os.path.join(out_dir, "plots", "learning_curves.png"), title=f"{mode} + agg")
        if yv is not None and ov is not None:
            if task.task_type == "regression":
                plot_regression_scatter(
                    yv.detach().cpu().numpy().reshape(-1),
                    ov.detach().cpu().numpy().reshape(-1),
                    os.path.join(out_dir, "plots", "scatter_valid.png"),
                    title="Valid",
                )
            else:
                cm = confusion_matrix(yv, ov, num_classes=task.num_classes)
                plot_confusion_matrix(cm, os.path.join(out_dir, "plots", "confusion_valid.png"), title="Valid")
        if task.num_classes == 2 and yt is not None and ot is not None:
            score = torch.softmax(ot, dim=-1)[:, 1].detach().cpu().numpy().reshape(-1)
            yt_np = yt.detach().cpu().numpy().reshape(-1)
            fpr, tpr = compute_roc_curve_binary(yt_np, score)
            plot_roc_curve(fpr, tpr, os.path.join(out_dir, "plots", "roc_test.png"), title="Test", auc=test_m.get("auroc"))

    # Save preds
    if _is_main_process() and bool(out_cfg.get("save_preds", True)) and yt is not None and ot is not None:
        gt = yt.detach().cpu().numpy()
        if task.task_type == "regression":
            pred = ot.detach().cpu().numpy().reshape(-1)
        else:
            pred = torch.softmax(ot, dim=-1).detach().cpu().numpy()
        np.save(os.path.join(out_dir, "test_gt.npy"), gt)
        np.save(os.path.join(out_dir, "test_pred.npy"), pred)

    # Summary
    summary = {
        "mode": mode,
        "out_dir": out_dir,
        "best_metric": str(best_metric_name),
        "best_val_loss": float(best_state["best_val_loss"]),
        "best_val_primary": float(best_state["best_val_primary"]),
        "best_val_epoch": int(best_state.get("best_epoch", -1)),
        "train": {"loss": float(train_loss_eval), "metrics": train_m},
        "valid": {"loss": float(valid_loss), "metrics": valid_m},
        "test": {"loss": float(test_loss), "metrics": test_m},
        "test_loaded_from_best_ckpt": {"loss": float(test_loss), "metrics": test_m},
        "test_at_best_val_epoch": test_at_best_val_epoch,
        "test_compare": test_compare,
        "layer_weights": best_state.get("layer_weights", None),
        "capture_layers": capture_layers,
        "t_list": t_list,
    }
    if _is_main_process():
        save_json(summary, os.path.join(out_dir, "metrics.json"))
        n_params, mb = _count_trainable([p for p in dit.parameters() if p.requires_grad] + [p for p in aggregator.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad])
        print(f"[trainable] params={n_params} (~{mb:.1f} MB fp32)")
        print(f"[ok] wrote: {out_dir}")

    if use_ddp and _is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
