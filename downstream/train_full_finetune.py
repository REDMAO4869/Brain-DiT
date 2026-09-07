from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
from downstream.feature_extractor_raw import load_stage1_raw, resolve_capture_layer
from downstream.metrics import (
    accuracy,
    auroc_binary,
    balanced_accuracy,
    confusion_matrix,
    f1_weighted,
    mae,
    mse,
    pearsonr,
    r2_score,
)
from downstream.noise_utils import NoiseMode, noise_for_batch
from downstream.plots import (
    compute_roc_curve_binary,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_regression_scatter,
    plot_roc_curve,
)
from downstream.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 (raw): Full fine-tune DiT + head")
    p.add_argument("--config", required=True, help="YAML config path")
    return p.parse_args()


@dataclass
class TaskSpec:
    task_type: str
    num_classes: int


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


def _resolve_split_csv(cfg: Dict[str, Any], split: str) -> str:
    data_cfg = cfg["data"]
    splits_cfg = data_cfg.get("splits", None)
    if isinstance(splits_cfg, dict) and split in splits_cfg and isinstance(splits_cfg[split], dict) and "csv" in splits_cfg[split]:
        return str(splits_cfg[split]["csv"])

    dataset = str(data_cfg["dataset_name"])
    atlas = str(data_cfg.get("atlas", "AA424"))
    fold = int(data_cfg.get("fold_id", 0))
    split_root = str(data_cfg["split_root"])
    return os.path.join(split_root, dataset, atlas, f"{split}_{fold}.csv")


def _build_dataset(cfg: Dict[str, Any], split: str):
    data_cfg = cfg["data"]
    label_cfg = cfg["label"]

    csv_path = _resolve_split_csv(cfg, split)
    split_rows = read_split_csv(
        csv_path,
        subject_col=str(data_cfg.get("subject_col", "Subject")),
        path_col=str(data_cfg.get("path_col", "Path")),
    )

    labels = read_labels_csv(
        str(label_cfg["label_csv_path"]),
        subject_col=str(label_cfg["label_subject_col"]),
        label_col=str(label_cfg["label_target_col"]),
        label_map=label_cfg.get("label_map", None),
    )
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


def _task_from_cfg(cfg: Dict[str, Any]) -> TaskSpec:
    label_cfg = cfg["label"]
    task_type = str(label_cfg.get("task_type", "classification"))
    if task_type not in ("classification", "regression"):
        raise ValueError("label.task_type must be classification or regression")

    if task_type == "regression":
        return TaskSpec(task_type=task_type, num_classes=1)

    num_classes = int(label_cfg.get("num_classes", 0))
    if num_classes <= 1:
        raise ValueError("For classification, label.num_classes must be set >= 2")
    return TaskSpec(task_type=task_type, num_classes=num_classes)


def _count_trainable(params: List[torch.nn.Parameter]) -> Tuple[int, float]:
    n = int(sum(p.numel() for p in params if p.requires_grad))
    mb = (n * 4) / (1024.0**2)
    return n, mb


def _to_x0(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).unsqueeze(1).contiguous()


def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return torch.mean(tokens, dim=1)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    _set_seed(int(cfg.get("train", {}).get("seed", 0)))

    device_str = str(cfg.get("train", {}).get("device", cfg.get("device", "auto")))
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    out_cfg = cfg.get("output", {})
    out_root = os.path.abspath(str(out_cfg.get("out_root", "checkpoints_raw")))
    data_cfg = cfg["data"]
    label_cfg = cfg["label"]
    dataset_name = str(data_cfg.get("dataset_name", "DATA"))
    task_name = str(label_cfg.get("label_target_col", "task"))
    fold = int(data_cfg.get("fold_id", 0))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, dataset_name, task_name, "raw_fullft", f"fold{fold}", run_id)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))

    ckpt_cfg = cfg["checkpoints"]
    dit, cond, schedule, cfg1 = load_stage1_raw(str(ckpt_cfg["stage1_ckpt"]), device=device)

    cond.eval()
    for p in cond.parameters():
        p.requires_grad = False

    dit.train()
    for p in dit.parameters():
        p.requires_grad = True

    task = _task_from_cfg(cfg)

    emb_cfg = cfg.get("embedding", {})
    timestep = int(emb_cfg.get("timestep", 10))
    capture_layer = resolve_capture_layer(dit, int(emb_cfg.get("capture_layer", -2)))
    pool = str(emb_cfg.get("pool", "mean"))
    if pool != "mean":
        raise ValueError("Only embedding.pool=mean is supported")

    noise_mode: NoiseMode = str(emb_cfg.get("noise_mode", "per_subject"))  # type: ignore[assignment]
    noise_seed = int(emb_cfg.get("noise_seed", 0))

    ds_tr, miss_tr = _build_dataset(cfg, "train")
    ds_va, miss_va = _build_dataset(cfg, "valid")
    ds_te, miss_te = _build_dataset(cfg, "test")

    all_missing = list(miss_tr) + list(miss_va) + list(miss_te)
    if len(all_missing) > 0:
        miss_path = os.path.join(out_dir, "missing_labels.csv")
        write_missing_labels_csv(all_missing, miss_path)
        print(f"[warn] missing labels: {len(all_missing)} -> {miss_path}")

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    valid_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    d_model = int(getattr(dit, "d_model"))
    if task.task_type == "regression":
        head = nn.Linear(d_model, 1)
        loss_fn = nn.MSELoss()
    else:
        head = nn.Linear(d_model, task.num_classes)
        tr_cfg = cfg.get("train", {})
        use_class_weights = bool(tr_cfg.get("use_class_weights", True))
        if use_class_weights:
            count_loader = DataLoader(
                ds_tr,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            counts = torch.zeros((int(task.num_classes),), dtype=torch.float32)
            for b in tqdm(count_loader, desc="count_classes", ncols=120):
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

    head = head.to(device)
    for p in head.parameters():
        p.requires_grad = True

    opt_cfg = cfg.get("optim", {})
    dit_lr = float(opt_cfg.get("dit_lr", 1e-5))
    head_lr = float(opt_cfg.get("head_lr", 1e-4))
    wd_dit = float(opt_cfg.get("weight_decay_dit", 1e-2))
    wd_head = float(opt_cfg.get("weight_decay_head", 0.0))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in dit.parameters() if p.requires_grad], "lr": dit_lr, "weight_decay": wd_dit},
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": head_lr, "weight_decay": wd_head},
        ],
        betas=betas,
        eps=eps,
    )

    tr_cfg = cfg.get("train", {})
    use_amp = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip_norm = float(tr_cfg.get("grad_clip_norm", 0.0))
    epochs = int(tr_cfg.get("epochs", 10))
    patience = int(tr_cfg.get("patience", 10))
    warmup_ratio = float(tr_cfg.get("warmup_ratio", 0.05))
    best_metric_name = str(tr_cfg.get("best_metric", "f1_weighted" if task.task_type == "classification" else "pearson"))

    if task.task_type == "classification":
        valid_best = {"f1_weighted", "acc", "balanced_acc", "auroc"}
    else:
        valid_best = {"pearson", "r2", "mse", "mae"}
    if best_metric_name not in valid_best:
        raise ValueError(f"train.best_metric={best_metric_name} not valid for task_type={task.task_type}")
    best_metric_higher_is_better = best_metric_name not in {"mse", "mae"}

    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)

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

    logs_path = os.path.join(out_dir, "logs.csv")
    with open(logs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_f1_weighted",
                "val_pearson",
                "val_r2",
            ],
        )
        w.writeheader()

    history = {"train_loss": [], "val_metric": []}
    best_val_loss: Optional[float] = None
    best_state: Optional[Dict[str, Any]] = None
    bad = 0

    def _forward_batch(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["x"].to(device=device)
        y = batch["y"].to(device=device)

        x0 = _to_x0(x)
        b = int(x0.shape[0])
        t_vec = torch.full((b,), int(timestep), device=device, dtype=torch.int64)

        subjects = list(batch["subject"])
        indices = [int(i) for i in batch["index"]] if "index" in batch else None

        eps = noise_for_batch(
            mode=noise_mode,
            subjects=subjects,
            sample_indices=indices,
            global_seed=noise_seed,
            shape_per_sample=tuple(x0.shape[1:]),
            device=device,
            dtype=x0.dtype,
        )
        xt = schedule.q_sample(x0, t_vec, eps)

        with torch.no_grad():
            y_c = cond.encode(batch_size=b, device=device, dtype=xt.dtype, cond=None)

        out = dit(xt, t_vec, y_c, return_hiddens=True, capture_layers=[int(capture_layer)])
        if out.hiddens is None or int(capture_layer) not in out.hiddens:
            raise RuntimeError("Requested hidden layer not captured")
        tokens = out.hiddens[int(capture_layer)]
        emb = _pool_tokens(tokens)
        pred = head(emb)
        return y, pred

    def _eval_loader(loader: DataLoader) -> Tuple[Dict[str, float], float, float, torch.Tensor, torch.Tensor]:
        dit.eval()
        head.eval()
        all_y: List[torch.Tensor] = []
        all_out: List[torch.Tensor] = []
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                y, out = _forward_batch(batch)
                if task.task_type == "regression":
                    loss = loss_fn(out, y.view(-1, 1))
                else:
                    loss = loss_fn(out, y.long().view(-1))
                losses.append(float(loss.item()))
                all_y.append(y)
                all_out.append(out)

        yt = torch.cat(all_y, dim=0)
        ot = torch.cat(all_out, dim=0)

        if task.task_type == "regression":
            pred = ot.view(-1)
            yv = yt.view(-1)
            m = {"mse": mse(yv, pred), "mae": mae(yv, pred), "pearson": pearsonr(yv, pred), "r2": r2_score(yv, pred)}
        else:
            yv = yt.long().view(-1)
            m = {
                "acc": accuracy(yv, ot),
                "balanced_acc": balanced_accuracy(yv, ot),
                "f1_weighted": f1_weighted(yv, ot, num_classes=task.num_classes),
            }
            if task.num_classes == 2:
                score = torch.softmax(ot, dim=-1)[:, 1]
                m["auroc"] = auroc_binary(yv, score)

        primary = float(m.get(best_metric_name, 0.0))
        if not best_metric_higher_is_better:
            primary = -primary
        return m, primary, float(np.mean(losses) if losses else 0.0), yt, ot

    global_step = 0
    epoch_pbar = tqdm(range(1, epochs + 1), desc="epochs", ncols=120)
    for epoch in epoch_pbar:
        dit.train()
        head.train()

        losses = []
        train_pbar = tqdm(train_loader, desc=f"train[{epoch}/{epochs}]", leave=False, ncols=120)
        for batch in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            _apply_lr_scale(global_step)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y, out = _forward_batch(batch)
                if task.task_type == "regression":
                    loss = loss_fn(out, y.view(-1, 1))
                else:
                    loss = loss_fn(out, y.long().view(-1))

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dit.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad],
                    grad_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.item()))
            train_pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
            global_step += 1

        val_m, val_primary, val_loss, _yv, _ov = _eval_loader(valid_loader)
        history["train_loss"].append(float(np.mean(losses) if losses else 0.0))
        history["val_metric"].append(float(val_primary))

        if task.task_type == "regression":
            epoch_pbar.set_postfix(train_loss=f"{history['train_loss'][-1]:.4f}", val_r2=f"{val_m.get('r2', 0.0):.3f}", val_pearson=f"{val_m.get('pearson', 0.0):.3f}")
        else:
            epoch_pbar.set_postfix(train_loss=f"{history['train_loss'][-1]:.4f}", val_acc=f"{val_m.get('acc', 0.0):.3f}", val_f1w=f"{val_m.get('f1_weighted', 0.0):.3f}")

        with open(logs_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_acc",
                    "val_f1_weighted",
                    "val_pearson",
                    "val_r2",
                ],
            )
            w.writerow(
                {
                    "epoch": epoch,
                    "train_loss": history["train_loss"][-1],
                    "val_loss": val_loss,
                    "val_acc": val_m.get("acc", None),
                    "val_f1_weighted": val_m.get("f1_weighted", None),
                    "val_pearson": val_m.get("pearson", None),
                    "val_r2": val_m.get("r2", None),
                }
            )

        improved = best_val_loss is None or float(val_loss) < best_val_loss
        if improved:
            best_val_loss = float(val_loss)
            best_state = {
                "dit": {k: v.detach().cpu().clone() for k, v in dit.state_dict().items()},
                "cond": {k: v.detach().cpu().clone() for k, v in cond.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "cfg": cfg,
                "best_val_loss": float(best_val_loss),
                "best_val_primary": float(val_primary),
            }
            torch.save(best_state, os.path.join(out_dir, "best.pt"))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[early_stop] patience reached ({patience})")
                break

    if best_state is None:
        raise RuntimeError("No best state captured")

    best_path = os.path.join(out_dir, "best.pt")
    best_state = torch.load(best_path, map_location=device)
    dit.load_state_dict(best_state["dit"], strict=True)
    cond.load_state_dict(best_state["cond"], strict=True)
    head.load_state_dict(best_state["head"], strict=True)

    train_m, _p, ytr, otr = _eval_loader(train_loader)
    valid_m, _p, yv, ov = _eval_loader(valid_loader)
    test_m, _primary, yt, ot = _eval_loader(test_loader)

    save_json(
        {
            "task": task.__dict__,
            "best_metric": best_metric_name,
            "best_metric_higher_is_better": best_metric_higher_is_better,
            "best_val_loss": float(best_state["best_val_loss"]),
            "best_val_primary": float(best_state["best_val_primary"]),
            "train": train_m,
            "valid": valid_m,
            "test": test_m,
        },
        os.path.join(out_dir, "metrics.json"),
    )
    n_params, mb = _count_trainable([p for p in dit.parameters() if p.requires_grad] + [p for p in head.parameters() if p.requires_grad])
    print(f"[trainable] params={n_params} (~{mb:.1f} MB fp32)")
    plot_learning_curves(history, os.path.join(out_dir, "plots", "learning_curves.png"), title="Full Fine-tune (raw)")

    if task.task_type == "regression":
        yp_v = ov.detach().cpu().numpy().reshape(-1)
        yt_v = yv.detach().cpu().numpy().reshape(-1)
        plot_regression_scatter(yt_v, yp_v, os.path.join(out_dir, "plots", "scatter_valid.png"), title="Valid")
    else:
        cm = confusion_matrix(yv, ov, num_classes=task.num_classes)
        plot_confusion_matrix(cm, os.path.join(out_dir, "plots", "confusion_valid.png"), title="Valid")
        if task.num_classes == 2:
            score = torch.softmax(ot, dim=-1)[:, 1].detach().cpu().numpy().reshape(-1)
            yt_np = yt.detach().cpu().numpy().reshape(-1)
            fpr, tpr = compute_roc_curve_binary(yt_np, score)
            plot_roc_curve(fpr, tpr, os.path.join(out_dir, "plots", "roc_test.png"), title="Test", auc=test_m.get("auroc"))

    if bool(out_cfg.get("save_preds", True)):
        np.save(os.path.join(out_dir, "pred_valid.npy"), ov.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "y_valid.npy"), yv.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "pred_test.npy"), ot.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "y_test.npy"), yt.detach().cpu().numpy())

    print(f"Done. out_dir={out_dir} test={test_m}")


if __name__ == "__main__":
    main()
