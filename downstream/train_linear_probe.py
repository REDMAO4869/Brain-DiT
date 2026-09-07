from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from downstream._path import ensure_repo_root

ensure_repo_root()

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
from downstream.plots import (
    compute_roc_curve_binary,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_regression_scatter,
    plot_roc_curve,
)
from downstream.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 (raw): train linear probe on extracted embeddings")
    p.add_argument("--config", required=True, help="YAML config path")
    return p.parse_args()


@dataclass
class TaskSpec:
    task_type: str  # classification|regression
    num_classes: int


def _load_split(emb_dir: str, split: str):
    emb_path = os.path.join(emb_dir, f"{split}_emb.npy")
    y_path = os.path.join(emb_dir, f"{split}_y.npy")
    if not os.path.isfile(emb_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(f"Missing embedding files for split={split}: {emb_path} {y_path}")
    emb = np.load(emb_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)
    return emb, y


def _infer_task(cfg: Dict[str, Any], y_train: np.ndarray) -> TaskSpec:
    task = cfg.get("task", {})
    t = str(task.get("type", "regression"))
    if t not in ("classification", "regression"):
        raise ValueError("task.type must be classification or regression")

    if t == "regression":
        return TaskSpec(task_type=t, num_classes=1)

    unique = np.unique(y_train)
    if not np.all(np.isclose(unique, np.round(unique))):
        raise ValueError("For classification, labels must be numeric class ids (e.g., 0/1/2)")
    unique_int = unique.astype(np.int64)
    if unique_int.min() < 0:
        raise ValueError("Class ids must be >= 0")
    k = int(unique_int.max()) + 1
    return TaskSpec(task_type=t, num_classes=k)


def _evaluate(task: TaskSpec, y_true_t: torch.Tensor, logits_or_pred: torch.Tensor) -> Dict[str, float]:
    if task.task_type == "regression":
        pred = logits_or_pred.view(-1)
        y = y_true_t.view(-1)
        return {
            "mse": mse(y, pred),
            "mae": mae(y, pred),
            "pearson": pearsonr(y, pred),
            "r2": r2_score(y, pred),
        }

    logits = logits_or_pred
    y = y_true_t.long().view(-1)
    out = {
        "acc": accuracy(y, logits),
        "balanced_acc": balanced_accuracy(y, logits),
        "f1_weighted": f1_weighted(y, logits, num_classes=task.num_classes),
    }
    if task.num_classes == 2:
        score = torch.softmax(logits, dim=-1)[:, 1]
        out["auroc"] = auroc_binary(y, score)
    return out


def main() -> None:
    args = parse_args()
    root_cfg = load_yaml(args.config)

    cfg = root_cfg.get("linear_probe", root_cfg)

    out_dir = os.path.abspath(str(cfg["output_dir"]))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))

    emb_dir = os.path.abspath(str(cfg["embeddings_dir"]))
    tr_emb, tr_y = _load_split(emb_dir, "train")
    va_emb, va_y = _load_split(emb_dir, "valid")
    te_emb, te_y = _load_split(emb_dir, "test")

    task = _infer_task(cfg, tr_y)

    device_str = str(root_cfg.get("device", cfg.get("device", "auto")))
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    x_train = torch.from_numpy(tr_emb)
    x_val = torch.from_numpy(va_emb)
    x_test = torch.from_numpy(te_emb)

    if task.task_type == "regression":
        y_train = torch.from_numpy(tr_y).float().view(-1, 1)
        y_val = torch.from_numpy(va_y).float().view(-1, 1)
        y_test = torch.from_numpy(te_y).float().view(-1, 1)
        head = nn.Linear(x_train.shape[1], 1)
        loss_fn = nn.MSELoss()
    else:
        y_train = torch.from_numpy(tr_y).long().view(-1)
        y_val = torch.from_numpy(va_y).long().view(-1)
        y_test = torch.from_numpy(te_y).long().view(-1)
        head = nn.Linear(x_train.shape[1], task.num_classes)
        train_cfg = cfg.get("train", {})
        use_class_weights = bool(train_cfg.get("use_class_weights", True))
        if use_class_weights:
            counts = torch.bincount(y_train, minlength=task.num_classes).float()
            total = float(torch.sum(counts).item())
            denom = (counts * float(task.num_classes)).clamp_min(1.0)
            w = total / denom
            w[counts == 0] = 0.0
            class_weights = w.to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

    head = head.to(device)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    test_ds = TensorDataset(x_test, y_test)

    tr_cfg = cfg.get("train", {})
    batch_size = int(tr_cfg.get("batch_size", 64))
    lr = float(tr_cfg.get("lr", 1e-3))
    wd = float(tr_cfg.get("weight_decay", 0.0))
    epochs = int(tr_cfg.get("epochs", 50))
    patience = int(tr_cfg.get("patience", 10))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    best_val_loss = None
    best_state = None
    bad = 0

    history = {"train_loss": [], "val_metric": [], "val_loss": []}

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

    def _eval_loader(loader: DataLoader) -> Tuple[Dict[str, float], float, torch.Tensor, torch.Tensor]:
        head.eval()
        with torch.no_grad():
            all_y = []
            all_out = []
            losses: List[float] = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = head(xb)
                if task.task_type == "regression":
                    loss = loss_fn(out, yb.view(-1, 1))
                else:
                    loss = loss_fn(out, yb)
                losses.append(float(loss.item()))
                all_y.append(yb)
                all_out.append(out)
            yv = torch.cat(all_y, dim=0)
            ov = torch.cat(all_out, dim=0)
            return _evaluate(task, yv, ov), float(np.mean(losses) if losses else 0.0), yv, ov

    epoch_pbar = tqdm(range(1, epochs + 1), desc="epochs", ncols=120)
    for epoch in epoch_pbar:
        head.train()
        losses = []
        running = {"loss_sum": 0.0, "n": 0, "correct": 0}

        train_pbar = tqdm(train_loader, desc=f"train[{epoch}/{epochs}]", leave=False, ncols=120)
        for step, (xb, yb) in enumerate(train_pbar, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = head(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            li = float(loss.item())
            losses.append(li)
            running["loss_sum"] += li
            running["n"] += int(xb.shape[0])

            if task.task_type == "classification":
                pred = torch.argmax(out.detach(), dim=-1)
                running["correct"] += int((pred == yb).sum().item())
                acc = running["correct"] / max(1, running["n"])
                train_pbar.set_postfix(loss=f"{running['loss_sum']/max(1, step):.4f}", acc=f"{acc:.3f}")
            else:
                mse_b = torch.mean((out.detach() - yb.view(-1, 1)) ** 2).item()
                train_pbar.set_postfix(loss=f"{running['loss_sum']/max(1, step):.4f}", mse=f"{mse_b:.4f}")

        metrics, val_loss, yv, ov = _eval_loader(val_loader)

        primary = metrics["r2"] if task.task_type == "regression" else metrics["f1_weighted"]

        history["train_loss"].append(float(np.mean(losses) if losses else 0.0))
        history["val_metric"].append(float(primary))
        history["val_loss"].append(float(val_loss))

        if task.task_type == "regression":
            epoch_pbar.set_postfix(
                train_loss=f"{history['train_loss'][-1]:.4f}",
                val_r2=f"{metrics.get('r2', 0.0):.3f}",
                val_pearson=f"{metrics.get('pearson', 0.0):.3f}",
            )
        else:
            epoch_pbar.set_postfix(
                train_loss=f"{history['train_loss'][-1]:.4f}",
                val_acc=f"{metrics.get('acc', 0.0):.3f}",
                val_f1w=f"{metrics.get('f1_weighted', 0.0):.3f}",
            )

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
                    "epoch": int(epoch),
                    "train_loss": float(history["train_loss"][-1]),
                    "val_loss": float(val_loss),
                    "val_acc": float(metrics.get("acc", 0.0)),
                    "val_f1_weighted": float(metrics.get("f1_weighted", 0.0)),
                    "val_pearson": float(metrics.get("pearson", 0.0)),
                    "val_r2": float(metrics.get("r2", 0.0)),
                }
            )

        improved = best_val_loss is None or float(val_loss) < best_val_loss
        if improved:
            best_val_loss = float(val_loss)
            best_state = {
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "task": task.__dict__,
                "best_val_loss": float(best_val_loss),
                "best_val_primary": float(primary),
            }
            torch.save(best_state, os.path.join(out_dir, "best.pt"))
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is None:
        raise RuntimeError("No best state captured")

    best_path = os.path.join(out_dir, "best.pt")
    best_state = torch.load(best_path, map_location=device)
    head.load_state_dict(best_state["head"], strict=True)

    train_metrics, train_loss_eval, ytr, otr = _eval_loader(train_loader)
    val_metrics, val_loss_eval, yv, ov = _eval_loader(val_loader)
    test_metrics, test_loss_eval, yt, ot = _eval_loader(test_loader)

    save_json(
        {
            "task": task.__dict__,
            "best_val_loss": float(best_state["best_val_loss"]),
            "best_val_primary": float(best_state["best_val_primary"]),
            "train": {"loss": float(train_loss_eval), "metrics": train_metrics},
            "valid": {"loss": float(val_loss_eval), "metrics": val_metrics},
            "test": {"loss": float(test_loss_eval), "metrics": test_metrics},
        },
        os.path.join(out_dir, "metrics.json"),
    )
    plot_learning_curves(history, os.path.join(out_dir, "plots", "learning_curves.png"), title="Linear Probe (raw)")

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
            plot_roc_curve(fpr, tpr, os.path.join(out_dir, "plots", "roc_test.png"), title="Test", auc=test_metrics.get("auroc"))

    print(f"Done. out_dir={out_dir} test={test_metrics}")


if __name__ == "__main__":
    main()
