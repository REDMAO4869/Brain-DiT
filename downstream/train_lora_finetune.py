from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from downstream.feature_extractor_raw import (
    EmbedConfig,
    Stage3RawFeatureExtractor,
    load_stage1_raw,
    resolve_capture_layer,
)
from downstream.lora import LoRAConfig, apply_lora_to_last_k_blocks
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
    p = argparse.ArgumentParser(description="Stage-3 (raw): LoRA fine-tuning on top of DiT embeddings")
    p.add_argument("--config", required=True, help="YAML config path")
    return p.parse_args()


def _build_dataset(cfg: Dict[str, Any], split: str):
    data_cfg = cfg["data"]
    split_cfg = data_cfg["splits"][split]

    split_rows = read_split_csv(
        split_cfg["csv"],
        subject_col=str(data_cfg.get("subject_col", "Subject")),
        path_col=str(data_cfg.get("path_col", "Path")),
    )
    labels = read_labels_csv(
        data_cfg["labels_csv"],
        subject_col=str(data_cfg["labels_subject_col"]),
        label_col=str(data_cfg["labels_label_col"]),
        label_map=data_cfg.get("label_map", None),
    )
    keep, missing = join_split_with_labels(split_rows, labels, split_name=split)

    ds = DownstreamCSVDataset(
        rows=keep,
        labels=labels,
        seq_len=int(data_cfg.get("seq_len")) if data_cfg.get("seq_len") is not None else None,
        crop_mode=str(data_cfg.get("crop_mode", "center")),
        pad_mode=str(data_cfg.get("pad_mode", "zeros")),
        roi_dim=int(data_cfg.get("roi_dim")) if data_cfg.get("roi_dim") is not None else None,
        path_prefix=str(data_cfg.get("path_prefix")) if data_cfg.get("path_prefix") is not None else None,
        strict_seq_len=bool(data_cfg.get("strict_seq_len", False)),
    )
    return ds, missing


def _infer_task(cfg: Dict[str, Any], y_train: List[float]):
    task_cfg = cfg.get("task", {})
    t = str(task_cfg.get("type", "regression"))
    if t not in ("classification", "regression"):
        raise ValueError("task.type must be classification or regression")

    if t == "regression":
        return t, 1

    y = np.asarray(y_train, dtype=np.float32)
    unique = np.unique(y)
    if not np.all(np.isclose(unique, np.round(unique))):
        raise ValueError("For classification, labels must be numeric class ids (e.g., 0/1/2)")
    k = int(np.max(unique)) + 1
    return t, k


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    out_dir = os.path.abspath(str(cfg["output_dir"]))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "plots"))

    device_str = str(cfg.get("device", "auto"))
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    dit, cond, schedule, cfg1 = load_stage1_raw(str(cfg["stage1_ckpt_path"]), device=device)

    lora_cfg = cfg.get("lora", {})
    lc = LoRAConfig(
        r=int(lora_cfg.get("r", 8)),
        alpha=float(lora_cfg.get("alpha", 16.0)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    replaced = apply_lora_to_last_k_blocks(dit, last_k=int(lora_cfg.get("last_k_blocks", 2)), cfg=lc)
    print(f"[lora] replaced blocks={replaced}")

    embed_cfg = cfg.get("embedding", {})
    capture_layer = resolve_capture_layer(dit, int(embed_cfg.get("capture_layer", -1)))
    ec = EmbedConfig(
        timestep=int(embed_cfg.get("timestep", 100)),
        capture_layer=capture_layer,
        pool=str(embed_cfg.get("pool", "mean")),
    )
    pred_type = str(cfg1.get("diffusion", {}).get("pred_type", "v"))
    extractor = Stage3RawFeatureExtractor(
        dit=dit,
        cond_encoder=cond,
        schedule=schedule,
        embed_cfg=ec,
        device=device,
        noise_seed=int(cfg.get("noise_seed", 0)),
        pred_type=pred_type,
    )

    dl_cfg = cfg.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 4))
    num_workers = int(dl_cfg.get("num_workers", 0))

    train_ds, miss_tr = _build_dataset(cfg, "train")
    valid_ds, miss_va = _build_dataset(cfg, "valid")
    test_ds, miss_te = _build_dataset(cfg, "test")
    all_missing = list(miss_tr) + list(miss_va) + list(miss_te)
    if len(all_missing) > 0:
        miss_path = os.path.join(out_dir, "missing_labels.csv")
        write_missing_labels_csv(all_missing, miss_path)
        print(f"[warn] missing labels: {len(all_missing)} -> {miss_path}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    task_type, num_classes = _infer_task(cfg, [float(train_ds[i]["y"].item()) for i in range(min(len(train_ds), 1000))])

    d_model = int(getattr(dit, "d_model"))
    if task_type == "regression":
        head = nn.Linear(d_model, 1)
        loss_fn = nn.MSELoss()
    else:
        head = nn.Linear(d_model, num_classes)
        tr_cfg = cfg.get("train", {})
        use_class_weights = bool(tr_cfg.get("use_class_weights", True))
        if use_class_weights:
            count_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            counts = torch.zeros((int(num_classes),), dtype=torch.float32)
            for b in tqdm(count_loader, desc="count_classes", ncols=120):
                yy = b["y"].long().view(-1)
                counts += torch.bincount(yy.cpu(), minlength=int(num_classes)).float()
            total = float(torch.sum(counts).item())
            denom = (counts * float(num_classes)).clamp_min(1.0)
            w = total / denom
            w[counts == 0] = 0.0
            class_weights = w.to(device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

    head = head.to(device)

    for p in dit.parameters():
        p.requires_grad = False
    for p in cond.parameters():
        p.requires_grad = False

    for name, p in dit.named_parameters():
        if "q_A" in name or "q_B" in name or "v_A" in name or "v_B" in name:
            p.requires_grad = True

    for p in head.parameters():
        p.requires_grad = True

    trainable = [p for p in list(dit.parameters()) + list(head.parameters()) if p.requires_grad]

    tr_cfg = cfg.get("train", {})
    lr = float(tr_cfg.get("lr", 1e-4))
    wd = float(tr_cfg.get("weight_decay", 0.0))
    epochs = int(tr_cfg.get("epochs", 20))
    patience = int(tr_cfg.get("patience", 5))

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

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

    noise_mode = str(cfg.get("embedding", {}).get("noise_mode", "per_subject"))

    def eval_loader(loader):
        extractor.dit.eval()
        head.eval()
        all_y = []
        all_out = []
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device=device)
                y = batch["y"].to(device=device)
                feat = extractor.extract_features(
                    x,
                    enable_grad=False,
                    noise_ids=list(batch["subject"]),
                    noise_mode=noise_mode,
                )
                out = head(feat)
                if task_type == "regression":
                    loss = loss_fn(out, y.view(-1, 1))
                else:
                    loss = loss_fn(out, y.long().view(-1))
                losses.append(float(loss.item()))
                all_y.append(y)
                all_out.append(out)
        yv = torch.cat(all_y, dim=0)
        ov = torch.cat(all_out, dim=0)
        if task_type == "regression":
            pred = ov.view(-1)
            yt = yv.view(-1)
            m = {"mse": mse(yt, pred), "mae": mae(yt, pred), "pearson": pearsonr(yt, pred), "r2": r2_score(yt, pred)}
            primary = m["r2"]
        else:
            yt = yv.long().view(-1)
            m = {
                "acc": accuracy(yt, ov),
                "balanced_acc": balanced_accuracy(yt, ov),
                "f1_weighted": f1_weighted(yt, ov, num_classes=num_classes),
            }
            primary = m["f1_weighted"]
            if num_classes == 2:
                score = torch.softmax(ov, dim=-1)[:, 1]
                m["auroc"] = auroc_binary(yt, score)
        return m, primary, float(np.mean(losses) if losses else 0.0), yv, ov

    epoch_pbar = tqdm(range(1, epochs + 1), desc="epochs", ncols=120)
    for epoch in epoch_pbar:
        extractor.dit.train()
        head.train()
        losses = []

        running = {"loss_sum": 0.0, "n": 0, "correct": 0}
        train_pbar = tqdm(train_loader, desc=f"train[{epoch}/{epochs}]", leave=False, ncols=120)
        for step, batch in enumerate(train_pbar, start=1):
            x = batch["x"].to(device=device)
            y = batch["y"].to(device=device)

            opt.zero_grad(set_to_none=True)
            feat = extractor.extract_features(
                x,
                enable_grad=True,
                noise_ids=list(batch["subject"]),
                noise_mode=noise_mode,
            )
            out = head(feat)
            if task_type == "regression":
                loss = loss_fn(out, y.view(-1, 1))
            else:
                loss = loss_fn(out, y.long().view(-1))
            loss.backward()
            opt.step()

            li = float(loss.item())
            losses.append(li)
            running["loss_sum"] += li
            running["n"] += int(x.shape[0])

            if task_type == "classification":
                pred = torch.argmax(out.detach(), dim=-1)
                yy = y.long().view(-1)
                running["correct"] += int((pred == yy).sum().item())
                acc = running["correct"] / max(1, running["n"])
                train_pbar.set_postfix(loss=f"{running['loss_sum']/max(1, step):.4f}", acc=f"{acc:.3f}")
            else:
                mse_b = torch.mean((out.detach().view(-1) - y.view(-1)) ** 2).item()
                train_pbar.set_postfix(loss=f"{running['loss_sum']/max(1, step):.4f}", mse=f"{mse_b:.4f}")

        val_m, primary, val_loss, _yv, _ov = eval_loader(valid_loader)
        history["train_loss"].append(float(np.mean(losses) if losses else 0.0))
        history["val_metric"].append(float(primary))
        history["val_loss"].append(float(val_loss))

        if task_type == "regression":
            epoch_pbar.set_postfix(
                train_loss=f"{history['train_loss'][-1]:.4f}",
                val_r2=f"{val_m.get('r2', 0.0):.3f}",
                val_pearson=f"{val_m.get('pearson', 0.0):.3f}",
            )
        else:
            epoch_pbar.set_postfix(
                train_loss=f"{history['train_loss'][-1]:.4f}",
                val_acc=f"{val_m.get('acc', 0.0):.3f}",
                val_f1w=f"{val_m.get('f1_weighted', 0.0):.3f}",
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
                    "val_acc": float(val_m.get("acc", 0.0)),
                    "val_f1_weighted": float(val_m.get("f1_weighted", 0.0)),
                    "val_pearson": float(val_m.get("pearson", 0.0)),
                    "val_r2": float(val_m.get("r2", 0.0)),
                }
            )

        improved = best_val_loss is None or float(val_loss) < best_val_loss
        if improved:
            best_val_loss = float(val_loss)
            best_state = {
                "dit": {k: v.detach().cpu().clone() for k, v in dit.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "cfg": cfg,
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
        raise RuntimeError("No best state")

    best_path = os.path.join(out_dir, "best.pt")
    best_state = torch.load(best_path, map_location=device)
    dit.load_state_dict(best_state["dit"], strict=True)
    head.load_state_dict(best_state["head"], strict=True)

    train_m, _p, _train_loss_eval, ytr, otr = eval_loader(train_loader)
    valid_m, _p, valid_loss_eval, yv, ov = eval_loader(valid_loader)
    test_m, _primary, test_loss_eval, yt, ot = eval_loader(test_loader)

    save_json(
        {
            "task": {"type": task_type, "num_classes": int(num_classes)},
            "best_val_loss": float(best_state["best_val_loss"]),
            "best_val_primary": float(best_state["best_val_primary"]),
            "train": {"loss": float(_train_loss_eval), "metrics": train_m},
            "valid": {"loss": float(valid_loss_eval), "metrics": valid_m},
            "test": {"loss": float(test_loss_eval), "metrics": test_m},
        },
        os.path.join(out_dir, "metrics.json"),
    )
    plot_learning_curves(history, os.path.join(out_dir, "plots", "learning_curves.png"), title="LoRA Fine-tune (raw)")

    if task_type == "regression":
        yp_v = ov.detach().cpu().numpy().reshape(-1)
        yt_v = yv.detach().cpu().numpy().reshape(-1)
        plot_regression_scatter(yt_v, yp_v, os.path.join(out_dir, "plots", "scatter_valid.png"), title="Valid")
    else:
        cm = confusion_matrix(yv, ov, num_classes=int(num_classes))
        plot_confusion_matrix(cm, os.path.join(out_dir, "plots", "confusion_valid.png"), title="Valid")
        if num_classes == 2:
            score = torch.softmax(ot, dim=-1)[:, 1].detach().cpu().numpy().reshape(-1)
            yt_np = yt.detach().cpu().numpy().reshape(-1)
            fpr, tpr = compute_roc_curve_binary(yt_np, score)
            plot_roc_curve(fpr, tpr, os.path.join(out_dir, "plots", "roc_test.png"), title="Test", auc=test_m.get("auroc"))

    print(f"Done. out_dir={out_dir} test={test_m}")


if __name__ == "__main__":
    main()
