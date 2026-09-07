from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from ._path import ensure_repo_root

ensure_repo_root()


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    return float(torch.mean((y_pred - y_true) ** 2).item())


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    return float(torch.mean(torch.abs(y_pred - y_true)).item())


def pearsonr(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    y_true = y_true - torch.mean(y_true)
    y_pred = y_pred - torch.mean(y_pred)
    denom = torch.sqrt(torch.sum(y_true**2) * torch.sum(y_pred**2)).clamp_min(eps)
    return float((torch.sum(y_true * y_pred) / denom).item())


def accuracy(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    y_true = y_true.long().view(-1)
    pred = torch.argmax(y_pred_logits, dim=-1).long().view(-1)
    return float(torch.mean((pred == y_true).float()).item())


def confusion_matrix(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int) -> np.ndarray:
    y_true = y_true.long().view(-1)
    pred = torch.argmax(y_pred_logits, dim=-1).long().view(-1)
    k = int(num_classes)
    if k <= 0:
        raise ValueError("num_classes must be > 0")
    cm = torch.zeros((k, k), dtype=torch.int64, device=y_true.device)
    for t, p in zip(y_true, pred):
        ti = int(t.item())
        if ti < 0 or ti >= k:
            continue
        cm[ti, int(p.item())] += 1
    return cm.detach().cpu().numpy()


def f1_weighted(y_true: torch.Tensor, y_pred_logits: torch.Tensor, num_classes: int, eps: float = 1e-12) -> float:
    cm = confusion_matrix(y_true, y_pred_logits, num_classes=num_classes).astype(np.float64)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    precision = tp / np.maximum(tp + fp, eps)
    recall = tp / np.maximum(tp + fn, eps)
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, eps)

    if np.sum(support) <= 0:
        return 0.0
    w = support / np.sum(support)
    return float(np.sum(w * f1))


def balanced_accuracy(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    y_true = y_true.long().view(-1)
    pred = torch.argmax(y_pred_logits, dim=-1).long().view(-1)
    num_classes = int(torch.max(y_true).item()) + 1
    accs = []
    for c in range(num_classes):
        m = y_true == c
        if int(torch.sum(m).item()) == 0:
            continue
        accs.append(float(torch.mean((pred[m] == c).float()).item()))
    if len(accs) == 0:
        return 0.0
    return float(np.mean(accs))


def auroc_binary(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.long().view(-1).cpu()
    y_score = y_score.float().view(-1).cpu()
    n = int(y_true.numel())
    if n == 0:
        return 0.0

    order = torch.argsort(y_score)
    y_true = y_true[order]

    n_pos = int(torch.sum(y_true == 1).item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = torch.arange(1, n + 1)
    sum_ranks_pos = int(torch.sum(ranks[y_true == 1]).item())

    u = sum_ranks_pos - (n_pos * (n_pos + 1)) // 2
    auc = u / float(n_pos * n_neg)
    return float(auc)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return float((1.0 - (ss_res / ss_tot.clamp_min(eps))).item())


def evaluate_classification(y_true: torch.Tensor, logits: torch.Tensor, *, num_classes: int) -> Dict[str, float]:
    y = y_true.long().view(-1)
    out = {
        "acc": accuracy(y, logits),
        "balanced_acc": balanced_accuracy(y, logits),
        "f1_weighted": f1_weighted(y, logits, num_classes=int(num_classes)),
    }
    if int(num_classes) == 2:
        score = torch.softmax(logits, dim=-1)[:, 1]
        out["auroc"] = auroc_binary(y, score)
    return out


def evaluate_regression(y_true: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    yt = y_true.view(-1).float()
    yp = pred.view(-1).float()
    return {"mse": mse(yt, yp), "mae": mae(yt, yp), "pearson": pearsonr(yt, yp), "r2": r2_score(yt, yp)}


@dataclass(frozen=True)
class EvalResult:
    metrics: Dict[str, float]
