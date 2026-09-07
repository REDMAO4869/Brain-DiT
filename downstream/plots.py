from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ._path import ensure_repo_root

ensure_repo_root()


def plot_learning_curves(history: Dict[str, List[float]], out_path: str, *, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    for k, v in history.items():
        if len(v) == 0:
            continue
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, *, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.7)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.title(title)
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, out_path: str, *, title: str, auc: Optional[float] = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=(f"AUC={auc:.3f}" if auc is not None else None))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title(title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    if auc is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_path: str, *, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = np.asarray(cm)
    plt.figure(figsize=(5.0, 4.5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 200):
    thresholds = np.linspace(np.max(y_score), np.min(y_score), num_thresholds)
    tpr = []
    fpr = []
    y_true = y_true.astype(np.int64)
    for thr in thresholds:
        pred = (y_score >= thr).astype(np.int64)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        tpr.append(tp / (tp + fn + 1e-12))
        fpr.append(fp / (fp + tn + 1e-12))
    return np.asarray(fpr), np.asarray(tpr)
