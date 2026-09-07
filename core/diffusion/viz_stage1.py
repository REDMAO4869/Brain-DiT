from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional

import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def save_latent_stats(z0, z0_hat, save_path_prefix: str) -> str:
    """Save min/max/mean/std for z0 and z0_hat to a JSON file.

    Returns the written json path.
    """
    z0_np = _to_np(z0)
    z0h_np = _to_np(z0_hat)

    def stats(a: np.ndarray):
        return {
            "shape": list(a.shape),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
        }

    payload = {"z0": stats(z0_np), "z0_hat": stats(z0h_np)}
    out_path = save_path_prefix + "_latent_stats.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return out_path


def save_x_recon_plots(
    x,
    x_hat,
    save_path_prefix: str,
    roi_plot_indices: Optional[Iterable[int]] = None,
    save_timeseries: bool = True,
) -> List[str]:
    """Save GT / Recon / Residual heatmaps (PNG) and optional ROI timeseries plots.

    Shape convention (must not change):
      - x, x_hat are (T, N)
      - Heatmap uses x-axis = T, y-axis = ROI(N)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_np = _to_np(x)
    xh_np = _to_np(x_hat)
    if x_np.ndim != 2 or xh_np.ndim != 2:
        raise ValueError(f"Expected (T,N) arrays, got x={x_np.shape} x_hat={xh_np.shape}")
    if x_np.shape != xh_np.shape:
        raise ValueError(f"Shape mismatch: x={x_np.shape} x_hat={xh_np.shape}")

    t, n = x_np.shape
    residual = x_np - xh_np

    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    out_paths: List[str] = []

    def save_heatmap(arr: np.ndarray, title: str, out_path: str) -> None:
        # arr is (T,N) -> transpose to (N,T) for ROI (y) vs time (x)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(arr.T, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Time (T)")
        ax.set_ylabel("ROI (N)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    p_gt = save_path_prefix + "_heatmap_gt.png"
    p_rec = save_path_prefix + "_heatmap_recon.png"
    p_res = save_path_prefix + "_heatmap_residual.png"
    save_heatmap(x_np, "GT X", p_gt)
    save_heatmap(xh_np, "Recon X_hat", p_rec)
    save_heatmap(residual, "Residual (GT - Recon)", p_res)
    out_paths.extend([p_gt, p_rec, p_res])

    if save_timeseries:
        indices = list(roi_plot_indices) if roi_plot_indices is not None else [0, 100, 200]
        indices = [int(i) for i in indices]
        indices = [i for i in indices if 0 <= i < n]
        if len(indices) > 0:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            tt = np.arange(t)
            for ridx in indices[:3]:
                ax.plot(tt, x_np[:, ridx], label=f"GT roi={ridx}")
                ax.plot(tt, xh_np[:, ridx], linestyle="--", label=f"Recon roi={ridx}")
            ax.set_xlabel("Time (T)")
            ax.set_ylabel("Signal")
            ax.set_title("ROI timeseries: GT vs Recon")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            p_ts = save_path_prefix + "_timeseries.png"
            fig.savefig(p_ts, dpi=150)
            plt.close(fig)
            out_paths.append(p_ts)

    return out_paths
