from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
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
from downstream.utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 (raw): extract offline embeddings for linear probe")
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
    labels, label_map_used = read_labels_csv(
        data_cfg["labels_csv"],
        subject_col=str(data_cfg["labels_subject_col"]),
        label_col=str(data_cfg["labels_label_col"]),
        label_map=data_cfg.get("label_map", None),
        return_label_map=True,
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
    return ds, missing, label_map_used


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    out_dir = os.path.abspath(str(cfg["output_dir"]))
    ensure_dir(out_dir)

    device_str = str(cfg.get("device", "auto"))
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    dit, cond, schedule, cfg1 = load_stage1_raw(str(cfg["stage1_ckpt_path"]), device=device)

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
    batch_size = int(dl_cfg.get("batch_size", 8))
    num_workers = int(dl_cfg.get("num_workers", 0))

    all_missing = []
    label_map_used_any: Dict[str, int] = {}

    meta: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage1_ckpt_path": str(cfg["stage1_ckpt_path"]),
        "embedding": {"timestep": ec.timestep, "capture_layer": ec.capture_layer, "pool": ec.pool},
        "noise_seed": int(cfg.get("noise_seed", 0)),
        "stage1": {
            "diffusion": cfg1.get("diffusion", {}),
            "dit": cfg1.get("dit", {}),
            "condition": cfg1.get("condition", {}),
        },
        "data": cfg.get("data", {}),
        "label_map_used": label_map_used_any,
        "splits": {},
    }

    noise_mode = str(embed_cfg.get("noise_mode", "per_subject"))

    for split in ("train", "valid", "test"):
        ds, missing, label_map_used = _build_dataset(cfg, split)
        all_missing.extend(missing)
        if label_map_used and not label_map_used_any:
            label_map_used_any.update(label_map_used)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        feats: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        extract_debug = cfg.get("extract_debug", {})
        debug_on = bool(extract_debug.get("enable", False))
        debug_max_batches = int(extract_debug.get("max_batches", 1))
        debug_outdir = os.path.join(out_dir, str(extract_debug.get("outdir", "debug")))

        batch_i = 0
        for batch in tqdm(loader, desc=f"extract[{split}]", ncols=120):
            x = batch["x"].to(device=device)
            do_debug = debug_on and (batch_i < debug_max_batches)
            dbg_batch_dir = os.path.join(debug_outdir, split, f"batch_{batch_i}") if do_debug else None
            feat = extractor.extract_features(
                x,
                enable_grad=False,
                debug=do_debug,
                debug_outdir=dbg_batch_dir,
                noise_ids=list(batch["subject"]),
                noise_mode=noise_mode,
            )
            batch_i += 1
            feats.append(feat.detach().cpu().numpy().astype(np.float32))

            y = batch["y"].detach().cpu().numpy().astype(np.float32)
            ys.append(y)

        emb = np.concatenate(feats, axis=0)
        y = np.concatenate(ys, axis=0)

        emb_path = os.path.join(out_dir, f"{split}_emb.npy")
        y_path = os.path.join(out_dir, f"{split}_y.npy")
        np.save(emb_path, emb)
        np.save(y_path, y)
        meta["splits"][split] = {"num_samples": int(emb.shape[0]), "emb_dim": int(emb.shape[1])}
        print(f"[ok] wrote {emb_path} {y_path} emb={emb.shape} y={y.shape}")

    if len(all_missing) > 0:
        miss_path = os.path.join(out_dir, "missing_labels.csv")
        write_missing_labels_csv(all_missing, miss_path)
        print(f"[warn] missing labels: {len(all_missing)} -> {miss_path}")

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"[ok] wrote {meta_path}")

    print(f"Done. out_dir={out_dir}")


if __name__ == "__main__":
    main()
