#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SampleRow:
    subject: str
    path: str


def _write_split_csv(path: str, rows: List[SampleRow]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Subject", "Path"])
        w.writeheader()
        for r in rows:
            w.writerow({"Subject": r.subject, "Path": r.path})


def _write_stage1_metadata(path: str, subjects: List[str], rng: np.random.Generator) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["Subject", "age", "Gender", "ADHD_DX", "PPMI_DX", "ABIDE_DX", "ADNI_DX"]
    adhd_labels = ["ADHD-Combined", "ADHD-Inattentive", "ADHD-Hyperactive/Impulsive"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for sid in subjects:
            idx = int(sid.split("-")[-1])
            row = {
                "Subject": sid,
                "age": float(rng.integers(18, 65)),
                "Gender": "M" if (idx % 2 == 0) else "F",
                "ADHD_DX": "",
                "PPMI_DX": "",
                "ABIDE_DX": "",
                "ADNI_DX": "",
            }

            # Mimic the project metadata style: one diagnosis column is active, others blank.
            family = idx % 4
            block = idx // 4
            if family == 0:
                row["ADHD_DX"] = adhd_labels[block % len(adhd_labels)] if (block % 2 == 0) else "Control"
            elif family == 1:
                row["PPMI_DX"] = "PD" if (block % 3 == 0) else "Control"
            elif family == 2:
                row["ABIDE_DX"] = "Autism" if (block % 2 == 1) else "Control"
            else:
                row["ADNI_DX"] = "AD" if (block % 2 == 0) else "Control"

            w.writerow(row)


def _write_downstream_labels(path: str, subjects: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Subject", "target"])
        w.writeheader()
        for sid in subjects:
            idx = int(sid.split("-")[-1])
            w.writerow({"Subject": sid, "target": int(idx % 2)})


def main() -> None:
    p = argparse.ArgumentParser(description="Generate toy fMRI data and CSV splits for BrainDIT demos")
    p.add_argument("--out_dir", default="toy_data/demo", help="Output root directory")
    p.add_argument("--num_samples", type=int, default=60)
    p.add_argument("--seq_len", type=int, default=40)
    p.add_argument("--roi_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.num_samples < 10:
        raise ValueError("num_samples must be >= 10")

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.abspath(args.out_dir)
    npy_dir = os.path.join(out_dir, "npy")
    split_dir = os.path.join(out_dir, "splits")
    label_dir = os.path.join(out_dir, "labels")
    os.makedirs(npy_dir, exist_ok=True)

    subjects: List[str] = []
    rows: List[SampleRow] = []

    for i in range(args.num_samples):
        sid = f"sub-{i:05d}"
        x = rng.normal(loc=0.0, scale=1.0, size=(args.seq_len, args.roi_dim)).astype(np.float32)
        if i % 2 == 1:
            x[:, : min(8, args.roi_dim)] += 0.3
        path = os.path.join(npy_dir, f"{sid}.npy")
        np.save(path, x)
        subjects.append(sid)
        rows.append(SampleRow(subject=sid, path=path))

    n_train = int(args.num_samples * 0.7)
    n_val = int(args.num_samples * 0.15)
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    _write_split_csv(os.path.join(split_dir, "train.csv"), train_rows)
    _write_split_csv(os.path.join(split_dir, "val.csv"), val_rows)
    _write_split_csv(os.path.join(split_dir, "test.csv"), test_rows)

    _write_stage1_metadata(os.path.join(label_dir, "metadata_stage1.csv"), subjects, rng)
    _write_downstream_labels(os.path.join(label_dir, "labels_downstream.csv"), subjects)

    print("[toy] generated:")
    print(f"  out_dir: {out_dir}")
    print(f"  npy: {len(rows)} files, shape=({args.seq_len}, {args.roi_dim})")
    print(f"  splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")


if __name__ == "__main__":
    main()
