from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ._path import ensure_repo_root

ensure_repo_root()

from core.data.splits import SplitRow, read_split_csv
from core.data.dataset import fix_length_time_first


@dataclass(frozen=True)
class MissingLabelRow:
    split: str
    subject: str
    path: str
    reason: str


def read_labels_csv(
    csv_path: str,
    *,
    subject_col: str,
    label_col: str,
    label_map: Optional[Dict[str, Union[int, float]]] = None,
    return_label_map: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, int]]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    raw_by_subject: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {csv_path}")
        if subject_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(
                f"Labels CSV must contain columns {subject_col!r} and {label_col!r}; got {reader.fieldnames}"
            )
        for r in reader:
            subject = str(r[subject_col]).strip()
            raw = str(r[label_col]).strip()
            if subject == "":
                continue
            if raw == "" or raw.lower() in {"nan", "none", "null"}:
                continue
            raw_by_subject[subject] = raw

    if len(raw_by_subject) == 0:
        raise ValueError(f"No valid labels found in {csv_path} using ({subject_col}, {label_col})")

    if label_map is not None:
        map_int: Dict[str, int] = {str(k): int(v) for k, v in label_map.items()}
        out: Dict[str, float] = {}
        unknown = set()
        for subject, raw in raw_by_subject.items():
            if raw not in map_int:
                unknown.add(raw)
                continue
            out[subject] = float(map_int[raw])
        if len(out) == 0:
            raise ValueError(
                "After applying label_map, no labels remained. "
                f"Unknown labels: {sorted(list(unknown))[:20]}"
            )
        return (out, map_int) if return_label_map else out

    out_num: Dict[str, float] = {}
    all_numeric = True
    for subject, raw in raw_by_subject.items():
        try:
            out_num[subject] = float(raw)
        except ValueError:
            all_numeric = False
            break

    if all_numeric:
        return (out_num, {}) if return_label_map else out_num

    uniques = sorted(set(raw_by_subject.values()))
    auto_map = {lab: i for i, lab in enumerate(uniques)}
    out_auto: Dict[str, float] = {subject: float(auto_map[raw]) for subject, raw in raw_by_subject.items()}
    return (out_auto, auto_map) if return_label_map else out_auto


def join_split_with_labels(
    split_rows: Sequence[SplitRow],
    labels: Dict[str, float],
    *,
    split_name: str,
) -> Tuple[List[SplitRow], List[MissingLabelRow]]:
    keep: List[SplitRow] = []
    missing: List[MissingLabelRow] = []
    for r in split_rows:
        if r.subject not in labels:
            missing.append(MissingLabelRow(split=split_name, subject=r.subject, path=r.path, reason="missing_label"))
            continue
        keep.append(r)
    return keep, missing


def write_missing_labels_csv(rows: Sequence[MissingLabelRow], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "Subject", "Path", "reason"])
        w.writeheader()
        for r in rows:
            w.writerow({"split": r.split, "Subject": r.subject, "Path": r.path, "reason": r.reason})


class DownstreamCSVDataset(Dataset):
    def __init__(
        self,
        *,
        rows: Sequence[SplitRow],
        labels: Dict[str, float],
        seq_len: Optional[int],
        crop_mode: str,
        pad_mode: str,
        roi_dim: Optional[int] = None,
        path_prefix: Optional[str] = None,
        strict_seq_len: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.labels = labels
        self.seq_len = seq_len
        self.crop_mode = str(crop_mode)
        self.pad_mode = str(pad_mode)
        self.roi_dim = roi_dim
        self.path_prefix = str(path_prefix) if path_prefix is not None else None
        self.strict_seq_len = bool(strict_seq_len)

        if len(self.rows) == 0:
            raise ValueError("Dataset has 0 rows after join")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path) or self.path_prefix is None:
            return path
        return os.path.join(self.path_prefix, path)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        path = self._resolve_path(r.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        x = np.load(path)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array (T, ROI) in {path}, got shape={x.shape}")
        x = x.astype(np.float32, copy=False)

        if self.roi_dim is not None and x.shape[1] != int(self.roi_dim):
            raise ValueError(f"ROI dim mismatch in {path}: expected {self.roi_dim}, got {x.shape[1]}")

        if self.strict_seq_len and self.seq_len is not None and x.shape[0] != int(self.seq_len):
            raise ValueError(f"Seq len mismatch in {path}: expected T={int(self.seq_len)}, got {int(x.shape[0])}")

        x = fix_length_time_first(x, self.seq_len, crop_mode=self.crop_mode, pad_mode=self.pad_mode)

        y = float(self.labels[r.subject])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "subject": r.subject,
            "path": path,
        }


__all__ = [
    "DownstreamCSVDataset",
    "MissingLabelRow",
    "join_split_with_labels",
    "read_labels_csv",
    "read_split_csv",
    "write_missing_labels_csv",
]
