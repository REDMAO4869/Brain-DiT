from __future__ import annotations

from dataclasses import dataclass
import csv
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_MISSING_TOKENS = {"", "nan", "na", "none", "null"}
_CONTROL_TOKENS = {"control", "cn", "normal", "healthy", "hc", "0", "0.0", "0.00", "0.000"}
_GENDER_MAP = {
    "m": 1.0,
    "male": 1.0,
    "1": 1.0,
    "1.0": 1.0,
    "f": 0.0,
    "female": 0.0,
    "0": 0.0,
    "0.0": 0.0,
}


def _normalize_text(v: object) -> str:
    return str(v).strip()


def _normalize_text_lower(v: object) -> str:
    return _normalize_text(v).lower()


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    s = _normalize_text_lower(v)
    return s in _MISSING_TOKENS


def _parse_float(v: object) -> Tuple[float, float]:
    if _is_missing(v):
        return 0.0, 0.0
    try:
        return float(str(v)), 1.0
    except Exception:
        return 0.0, 0.0


def _parse_gender(v: object) -> Tuple[float, float]:
    if _is_missing(v):
        return 0.0, 0.0
    key = _normalize_text_lower(v)
    if key in _GENDER_MAP:
        return float(_GENDER_MAP[key]), 1.0
    return 0.0, 0.0


def _normalize_dx(v: object) -> Optional[str]:
    if _is_missing(v):
        return None
    key = _normalize_text_lower(v)
    if key in _CONTROL_TOKENS:
        return "control"
    return key


@dataclass(frozen=True)
class LabelTableStats:
    rows_total: int
    subjects_unique: int
    duplicates: int
    missing_age: int
    missing_gender: int


class TabularConditioner:
    def __init__(
        self,
        *,
        label_table_path: str,
        subject_col: str = "Subject",
        age_col: str = "age",
        gender_col: str = "Gender",
        dx_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.subject_col = str(subject_col)
        self.age_col = str(age_col)
        self.gender_col = str(gender_col)
        self.dx_cols = list(dx_cols) if dx_cols is not None else ["ADHD_DX", "PPMI_DX", "ABIDE_DX", "ADNI_DX"]

        rows: Dict[str, Dict[str, object]] = {}
        duplicates = 0
        rows_total = 0
        with open(label_table_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Label table has no header: {label_table_path}")
            missing_cols = [c for c in [self.subject_col, self.age_col, self.gender_col, *self.dx_cols] if c not in reader.fieldnames]
            if missing_cols:
                raise ValueError(f"Label table missing columns: {missing_cols}")
            for r in reader:
                rows_total += 1
                subject = _normalize_text(r.get(self.subject_col, ""))
                if subject == "":
                    continue
                if subject in rows:
                    duplicates += 1
                    continue
                rows[subject] = {k: r.get(k) for k in [self.age_col, self.gender_col, *self.dx_cols]}

        self._rows = rows
        self.stats = LabelTableStats(
            rows_total=rows_total,
            subjects_unique=len(rows),
            duplicates=duplicates,
            missing_age=0,
            missing_gender=0,
        )

        self.dx_label_maps: Dict[str, Dict[str, int]] = {}
        for col in self.dx_cols:
            labels = set()
            for r in rows.values():
                norm = _normalize_dx(r.get(col))
                if norm is not None:
                    labels.add(norm)
            labels.add("control")
            ordered = ["control"] + sorted([l for l in labels if l != "control"])
            self.dx_label_maps[col] = {label: i for i, label in enumerate(ordered)}

        self._tabular_by_subject: Dict[str, List[float]] = {}
        missing_age = 0
        missing_gender = 0
        for subject, r in rows.items():
            feats: List[float] = []
            age_val, age_mask = _parse_float(r.get(self.age_col))
            if age_mask == 0.0:
                missing_age += 1
            feats.extend([age_val, age_mask])

            gender_val, gender_mask = _parse_gender(r.get(self.gender_col))
            if gender_mask == 0.0:
                missing_gender += 1
            feats.extend([gender_val, gender_mask])

            for col in self.dx_cols:
                norm = _normalize_dx(r.get(col))
                if norm is None:
                    norm = "control"
                mapping = self.dx_label_maps[col]
                idx = mapping.get(norm, 0)
                onehot = [0.0] * len(mapping)
                onehot[idx] = 1.0
                feats.extend(onehot)

            self._tabular_by_subject[subject] = feats

        self.stats = LabelTableStats(
            rows_total=rows_total,
            subjects_unique=len(rows),
            duplicates=duplicates,
            missing_age=missing_age,
            missing_gender=missing_gender,
        )

        self.tabular_dim = len(next(iter(self._tabular_by_subject.values()))) if self._tabular_by_subject else 0

    def has_subject(self, subject: str) -> bool:
        return str(subject) in self._tabular_by_subject

    def filter_subjects(self, subjects: Iterable[str]) -> List[str]:
        return [str(s) for s in subjects if self.has_subject(s)]

    def encode_subjects(
        self,
        subjects: Sequence[str],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        feats: List[List[float]] = []
        for s in subjects:
            key = str(s)
            if key not in self._tabular_by_subject:
                raise KeyError(f"Subject not found in label table: {key}")
            feats.append(self._tabular_by_subject[key])
        t = torch.tensor(feats, dtype=torch.float32)
        if device is not None or dtype is not None:
            t = t.to(device=device, dtype=dtype)
        return t
