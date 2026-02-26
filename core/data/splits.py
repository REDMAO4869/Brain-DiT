from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class SplitRow:
    subject: str
    path: str


def read_split_csv(
    csv_path: str,
    *,
    subject_col: str = "Subject",
    path_col: str = "Path",
) -> List[SplitRow]:
    """Read a split CSV with an explicit header.

    This is the canonical CSV split format used across the project.

    Requirements:
      - CSV must exist
      - CSV must have a header
      - Must contain exactly-named columns `subject_col` and `path_col`

    Returns:
      List of SplitRow(subject, path) in file order.
    """

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    rows: List[SplitRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Split CSV has no header: {csv_path}")
        for i, r in enumerate(reader):
            if subject_col not in r or path_col not in r:
                raise ValueError(
                    f"Split CSV must contain columns {subject_col!r} and {path_col!r}; got {reader.fieldnames}"
                )
            subject = str(r[subject_col]).strip()
            path = str(r[path_col]).strip()
            if subject == "" or path == "":
                raise ValueError(f"Empty {subject_col}/{path_col} at row {i+2} in {csv_path}")
            rows.append(SplitRow(subject=subject, path=path))
    return rows


def _canonical_split_name(split: str) -> str:
    s = str(split).strip().lower()
    if s == "valid":
        return "val"
    return s


CsvSplitsMapping = Dict[str, Dict[str, str]]


def normalize_multi_dataset_csv_splits(
    data_cfg: Mapping[str, object],
    *,
    allow_missing_test: bool = True,
) -> Tuple[CsvSplitsMapping, List[str]]:
    """Normalize flexible YAML schemas into a canonical mapping.

    Canonical output:
      {
        "train": {"HCP": "/path/HCP_train.csv", "CHCP": "/path/CHCP_train.csv"},
        "val":   {...},
        "test":  {...},
      }

    Supported input schemas (under `data_cfg`):

    1) Split-aggregated mapping:
      data:
        splits:
          train:
            HCP: /path/HCP_train.csv
            CHCP: /path/CHCP_train.csv
          val:
            HCP: /path/HCP_val.csv
          test:
            HCP: /path/HCP_test.csv

    2) Stage-3-style single dataset:
      data:
        dataset: HCP
        splits:
          train: {csv: /path/train.csv}
          valid: {csv: /path/valid.csv}
          test: {csv: /path/test.csv}

    3) Dataset-list schema:
      data:
        datasets:
          - name: HCP
            train_csv: ...
            val_csv: ...
            test_csv: ...

    Notes:
      - We do NOT guess CSV column names; they must match `subject_col`/`path_col`.
      - `valid` is accepted as an alias of `val`.
    """

    splits_cfg = data_cfg.get("splits")
    datasets_cfg = data_cfg.get("datasets")

    out: CsvSplitsMapping = {"train": {}, "val": {}, "test": {}}

    # Schema 3: datasets is a list of dicts with explicit train_csv/val_csv/test_csv.
    if isinstance(datasets_cfg, list) and len(datasets_cfg) > 0 and isinstance(datasets_cfg[0], dict):
        for d in datasets_cfg:
            name = str(d.get("name", "")).strip()
            if not name:
                raise ValueError("data.datasets[*].name is required when using CSV splits")
            train_csv = d.get("train_csv")
            val_csv = d.get("val_csv")
            test_csv = d.get("test_csv")
            if train_csv is None or val_csv is None:
                raise ValueError(f"data.datasets[{name}].train_csv and val_csv are required")
            out["train"][name] = str(train_csv)
            out["val"][name] = str(val_csv)
            if test_csv is not None:
                out["test"][name] = str(test_csv)

        ds_names = sorted(set(out["train"].keys()) | set(out["val"].keys()) | set(out["test"].keys()))
        if len(ds_names) == 0:
            raise ValueError("CSV splits resolved to 0 datasets")
        if not allow_missing_test and len(out["test"]) == 0:
            raise ValueError("data.datasets[*].test_csv is required")
        return out, ds_names

    # Schema 1 / 2: data.splits mapping
    if not isinstance(splits_cfg, dict):
        raise ValueError("data.splits must be a mapping when using CSV splits")

    # Stage-3-style single dataset (train: {csv: ...})
    def _maybe_single_dataset_entry(v: object) -> Optional[str]:
        if isinstance(v, dict) and "csv" in v:
            return str(v["csv"])
        return None

    # Detect if `train` looks like {csv: ...}
    single_train = _maybe_single_dataset_entry(splits_cfg.get("train"))
    if single_train is not None:
        dataset_name = data_cfg.get("dataset")
        if dataset_name is None:
            # Try to infer only if an explicit string-list datasets with length 1 exists.
            ds_list = datasets_cfg
            if isinstance(ds_list, list) and len(ds_list) == 1 and isinstance(ds_list[0], str):
                dataset_name = ds_list[0]
        if dataset_name is None:
            raise ValueError("data.dataset (or single-element data.datasets) is required for single-dataset CSV splits")
        ds_name = str(dataset_name)

        out["train"][ds_name] = single_train
        val_key = "val" if "val" in splits_cfg else "valid" if "valid" in splits_cfg else None
        if val_key is None:
            raise ValueError("data.splits.val (or valid) is required when using CSV splits")
        out["val"][ds_name] = str(splits_cfg[val_key]["csv"])
        if "test" in splits_cfg and isinstance(splits_cfg["test"], dict) and "csv" in splits_cfg["test"]:
            out["test"][ds_name] = str(splits_cfg["test"]["csv"])

        ds_names = [ds_name]
        if not allow_missing_test and len(out["test"]) == 0:
            raise ValueError("data.splits.test.csv is required")
        return out, ds_names

    # Schema 1: split -> dataset-> csv path
    for split_key_raw, entry in splits_cfg.items():
        split_key = _canonical_split_name(split_key_raw)
        if split_key not in out:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"data.splits.{split_key_raw} must be a mapping when using multi-dataset CSV splits")
        for ds_name, v in entry.items():
            if isinstance(v, str):
                out[split_key][str(ds_name)] = v
            elif isinstance(v, dict) and "csv" in v:
                out[split_key][str(ds_name)] = str(v["csv"])
            else:
                raise ValueError(
                    f"data.splits.{split_key_raw}.{ds_name} must be a csv path string or {{csv: ...}}, got {type(v)}"
                )

    if len(out["train"]) == 0 or len(out["val"]) == 0:
        raise ValueError("data.splits.train and data.splits.val/valid must be provided for CSV splits")

    ds_names = sorted(set(out["train"].keys()) | set(out["val"].keys()) | set(out["test"].keys()))
    if len(ds_names) == 0:
        raise ValueError("CSV splits resolved to 0 datasets")
    if not allow_missing_test and len(out["test"]) == 0:
        raise ValueError("data.splits.test must be provided for CSV splits")
    return out, ds_names
