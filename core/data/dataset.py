from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from core.data.splits import CsvSplitsMapping, SplitRow, normalize_multi_dataset_csv_splits, read_split_csv


def _is_dist_initialized() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _dist_rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if _is_dist_initialized() else 0
    except Exception:
        return 0


def _dist_broadcast_object(obj, src: int = 0):
    """Broadcast a picklable python object from src to all ranks (no-op if not in DDP)."""
    if not _is_dist_initialized():
        return obj
    import torch.distributed as dist

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


DEFAULT_DATA_ROOT = os.environ.get("BRAINDIT_DATA_ROOT", "/path/to/pretraining_data")


class PathWithMeta(str):
    """A path string carrying extra metadata (e.g. dataset_name) without breaking call sites.

    This allows the existing training loops to keep using `os.path.basename(paths[0])`
    and f-strings, while still enabling per-sample dataset attribution via attributes.
    """

    dataset_name: str
    subject_id: Optional[str]

    def __new__(cls, path: str, dataset_name: str, subject_id: Optional[str] = None):
        obj = str.__new__(cls, path)
        obj.dataset_name = dataset_name
        obj.subject_id = subject_id
        return obj

    def __reduce__(self):
        # Ensure this subclass of str can be pickled/unpickled across worker processes.
        return (PathWithMeta, (str(self), getattr(self, "dataset_name", "unknown"), getattr(self, "subject_id", None)))


def build_npy_file_list(
    split_dir: str,
    recursive: bool = False,
    ddp_broadcast: bool = True,
) -> List[str]:
    """List .npy files under split_dir.

    - Default is non-recursive (fast, matches your <DATASET>/<atlas>/<split>/*.npy layout).
    - If recursive=True, will scan subdirs too.
    - If ddp_broadcast=True and torch.distributed is initialized, only rank0 scans
      and broadcasts the resulting list to all ranks (avoids N-way slow directory scans).
    """

    split_dir = str(split_dir)
    if ddp_broadcast and _is_dist_initialized():
        if _dist_rank() == 0:
            files = _scan_npy_files(split_dir, recursive=recursive)
        else:
            files = []
        files = _dist_broadcast_object(files, src=0)
        return files

    return _scan_npy_files(split_dir, recursive=recursive)


def _scan_npy_files(split_dir: str, recursive: bool) -> List[str]:
    # Prefer scandir/walk over glob("**") for speed on large directories.
    files: List[str] = []
    if not recursive:
        with os.scandir(split_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".npy"):
                    files.append(entry.path)
    else:
        for root, _dirs, fnames in os.walk(split_dir):
            for name in fnames:
                if name.endswith(".npy"):
                    files.append(os.path.join(root, name))
    files.sort()
    return files


def _center_crop_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    t = x.shape[0]
    if t <= target_len:
        return x
    start = (t - target_len) // 2
    return x[start : start + target_len]


def _random_crop_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    t = x.shape[0]
    if t <= target_len:
        return x
    start = random.randint(0, t - target_len)
    return x[start : start + target_len]


def fix_length_time_first(
    x: np.ndarray,
    seq_len: Optional[int],
    crop_mode: str = "center",
    pad_mode: str = "zeros",
) -> np.ndarray:
    """Ensure time length equals seq_len for x shaped (T, ROI)."""
    if seq_len is None:
        return x

    t, roi = x.shape
    if t > seq_len:
        if crop_mode == "center":
            x = _center_crop_1d(x, seq_len)
        elif crop_mode == "random":
            x = _random_crop_1d(x, seq_len)
        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")
    elif t < seq_len:
        if pad_mode != "zeros":
            raise ValueError(f"Unknown pad_mode: {pad_mode}")
        pad = np.zeros((seq_len - t, roi), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)

    return x


@dataclass
class Sample:
    x: torch.Tensor  # (T, ROI)
    path: str


class HCPTimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        seq_len: Optional[int],
        crop_mode: str,
        pad_mode: str,
        roi_dim: Optional[int] = None,
        seed: int = 0,
        dataset_name: Optional[str] = None,
        scan_recursive: bool = True,
        strict_seq_len: bool = False,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.split_dir = os.path.join(data_root, split)
        self.seq_len = seq_len
        self.crop_mode = crop_mode
        self.pad_mode = pad_mode
        self.roi_dim = roi_dim
        self.seed = seed
        self.dataset_name = dataset_name or _infer_dataset_name_from_root(data_root)
        self.scan_recursive = bool(scan_recursive)
        self.strict_seq_len = bool(strict_seq_len)

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.files = build_npy_file_list(
            self.split_dir,
            recursive=self.scan_recursive,
            ddp_broadcast=True,
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        x = np.load(path)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array (T, ROI) in {path}, got shape={x.shape}")

        x = x.astype(np.float32, copy=False)

        if self.roi_dim is not None and x.shape[1] != self.roi_dim:
            raise ValueError(
                f"ROI dim mismatch in {path}: expected ROI={self.roi_dim}, got {x.shape[1]}"
            )

        if self.strict_seq_len and self.seq_len is not None and x.shape[0] != int(self.seq_len):
            raise ValueError(
                f"Seq len mismatch in {path}: expected T={int(self.seq_len)}, got {int(x.shape[0])}"
            )

        # Deterministic randomness per worker/epoch is handled by DataLoader seed;
        # here we only use Python's random when crop_mode=random.
        x = fix_length_time_first(x, self.seq_len, crop_mode=self.crop_mode, pad_mode=self.pad_mode)

        return torch.from_numpy(x), PathWithMeta(path, dataset_name=self.dataset_name)


class CSVTimeSeriesDataset(Dataset):
    """Time-series dataset defined by an explicit split CSV.

    CSV must include exact columns (subject_col, path_col) as enforced by `read_split_csv`.
    Each sample loads a .npy shaped (T, ROI).

    Returns:
      (x, path) where path is a `PathWithMeta` carrying dataset_name and subject_id.
    """

    def __init__(
        self,
        *,
        rows: Sequence[SplitRow],
        seq_len: Optional[int],
        crop_mode: str,
        pad_mode: str,
        roi_dim: Optional[int],
        dataset_name: str,
        path_prefix: Optional[str] = None,
        strict_seq_len: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.seq_len = seq_len
        self.crop_mode = str(crop_mode)
        self.pad_mode = str(pad_mode)
        self.roi_dim = roi_dim
        self.dataset_name = str(dataset_name)
        self.path_prefix = str(path_prefix) if path_prefix is not None else None
        self.strict_seq_len = bool(strict_seq_len)

        if len(self.rows) == 0:
            raise ValueError(f"CSVTimeSeriesDataset has 0 rows for dataset={self.dataset_name}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path) or self.path_prefix is None:
            return path
        return os.path.join(self.path_prefix, path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        r = self.rows[idx]
        path = self._resolve_path(r.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        x = np.load(path)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array (T, ROI) in {path}, got shape={x.shape}")
        x = x.astype(np.float32, copy=False)

        if self.roi_dim is not None and x.shape[1] != int(self.roi_dim):
            raise ValueError(f"ROI dim mismatch in {path}: expected ROI={self.roi_dim}, got {x.shape[1]}")

        if self.strict_seq_len and self.seq_len is not None and x.shape[0] != int(self.seq_len):
            raise ValueError(f"Seq len mismatch in {path}: expected T={int(self.seq_len)}, got {int(x.shape[0])}")

        x = fix_length_time_first(x, self.seq_len, crop_mode=self.crop_mode, pad_mode=self.pad_mode)

        return torch.from_numpy(x), PathWithMeta(path, dataset_name=self.dataset_name, subject_id=r.subject)


def _infer_dataset_name_from_root(data_root: str) -> str:
    """Best-effort inference for legacy configs.

    Legacy configs typically pass `.../<DATASET>/<atlas>` as data_root.
    We infer dataset_name as the parent directory name.
    """

    root = os.path.abspath(str(data_root))
    parent = os.path.basename(os.path.dirname(root))
    return parent or "unknown"


def _normalize_datasets(datasets: Any) -> List[str]:
    if datasets is None:
        return []
    if isinstance(datasets, (tuple, list)):
        out = [str(x) for x in datasets]
    else:
        raise ValueError("cfg['datasets'] must be a YAML list, e.g. ['HCP','ABCD']")
    out = [x for x in out if x]
    if len(out) == 0:
        raise ValueError("cfg['datasets'] is empty")
    return out


def build_dataset_roots(
    *,
    data_root: str,
    datasets: Sequence[str],
    atlas: str,
) -> Dict[str, str]:
    return {ds: os.path.join(str(data_root), str(ds), str(atlas)) for ds in datasets}


def build_split_datasets(
    *,
    data_root: str,
    datasets: Sequence[str],
    atlas: str,
    split: str,
    seq_len: Optional[int],
    crop_mode: str,
    pad_mode: str,
    roi_dim: Optional[int],
    seed: int,
    scan_recursive: bool = True,
    strict_seq_len: bool = False,
) -> Tuple[Dict[str, HCPTimeSeriesDataset], Dict[str, int]]:
    per_ds: Dict[str, HCPTimeSeriesDataset] = {}
    counts: Dict[str, int] = {}
    roots = build_dataset_roots(data_root=data_root, datasets=datasets, atlas=atlas)
    for ds_name, ds_root in roots.items():
        ds = HCPTimeSeriesDataset(
            data_root=ds_root,
            split=split,
            seq_len=seq_len,
            crop_mode=crop_mode,
            pad_mode=pad_mode,
            roi_dim=roi_dim,
            seed=seed,
            dataset_name=ds_name,
            scan_recursive=scan_recursive,
            strict_seq_len=bool(strict_seq_len),
        )
        per_ds[ds_name] = ds
        counts[ds_name] = len(ds)
    return per_ds, counts


def build_split_datasets_from_csv_splits(
    *,
    csv_splits: CsvSplitsMapping,
    split: str,
    seq_len: Optional[int],
    crop_mode: str,
    pad_mode: str,
    roi_dim: Optional[int],
    dataset_names: Optional[Sequence[str]] = None,
    path_prefix: Optional[str] = None,
    subject_col: str = "Subject",
    path_col: str = "Path",
    strict_seq_len: bool = False,
) -> Tuple[Dict[str, CSVTimeSeriesDataset], Dict[str, int]]:
    """Build per-dataset datasets from explicit CSV split definitions."""

    split_key = str(split).strip().lower()
    if split_key == "valid":
        split_key = "val"
    if split_key not in csv_splits:
        raise ValueError(f"CSV splits missing key: {split_key}. Available: {list(csv_splits.keys())}")

    mapping = csv_splits[split_key]
    names = list(dataset_names) if dataset_names is not None else sorted(mapping.keys())

    per_ds: Dict[str, CSVTimeSeriesDataset] = {}
    counts: Dict[str, int] = {}

    for ds_name in names:
        if ds_name not in mapping:
            continue
        csv_path = str(mapping[ds_name])
        rows = read_split_csv(csv_path, subject_col=str(subject_col), path_col=str(path_col))
        ds = CSVTimeSeriesDataset(
            rows=rows,
            seq_len=seq_len,
            crop_mode=crop_mode,
            pad_mode=pad_mode,
            roi_dim=roi_dim,
            dataset_name=str(ds_name),
            path_prefix=path_prefix,
            strict_seq_len=bool(strict_seq_len),
        )
        per_ds[str(ds_name)] = ds
        counts[str(ds_name)] = len(ds)

    if len(per_ds) == 0:
        raise ValueError(f"No datasets built for split={split_key} from CSV splits")
    return per_ds, counts


def combine_datasets_concat(per_dataset: Dict[str, Dataset]) -> Dataset:
    # Keep dataset order stable for reproducibility.
    names = sorted(per_dataset.keys())
    return ConcatDataset([per_dataset[n] for n in names])


def compute_sample_weights_for_concat_dataset(
    *,
    per_dataset_counts: Dict[str, int],
    dataset_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[str]]:
    """Per-sample weights for a ConcatDataset (datasets concatenated in sorted name order).

    - Balanced sampling: set dataset_weights=None (or all 1.0) => each dataset contributes equally.
      Achieved by per-sample weight ~ 1 / N_i.
    - Weighted sampling: probability(dataset=i) ∝ dataset_weights[i].
      Achieved by per-sample weight ~ w_i / N_i.
    """

    dataset_weights = dataset_weights or {}
    names = sorted(per_dataset_counts.keys())
    weights: List[float] = []
    for name in names:
        n = int(per_dataset_counts[name])
        if n <= 0:
            raise ValueError(f"Dataset {name} has zero samples")
        w = float(dataset_weights.get(name, 1.0))
        if w < 0:
            raise ValueError(f"dataset_weights[{name}] must be >=0")
        per_sample = w / float(n)
        weights.extend([per_sample] * n)
    return weights, names


def format_dataset_summary(
    *,
    data_root: str,
    datasets: Sequence[str],
    atlas: str,
    splits: Dict[str, str],
    split_counts: Dict[str, Dict[str, int]],
) -> str:
    lines: List[str] = []
    lines.append(f"data_root: {data_root}")
    lines.append(f"datasets: {list(datasets)}")
    lines.append(f"atlas: {atlas}")
    lines.append(f"splits: {splits}")
    for split_key in ["train", "val", "test"]:
        if split_key not in split_counts:
            continue
        counts = split_counts[split_key]
        total = sum(int(v) for v in counts.values())
        parts = ", ".join([f"{k}={counts[k]}" for k in sorted(counts.keys())])
        lines.append(f"{split_key}: {parts} | total={total}")
    return "\n".join(lines)


def resolve_data_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve multi-dataset config with backward-compatible fallback.

    New-style config:
      data_root: <Pretraining_OUTPUT>
      datasets: ["HCP", ...]
      atlas: "AA424"

    Legacy-style config:
      data_root: .../<DATASET>/<atlas>
      (no datasets/atlas fields)
    """

    data_section: Mapping[str, object] = cfg.get("data") if isinstance(cfg.get("data"), dict) else cfg

    data_root = str(data_section.get("data_root", cfg.get("data_root", DEFAULT_DATA_ROOT)))
    datasets = _normalize_datasets(data_section.get("datasets", cfg.get("datasets", None)))
    atlas = str(data_section.get("atlas", cfg.get("atlas", "AA424")))

    # Split mapping stays the same for directory-based loading.
    splits = data_section.get("splits", cfg.get("splits", {"train": "train", "val": "val", "test": "test"}))
    if not isinstance(splits, dict):
        raise ValueError("cfg['splits'] must be a mapping like {train: train, val: val, test: test}")

    # New multi-dataset layout is non-recursive by default; legacy kept recursive for backward compatibility.
    scan_recursive = bool(data_section.get("scan_recursive", cfg.get("scan_recursive", False)))

    # Legacy fallback: treat data_root as already pointing to <DATASET>/<atlas>.
    legacy = len(datasets) == 0
    if legacy:
        inferred = _infer_dataset_name_from_root(data_root)
        datasets = [inferred]
        # In legacy mode we do NOT append atlas; caller should use data_root as-is.
        atlas = os.path.basename(os.path.abspath(data_root)) or atlas
        scan_recursive = bool(data_section.get("scan_recursive", cfg.get("scan_recursive", True)))

    # Optional: explicit CSV splits (preferred, reproducible).
    csv_splits: Optional[CsvSplitsMapping] = None
    csv_datasets: Optional[List[str]] = None
    splits_cfg = data_section.get("splits")
    datasets_cfg = data_section.get("datasets")
    has_csv_like = False
    if isinstance(datasets_cfg, list) and len(datasets_cfg) > 0 and isinstance(datasets_cfg[0], dict):
        has_csv_like = True
    if isinstance(splits_cfg, dict):
        train_entry = splits_cfg.get("train")
        val_entry = splits_cfg.get("val") or splits_cfg.get("valid")
        if isinstance(train_entry, dict) or isinstance(val_entry, dict):
            # Either stage3-style {csv: ...} or multi-dataset mapping.
            has_csv_like = True

    if has_csv_like:
        csv_splits, csv_datasets = normalize_multi_dataset_csv_splits(data_section)

    return {
        "data_root": data_root,
        "datasets": datasets,
        "atlas": atlas,
        "splits": splits,
        "legacy": legacy,
        "scan_recursive": scan_recursive,
        "dataset_sampling": str(data_section.get("dataset_sampling", cfg.get("dataset_sampling", "concat"))),
        "eval_dataset_sampling": str(data_section.get("eval_dataset_sampling", cfg.get("eval_dataset_sampling", "concat"))),
        "dataset_weights": data_section.get("dataset_weights", cfg.get("dataset_weights", None)),
        "csv_splits": csv_splits,
        "csv_datasets": csv_datasets,
        "subject_col": str(data_section.get("subject_col", "Subject")),
        "path_col": str(data_section.get("path_col", "Path")),
        "path_prefix": (str(data_section.get("path_prefix")) if data_section.get("path_prefix") is not None else None),
        "strict_seq_len": bool(data_section.get("strict_seq_len", cfg.get("strict_seq_len", False))),
    }
