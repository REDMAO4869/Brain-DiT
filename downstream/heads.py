from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch.nn as nn


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class HeadSpec:
    task_type: TaskType
    num_classes: int


def build_head(*, d_model: int, task: HeadSpec, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> nn.Module:
    d = int(d_model)
    if task.task_type == "regression":
        return nn.Linear(d, 1)
    if task.num_classes <= 1:
        raise ValueError("num_classes must be >= 2 for classification")
    if hidden_dim is None or int(hidden_dim) <= 0:
        return nn.Linear(d, int(task.num_classes))
    h = int(hidden_dim)
    return nn.Sequential(
        nn.Linear(d, h),
        nn.ReLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(h, int(task.num_classes)),
    )
