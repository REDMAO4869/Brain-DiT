from __future__ import annotations

import os
import sys


def ensure_repo_root() -> str:
    """Ensure the repository root is on sys.path.

    This keeps local package imports stable when running scripts via
    `python downstream/...`.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
