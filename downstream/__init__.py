from __future__ import annotations

from ._path import ensure_repo_root

# Ensure repo root is importable when this package is imported as a module.
ensure_repo_root()

__all__ = ["ensure_repo_root"]
