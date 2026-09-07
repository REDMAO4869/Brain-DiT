from __future__ import annotations

import argparse
import sys

from downstream._path import ensure_repo_root

ensure_repo_root()


class _ArgvSwap:
    def __init__(self, argv):
        self._new = argv
        self._old = None

    def __enter__(self):
        self._old = sys.argv
        sys.argv = self._new

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self._old
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-3 raw runner")
    p.add_argument(
        "--mode",
        required=True,
        choices=["general", "extract", "probe", "lora", "fullft"],
        help="Which stage-3 job to run",
    )
    p.add_argument("--config", required=True, help="YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "general":
        from downstream.train_general_raw import main as _m

        with _ArgvSwap([sys.argv[0], "--config", args.config]):
            _m()
        return
    if args.mode == "extract":
        from downstream.extract_embeddings import main as _m

        with _ArgvSwap([sys.argv[0], "--config", args.config]):
            _m()
        return
    if args.mode == "probe":
        from downstream.train_linear_probe import main as _m

        with _ArgvSwap([sys.argv[0], "--config", args.config]):
            _m()
        return
    if args.mode == "lora":
        from downstream.train_lora_finetune import main as _m

        with _ArgvSwap([sys.argv[0], "--config", args.config]):
            _m()
        return
    if args.mode == "fullft":
        from downstream.train_full_finetune import main as _m

        with _ArgvSwap([sys.argv[0], "--config", args.config]):
            _m()
        return


if __name__ == "__main__":
    main()
