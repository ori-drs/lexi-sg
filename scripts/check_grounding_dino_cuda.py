#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lexisg.models.grounded_sam_wrapper import GroundingDINOModel, _bootstrap_repo_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether GroundingDINO can run on CUDA in the current environment."
    )
    parser.add_argument(
        "--gsa-path",
        default=None,
        help="Optional Grounded-Segment-Anything repo root. Falls back to GSA_PATH/default search paths.",
    )
    args = parser.parse_args()

    _bootstrap_repo_paths(gsa_repo_root=args.gsa_path, ram_repo_root=None)

    torch_cuda_available = bool(torch.cuda.is_available())
    cuda_probe_ok = False
    cuda_probe_error = None
    if torch_cuda_available:
        try:
            _ = torch.cuda.device_count()
            _ = torch.zeros(1, device="cuda")
            cuda_probe_ok = True
        except Exception as exc:
            cuda_probe_error = str(exc)

    try:
        from groundingdino.util.inference import Model  # noqa: F401

        groundingdino_import_ok = True
        groundingdino_import_error = None
    except Exception as exc:
        groundingdino_import_ok = False
        groundingdino_import_error = str(exc)

    custom_ops_ok = bool(GroundingDINOModel._has_custom_ops())
    effective_device = "cuda" if cuda_probe_ok and groundingdino_import_ok and custom_ops_ok else "cpu"

    print(f"torch.cuda.is_available(): {torch_cuda_available}")
    print(f"torch CUDA probe OK: {cuda_probe_ok}")
    if cuda_probe_error:
        print(f"torch CUDA probe error: {cuda_probe_error}")
    print(f"groundingdino import OK: {groundingdino_import_ok}")
    if groundingdino_import_error:
        print(f"groundingdino import error: {groundingdino_import_error}")
    print(f"groundingdino custom ops OK: {custom_ops_ok}")
    print(f"GroundingDINO effective device in this pipeline: {effective_device}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
