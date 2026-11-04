#!/usr/bin/env bash
# Pip + venv installation for lexi-sg, as an alternative to Docker.
#
# Run from the repository root:
#   bash scripts/setup_env.sh [venv_path]
#
# venv_path defaults to .venv. Safe to re-run: existing clones and an
# existing venv are reused rather than recreated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${1:-.venv}"

# 3.11 is what this repo's pinned dependencies (docker/requirements.lexisg.txt,
# and older packages like transformers==4.28.1) are tested against. Newer
# interpreters can hit missing prebuilt wheels for those old pins (e.g.
# tokenizers falling back to a source build that needs a Rust compiler).
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: $PYTHON_BIN not found. Install it, e.g.:" >&2
    echo "  sudo apt install python3.11 python3.11-venv" >&2
    exit 1
fi
PYVER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Using $PYTHON_BIN ($PYVER)"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# --- torch: pick a CUDA build matching the driver present, cu128 if none detected ---
CUDA_TAG="cu128"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "NVIDIA driver detected: ${DRIVER_CUDA:-unknown}"
else
    echo "No NVIDIA driver detected; installing CPU-only torch."
    CUDA_TAG=""
fi

if [ -n "$CUDA_TAG" ]; then
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
    pip install torch torchvision torchaudio
fi

# --- everything else, from the same file the Docker build uses ---
pip install --no-cache-dir -r docker/requirements.lexisg.txt
pip install "transformers==4.28.1" "fairscale==0.4.13" "supervision==0.26.0"

# --- third-party repositories ---
bash scripts/clone_thirdparty.sh

pip install --no-cache-dir --no-build-isolation thirdparty/Grounded-Segment-Anything/segment_anything
pip install --no-cache-dir --no-build-isolation thirdparty/map-anything
pip install --no-cache-dir --no-build-isolation thirdparty/recognize-anything
pip install --no-cache-dir --no-build-isolation --no-deps thirdparty/sam2
pip install --no-cache-dir thirdparty/openlex3d
pip install faiss-cpu

# --- GroundingDINO CUDA extension: patch for PyTorch 2.7+, build for the present GPU ---
# Must be built in place, not pip-installed -- see scripts/build_groundingdino.sh.
GDINO="thirdparty/Grounded-Segment-Anything/GroundingDINO"
# build_ext --inplace compiles the extension but doesn't install the package's
# own dependencies (yapf, pycocotools, ...) the way `pip install` would.
pip install --no-cache-dir -r "$GDINO/requirements.txt"
# Building this requires the system's own nvcc, matching the CUDA version
# torch was built with -- there is no pip-installable substitute (the
# nvidia-cuda-nvcc-cu12 wheel ships ptxas and headers but not nvcc itself).
# A mismatch here is common (e.g. a newer driver/toolkit than the pinned
# torch build) and isn't fatal: GroundingDINO falls back to CPU, just slower.
if bash scripts/build_groundingdino.sh "$GDINO"; then
    echo "GroundingDINO CUDA extension built."
else
    echo "WARNING: GroundingDINO CUDA extension build failed -- it will run on CPU."
    echo "This usually means the system CUDA toolkit (nvcc --version) doesn't" \
         "match torch's CUDA build ($(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null))." \
         "Installing a matching CUDA toolkit and re-running this script will fix it."
fi

pip install --no-cache-dir --no-deps -e .
pip check

echo
echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
echo "Then check the installation with:"
echo "  python scripts/check_grounding_dino_cuda.py"
