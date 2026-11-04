#!/usr/bin/env bash
# Build GroundingDINO's CUDA extension in place, for whichever GPU is present.
#
# Used by both docker/entrypoint.sh and scripts/setup_env.sh, so the two
# install paths can't drift apart on this step.
#
# Must build IN PLACE (setup.py build_ext --inplace), not via a normal
# `pip install`: lexisg/models/grounded_sam_wrapper.py's _bootstrap_repo_paths
# unconditionally inserts the GroundingDINO source tree at the front of
# sys.path, ahead of anything in site-packages. A `pip install .` puts the
# compiled _C*.so in site-packages, where it is never reached -- the
# unbuilt source-tree copy always wins the import, silently falling back to
# CPU regardless of whether the site-packages build succeeded.
#
# Usage: build_groundingdino.sh <path-to-GroundingDINO-dir>
set -euo pipefail

GDINO="$1"
CSRC="$GDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn"

# Upstream GroundingDINO predates PyTorch 2.7's removal of Tensor::type().
sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' "$CSRC/ms_deform_attn.h"
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' "$CSRC/ms_deform_attn_cuda.cu"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; c=torch.cuda.get_device_capability(); print(f"{c[0]}.{c[1]}")' 2>/dev/null || echo "8.0 8.6 8.9 9.0 12.0")"
fi
export TORCH_CUDA_ARCH_LIST
echo "Building GroundingDINO CUDA extension for arch(es): $TORCH_CUDA_ARCH_LIST"

( cd "$GDINO" && AM_I_DOCKER=true BUILD_WITH_CUDA=true python setup.py build_ext --inplace )
