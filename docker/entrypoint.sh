#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_DEFAULT_ENV:-lexisg}"

git config --global --add safe.directory /workspace/lexi-sg >/dev/null 2>&1 || true

pip install -e /workspace/lexi-sg --no-deps >/dev/null 2>&1 || true

GDINO="${GSA_PATH:-/workspace/lexi-sg/thirdparty/Grounded-Segment-Anything}/GroundingDINO"
if [ -d "${GDINO}" ] && ! ls "${GDINO}"/groundingdino/_C*.so >/dev/null 2>&1; then
    echo "[entrypoint] GroundingDINO CUDA extension missing in ${GDINO}; building in place..."
    if /workspace/lexi-sg/scripts/build_groundingdino.sh "${GDINO}" >/tmp/groundingdino_build.log 2>&1; then
        echo "[entrypoint] GroundingDINO CUDA extension built."
    else
        echo "[entrypoint] WARNING: GroundingDINO build failed; see /tmp/groundingdino_build.log. It will run on CPU."
    fi
fi

exec "$@"
