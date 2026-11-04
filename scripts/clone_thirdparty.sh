#!/usr/bin/env bash
# Clone the third-party repositories into thirdparty/, pinned to tested commits.
# Needed by both setup_env.sh and the Docker build. Safe to re-run.
#   bash scripts/clone_thirdparty.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p thirdparty
clone_pinned() {
    local url="$1" dir="$2" commit="$3"
    shift 3
    if [ ! -d "$dir/.git" ]; then
        git clone "$@" "$url" "$dir"
    fi
    # -c safe.directory: git refuses repos owned by a different user (e.g. run
    # as root in a container against a bind-mounted checkout) without this.
    git -c safe.directory='*' -C "$dir" checkout -q "$commit"
}
clone_pinned https://github.com/IDEA-Research/Grounded-Segment-Anything.git \
    thirdparty/Grounded-Segment-Anything 126abe633ffe333e16e4a0a4e946bc1003caf757 \
    --recurse-submodules
clone_pinned https://github.com/HyeonJaeGil/recognize-anything.git \
    thirdparty/recognize-anything 968b559542f07de90c63ff344e0a8f48913b65cf
clone_pinned https://github.com/HyeonJaeGil/sam2.git \
    thirdparty/sam2 1c51acb7520165145cf866da51bc402f046dc108
clone_pinned https://github.com/HyeonJaeGil/openlex3d.git \
    thirdparty/openlex3d fb89a88454fc9f36907e87e2c66460fa4a5b0807
clone_pinned https://github.com/facebookresearch/map-anything.git \
    thirdparty/map-anything 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
