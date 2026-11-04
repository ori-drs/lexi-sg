#!/usr/bin/env bash
# Download the checkpoints lexi-sg needs into checkpoints/.
#
# GroundingDINO and RAM are public and fetched automatically. DINOv3 is
# gated behind Meta's own licence acceptance and cannot be automated; this
# script prints instructions for it instead.
#
# Safe to re-run: existing files are verified against the server's reported
# size and left alone if they already match, rather than re-downloaded.
#
# Usage: bash scripts/fetch_checkpoints.sh [checkpoints_dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/checkpoints}"
mkdir -p "$DEST"

fetch() {
    local name="$1" url="$2"
    local dest="$DEST/$name"

    local remote_size
    remote_size="$(curl -sIL "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')"

    if [ -f "$dest" ] && [ -n "$remote_size" ] && [ "$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest")" = "$remote_size" ]; then
        echo "OK (already present, size matches): $name"
        return
    fi

    echo "Downloading $name ..."
    curl -fL --retry 3 -o "$dest" "$url"

    if [ -n "$remote_size" ]; then
        local got_size
        got_size="$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest")"
        if [ "$got_size" != "$remote_size" ]; then
            echo "error: $name size mismatch (got $got_size, expected $remote_size)." >&2
            echo "The download may be truncated -- delete $dest and re-run." >&2
            exit 1
        fi
    fi
    echo "OK: $name"
}

fetch groundingdino_swint_ogc.pth \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

fetch ram_swin_large_14m.pth \
    "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"

echo
DINOV3_BACKBONE="$DEST/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINOV3_TEXT="$DEST/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
if [ -f "$DINOV3_BACKBONE" ] && [ -f "$DINOV3_TEXT" ]; then
    echo "OK (already present): dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    echo "OK (already present): dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
else
    cat <<EOF
DINOv3 weights are gated behind Meta's own licence and can't be fetched
automatically. To get them:

  1. Accept the licence at:
     https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
  2. You'll be given a signed base URL. Download both files from the
     dinov3_vitl16/ path into $DEST:
       dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
       dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth

  e.g.:
     cd $DEST
     wget "<signed-base-url>/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
     wget "<signed-base-url>/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"

Signed URLs expire quickly -- download promptly after accepting the licence.
Keep the filenames exactly as above; lexisg/models/dino_handler.py looks for
them by name.
EOF
fi
