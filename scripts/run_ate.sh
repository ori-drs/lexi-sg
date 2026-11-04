#!/bin/bash
# ATE on every scene of a dataset (object segmentation off).
#   bash scripts/run_ate.sh <hm3d|aod> [main.py args...]
# Environment overrides: DATA_ROOT, OUT_ROOT, SCENES, GPU, VARIANT, CONFIG.

set -u
cd "$(dirname "$0")/.."

DATASET=${1:?usage: run_ate.sh <hm3d|aod> [main.py args...]}
shift

case "$DATASET" in
  hm3d)
    DEFAULT_SCENES="00824 00829 00843 00847 00873 00877 00890"
    POSE_CONVENTION=opengl
    ;;
  aod)
    DEFAULT_SCENES="ff-1 ff-2 ff-3 gf-1 gf-2 gf-3"
    POSE_CONVENTION=opencv
    ;;
  *)
    echo "unknown dataset '$DATASET' (expected hm3d or aod)" >&2
    exit 1
    ;;
esac

CONFIG=${CONFIG:-config/datasets/$DATASET.yaml}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets/$DATASET}
OUT_ROOT=${OUT_ROOT:-/workspace/lexi-sg/output_ate/$DATASET}
SCENES=${SCENES:-$DEFAULT_SCENES}
GPU=${GPU:-0}
VARIANT=${VARIANT:-default}

mkdir -p "$OUT_ROOT/logs"
for scene in $SCENES; do
  out="$OUT_ROOT/$VARIANT/$scene"
  if [ -f "$out/ate.json" ]; then
    echo "[$VARIANT] $scene: done, skipping"
    continue
  fi
  log="$OUT_ROOT/logs/${VARIANT}_${scene}.log"
  echo "[$VARIANT] $scene: running (gpu $GPU, log $log)"
  if python main.py --config "$CONFIG" \
      --set data_path="$DATA_ROOT/$scene" \
      --set output_folder="$out" \
      --set model_devices.reconstruction=$GPU \
      --set model_devices.semantic=$GPU \
      --set object_segmentation.enabled=false \
      "$@" > "$log" 2>&1 \
    && python scripts/eval_ate.py --output_dir "$out" --data_path "$DATA_ROOT/$scene" \
      --pose_convention $POSE_CONVENTION >> "$log" 2>&1; then
    echo "[$VARIANT] $scene: ok"
  else
    echo "[$VARIANT] $scene: FAILED, see $log"
  fi
done

python scripts/summarise_ate.py "$OUT_ROOT" --dataset "$DATASET" | tee "$OUT_ROOT/summary.txt"
