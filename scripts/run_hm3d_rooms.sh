#!/bin/bash
# Room segmentation on every HM3DSem-Walks scene, evaluated with the HOV-SG
# protocol (scripts/eval_rooms.py).
#
# Run inside the lexisg container, from /workspace/lexi-sg:
#   bash scripts/run_hm3d_rooms.sh
#
# Environment overrides: DATA_ROOT, OUT_ROOT, SCENES, GPU.

set -u
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/workspace/scratch/hm3d}
OUT_ROOT=${OUT_ROOT:-/workspace/scratch/output_hm3d_rooms}
SCENES=${SCENES:-"00824 00829 00843 00847 00873 00877 00890"}
GPU=${GPU:-0}

run_variant() {
  local variant=$1 gpu=$2; shift 2
  for scene in $SCENES; do
    local out="$OUT_ROOT/$variant/$scene"
    if [ -f "$out/rooms.json" ]; then
      echo "[$variant] $scene: done, skipping"
      continue
    fi
    mkdir -p "$OUT_ROOT/logs"
    local log="$OUT_ROOT/logs/${variant}_${scene}.log"
    echo "[$variant] $scene: running (gpu $gpu, log $log)"
    if python main.py --config config/datasets/hm3d.yaml \
        --set data_path="$DATA_ROOT/$scene" \
        --set output_folder="$out" \
        --set model_devices.reconstruction=$gpu \
        --set model_devices.semantic=$gpu \
        --set object_segmentation.enabled=false \
        "$@" > "$log" 2>&1 \
      && python scripts/eval_rooms.py --output_dir "$out" --data_path "$DATA_ROOT/$scene" \
        --pose_convention opengl --exclude_label doorway >> "$log" 2>&1; then
      echo "[$variant] $scene: ok"
    else
      echo "[$variant] $scene: FAILED, see $log"
    fi
  done
}

run_variant original "$GPU"

OUT_ROOT="$OUT_ROOT" SCENES="$SCENES" python - <<'EOF' | tee "$OUT_ROOT/summary.txt"
import json, os
root, scenes = os.environ["OUT_ROOT"], os.environ["SCENES"].split()
keys = ["prec@IoU=0.5", "recall@IoU=0.5", "ap", "hydra_prec", "hydra_recall"]
print(f"{'variant':9s} {'scene':6s} {'gt':>3s} {'pred':>4s} " + " ".join(f"{k.split('@')[0]:>12s}" for k in keys))
for variant in ("original",):
    rows = []
    for s in scenes:
        f = os.path.join(root, variant, s, "rooms.json")
        if not os.path.exists(f):
            print(f"{variant:9s} {s:6s} missing")
            continue
        m = json.load(open(f))
        rows.append(m)
        print(f"{variant:9s} {s:6s} {m['gt']:3d} {m['pred']:4d} " + " ".join(f"{m[k]:12.4f}" for k in keys))
    if rows:
        print(f"{variant:9s} {'mean':6s} {'':3s} {'':4s} " + " ".join(f"{sum(r[k] for r in rows)/len(rows):12.4f}" for k in keys))
    print()
EOF
