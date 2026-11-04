#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR_ARG=""
DATASET_ARG=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing value for --output-dir"
        exit 1
      fi
      OUTPUT_DIR_ARG="$1"
      ;;
    -d|--dataset)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing value for --dataset"
        exit 1
      fi
      DATASET_ARG="$1"
      ;;
    -h|--help)
      echo "Usage: $0 [--dataset <replica|scannetpp|hm3d>] [--output-dir <dir>] <log_dir> <scene1> [scene2 ...]"
      echo "Example: $0 --dataset scannetpp --output-dir /tmp/openlex_eval logs/20260304_212637_object_segmentation_fd361ab85f fd361ab85f"
      exit 0
      ;;
    *)
      ARGS+=("$1")
      ;;
  esac
  shift
done
set -- "${ARGS[@]}"

copy_tree_into() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src"/ "$dst"/
  else
    cp -a "$src"/. "$dst"/
  fi
}

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [--dataset <replica|scannetpp|hm3d>] [--output-dir <dir>] <log_dir> <scene1> [scene2 ...]"
  echo "Example: $0 --dataset replica --output-dir /tmp/openlex_eval logs/20260304_212637_object_segmentation_room0 room0"
  exit 1
fi

LOG_DIR="$(realpath "$1")"
shift
SCENES=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENLEX_CONFIG_DIR="${OPENLEX_CONFIG_DIR:-$REPO_ROOT/thirdparty/openlex3d/openlex3d/config}"
OPENLEX_EVAL_BIN="${OPENLEX_EVAL_BIN:-ol3_evaluate_segmentation}"
OPENLEX_DEVICE="${OPENLEX_DEVICE:-cuda}"
OPENLEX_DATASET="${DATASET_ARG:-${OPENLEX_DATASET:-replica}}"
OPENLEX_ALGO="${OPENLEX_ALGO:-$(basename "$LOG_DIR")}"
if [[ -n "$OUTPUT_DIR_ARG" ]]; then
  OPENLEX_OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR_ARG")"
else
  OPENLEX_OUTPUT_DIR="${OPENLEX_OUTPUT_DIR:-$LOG_DIR/openlex3d_eval}"
fi

OPENLEX_REPLICA_PATH="${OPENLEX_REPLICA_PATH:-}"
OPENLEX_SCANNETPP_PATH="${OPENLEX_SCANNETPP_PATH:-}"
OPENLEX_HM3D_PATH="${OPENLEX_HM3D_PATH:-}"

case "$OPENLEX_DATASET" in
  replica|scannetpp|hm3d)
    ;;
  *)
    echo "Unsupported OpenLex3D dataset: $OPENLEX_DATASET"
    exit 1
    ;;
esac

if [[ ! -d "$OPENLEX_CONFIG_DIR" ]]; then
  echo "OpenLex3D config dir not found: $OPENLEX_CONFIG_DIR"
  exit 1
fi

echo "OpenLex3D config: $OPENLEX_CONFIG_DIR"
echo "Dataset: $OPENLEX_DATASET"
echo "Evaluation output: $OPENLEX_OUTPUT_DIR"
echo "Algorithm name: $OPENLEX_ALGO"
echo "Device: $OPENLEX_DEVICE"
echo

ensure_scannetpp_prepared_semantics() {
  local dataset_root="$1"
  local scene="$2"
  local prepared_root="$dataset_root/prepared_semantics"
  local prepared_path="$prepared_root/$scene.pth"

  if [[ -f "$prepared_path" ]]; then
    return 0
  fi

  mkdir -p "$prepared_root"
  python - "$dataset_root" "$scene" << 'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

dataset_root = Path(sys.argv[1])
scene = sys.argv[2]
scene_root = dataset_root / scene / "scans"
mesh_path = scene_root / "mesh_aligned_0.05_semantic.ply"
segments_path = scene_root / "segments.json"
segments_anno_path = scene_root / "segments_anno.json"
out_path = dataset_root / "prepared_semantics" / f"{scene}.pth"

missing = [str(p) for p in (mesh_path, segments_path, segments_anno_path) if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing ScanNet++ files needed to build prepared semantics: {missing}")

mesh = PlyData.read(str(mesh_path))
vertex = mesh["vertex"].data
coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0

with open(segments_path, "r") as f:
    segments = json.load(f)
seg_indices = np.asarray(segments["segIndices"], dtype=np.int64)
if seg_indices.shape[0] != coords.shape[0]:
    raise ValueError(
        f"segments.json vertex count mismatch: segIndices={seg_indices.shape[0]} vs vertices={coords.shape[0]}"
    )

with open(segments_anno_path, "r") as f:
    anno = json.load(f)

segment_to_object = {}
for group in anno.get("segGroups", []):
    object_id = int(group.get("objectId", -1))
    for seg_id in group.get("segments", []):
        segment_to_object[int(seg_id)] = object_id

instance_ids = np.full(seg_indices.shape, -1, dtype=np.int64)
if segment_to_object:
    unique_segments = np.unique(seg_indices)
    for seg_id in unique_segments:
        if int(seg_id) in segment_to_object:
            instance_ids[seg_indices == seg_id] = segment_to_object[int(seg_id)]

payload = {
    "sampled_coords": coords,
    "sampled_colors": colors,
    "sampled_instance_anno_id": instance_ids,
}
torch.save(payload, out_path)
print(f"[INFO] created ScanNet++ prepared semantics: {out_path}")
PY
}

for scene in "${SCENES[@]}"; do
  pred_path="$LOG_DIR/openlex3d_output/$scene"
  if [[ ! -f "$pred_path/point_cloud.pcd" || ! -f "$pred_path/embeddings.npy" || ! -f "$pred_path/index.npy" ]]; then
    echo "[SKIP] Missing prediction files for scene '$scene' at: $pred_path"
    continue
  fi

  extra_overrides=()
  case "$OPENLEX_DATASET" in
    replica)
      if [[ -n "$OPENLEX_REPLICA_PATH" ]]; then
        extra_overrides+=("paths.replica_path=$OPENLEX_REPLICA_PATH")
      fi
      ;;
    scannetpp)
      if [[ -n "$OPENLEX_SCANNETPP_PATH" ]]; then
        ensure_scannetpp_prepared_semantics "$OPENLEX_SCANNETPP_PATH" "$scene"
        extra_overrides+=("paths.scannetpp_path=$OPENLEX_SCANNETPP_PATH")
      fi
      ;;
    hm3d)
      if [[ -n "$OPENLEX_HM3D_PATH" ]]; then
        extra_overrides+=("paths.hm3d_path=$OPENLEX_HM3D_PATH")
      fi
      ;;
  esac

  echo "[RUN] dataset=$OPENLEX_DATASET scene=$scene"
  "$OPENLEX_EVAL_BIN" \
    -cp "$OPENLEX_CONFIG_DIR" \
    -cn eval_segmentation \
    dataset="segmentation/$OPENLEX_DATASET" \
    dataset.scene="$scene" \
    evaluation.algorithm="$OPENLEX_ALGO" \
    evaluation.topn=5 \
    evaluation.predictions_path="$pred_path" \
    evaluation.output_path="$OPENLEX_OUTPUT_DIR" \
    model.device="$OPENLEX_DEVICE" \
    "${extra_overrides[@]}"

  old_scene_dir="$OPENLEX_OUTPUT_DIR/$OPENLEX_ALGO/${OPENLEX_DATASET}_top_5/$scene"
  new_scene_dir="$OPENLEX_OUTPUT_DIR/$OPENLEX_ALGO/top_5/$OPENLEX_DATASET/$scene"
  if [[ ! -f "$old_scene_dir/results.yaml" && -f "$new_scene_dir/results.yaml" ]]; then
    mkdir -p "$(dirname "$old_scene_dir")"
    copy_tree_into "$new_scene_dir" "$old_scene_dir"
  fi

  results_yaml="$old_scene_dir/results.yaml"
  if [[ ! -f "$results_yaml" ]]; then
    echo "[WARN] results.yaml not found: $results_yaml"
    if [[ -f "$new_scene_dir/results.yaml" ]]; then
      echo "[INFO] evaluator wrote results to new layout: $new_scene_dir/results.yaml"
      results_yaml="$new_scene_dir/results.yaml"
    else
      echo
      continue
    fi
  fi

  echo "[RESULT] $scene -> $results_yaml"
  python - "$results_yaml" << 'PY'
import sys
import yaml
from pathlib import Path

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text()) or {}
freq = data.get("freq", {})
if not freq:
    print("  freq: <missing>")
else:
    keys = ["correct", "close", "far", "wrong", "missing", "other"]
    shown = []
    for k in keys:
        if k in freq:
            shown.append(f"{k}={float(freq[k]):.4f}")
    if not shown:
        shown = [f"{k}={float(v):.4f}" for k, v in sorted(freq.items())]
    print("  freq:", ", ".join(shown))
PY

  # Optional: export object-level predicted labels from object embeddings.
  # This is not part of official OpenLex3D metric protocol; it is a debug report.
  if [[ "${OPENLEX_SAVE_OBJECT_LABELS:-1}" == "1" ]]; then
    python - "$pred_path" "$results_yaml" "$OPENLEX_CONFIG_DIR" "$OPENLEX_DEVICE" "$OPENLEX_DATASET" << 'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml

pred_path = Path(sys.argv[1])
results_yaml = Path(sys.argv[2])
config_dir = Path(sys.argv[3])
device = sys.argv[4]
dataset_name = sys.argv[5]

emb_path = pred_path / "embeddings.npy"
if not emb_path.exists():
    print(f"  [WARN] embeddings missing for object-label export: {emb_path}")
    raise SystemExit(0)

paths_yaml = config_dir / "paths" / "paths.yaml"
if not paths_yaml.exists():
    print(f"  [WARN] paths.yaml missing for object-label export: {paths_yaml}")
    raise SystemExit(0)

paths_cfg = yaml.safe_load(paths_yaml.read_text()) or {}
openlex_gt = paths_cfg.get("openlex_gt_path", None)
if not openlex_gt:
    print("  [WARN] openlex_gt_path is not set; skipping object-label export.")
    raise SystemExit(0)

prompt_list_path = Path(openlex_gt) / dataset_name / "prompt_list.txt"
if not prompt_list_path.exists():
    print(f"  [WARN] prompt list missing: {prompt_list_path}")
    raise SystemExit(0)

prompts = [ln.strip() for ln in prompt_list_path.read_text().splitlines() if ln.strip()]
if not prompts:
    print("  [WARN] prompt list is empty; skipping object-label export.")
    raise SystemExit(0)

emb = np.load(emb_path).astype(np.float32)
if emb.ndim != 2 or emb.shape[0] == 0:
    print(f"  [WARN] invalid embeddings shape {emb.shape}; skipping object-label export.")
    raise SystemExit(0)

if device.startswith("cuda") and not torch.cuda.is_available():
    device = "cpu"

tokenizer = open_clip.get_tokenizer("ViT-H-14")

def encode_text_features(run_device: str):
    if run_device.startswith("cuda"):
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        device=run_device,
    )

    text_feat_chunks = []
    batch_size = 128
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            text_tokens = tokenizer(batch_prompts).to(run_device)
            batch_feats = model.encode_text(text_tokens).float()
            batch_feats = batch_feats / batch_feats.norm(dim=-1, keepdim=True)
            text_feat_chunks.append(batch_feats.cpu())

    del model
    if run_device.startswith("cuda"):
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    return torch.cat(text_feat_chunks, dim=0).numpy().astype(np.float32), run_device

try:
    text_feats, used_device = encode_text_features(device)
except torch.OutOfMemoryError:
    if device.startswith("cuda"):
        print("  [WARN] CUDA OOM during object-label export; retrying on CPU.")
        text_feats, used_device = encode_text_features("cpu")
    else:
        raise
except RuntimeError as exc:
    if device.startswith("cuda") and "out of memory" in str(exc).lower():
        print("  [WARN] CUDA OOM during object-label export; retrying on CPU.")
        text_feats, used_device = encode_text_features("cpu")
    else:
        raise

emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), a_min=1e-12, a_max=None)
logits = emb_norm @ text_feats.T

top1_idx = np.argmax(logits, axis=1)
top5_idx = np.argsort(logits, axis=1)[:, -5:][:, ::-1]

out_dir = results_yaml.parent
csv_path = out_dir / "predicted_object_labels_topk.csv"
json_path = out_dir / "predicted_object_labels_topk.json"

rows = []
for obj_idx in range(logits.shape[0]):
    i1 = int(top1_idx[obj_idx])
    i5 = [int(i) for i in top5_idx[obj_idx]]
    row = {
        "object_index": obj_idx,
        "top1_label": prompts[i1],
        "top1_score": float(logits[obj_idx, i1]),
        "top5_labels": [prompts[i] for i in i5],
        "top5_scores": [float(logits[obj_idx, i]) for i in i5],
    }
    rows.append(row)

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["object_index", "top1_label", "top1_score", "top5_labels", "top5_scores"])
    for row in rows:
        w.writerow([
            row["object_index"],
            row["top1_label"],
            row["top1_score"],
            json.dumps(row["top5_labels"]),
            json.dumps(row["top5_scores"]),
        ])

json_path.write_text(json.dumps({
    "dataset": dataset_name,
    "device": used_device,
    "objects": rows,
}, indent=2))
print(f"  [INFO] wrote object-label debug export: {csv_path}")
PY
  fi

  echo
done
