#!/usr/bin/env python3
#
# Run object segmentation detect+track on sampled RGB images from a directory.
#
# Basic usage:
#   python scripts/run_object_segmentation_track_sampled_rgb.py \
#     rgb_dir=/path/to/rgb
#
# Sample a contiguous range with a fixed stride:
#   python scripts/run_object_segmentation_track_sampled_rgb.py \
#     rgb_dir=/path/to/rgb \
#     sampling.start_index=20 \
#     sampling.end_index=220 \
#     sampling.stride=5
#
# Uniformly sample a fixed number of frames from the selected range:
#   python scripts/run_object_segmentation_track_sampled_rgb.py \
#     rgb_dir=/path/to/rgb \
#     sampling.start_index=20 \
#     sampling.end_index=220 \
#     sampling.num_uniform_frames=32
#
# Override object segmentation service parameters with Hydra-style dotlist args:
#   python scripts/run_object_segmentation_track_sampled_rgb.py \
#     rgb_dir=/path/to/rgb \
#     object_segmentation.stages.track.sam2.offload_video_to_cpu=True
#
# Output location:
#   By default results are written under
#   logs/object_segmentation_track_sampled_rgb/<rgb_dir_name>/<timestamp>/
#   You can override this with:
#   output_dir=/your/output/path
#
# Notes:
#   - `sampling.end_index` is exclusive. Use `-1` to mean "until the end".
#   - If `sampling.num_uniform_frames > 0`, it takes precedence over `sampling.stride`.
#   - The script uses `service.yaml` from `config/object_segmentation/` as the default config.

from __future__ import annotations

from datetime import datetime
import os
import shutil
import statistics
import subprocess
import sys

import numpy as np
import yaml
from omegaconf import OmegaConf

from lexisg.utils.config import (
    get_object_segmentation_value,
    get_path,
    load_object_segmentation_service_config,
)
from lexisg.utils.data_loader import get_dataset
from lexisg.tasks.object_segmentation import ObjectSegmentationService


def build_base_config():
    return OmegaConf.create(
        {
            "object_segmentation": {
                "config_path": os.path.join(get_path(), "object_segmentation", "service.yaml"),
            },
            "rgb_dir": None,
            "scene_name": None,
            "subsample": 1,
            "input_image_max_side": -1,
            "output_dir": None,
            "sampling": {
                "start_index": 0,
                "end_index": -1,
                "stride": 1,
                "num_uniform_frames": 0,
            },
        }
    )


def usage():
    script_name = os.path.basename(__file__)
    return "\n".join(
        [
            f"Usage: python scripts/{script_name} rgb_dir=<path> [overrides...]",
            "",
            "Sampling overrides:",
            "  sampling.start_index=0",
            "  sampling.end_index=-1",
            "  sampling.stride=10",
            "  sampling.num_uniform_frames=24",
            "",
            "Notes:",
            "  - `sampling.end_index` is exclusive. `-1` means the end of the sequence.",
            "  - If `sampling.num_uniform_frames > 0`, uniform sampling is used inside the selected range.",
            "  - Otherwise frames are sampled with `sampling.stride`.",
            "",
            "Examples:",
            f"  python scripts/{script_name} rgb_dir=/data/scene/rgb",
            f"  python scripts/{script_name} rgb_dir=/data/scene/rgb sampling.start_index=20 sampling.end_index=220 sampling.stride=5",
            f"  python scripts/{script_name} rgb_dir=/data/scene/rgb sampling.num_uniform_frames=32 object_segmentation.stages.track.sam2.offload_video_to_cpu=True",
        ]
    )


def require_value(config, key: str):
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required override: {key}=...")
    return value


def resolve_output_dir(config) -> str:
    configured = config.get("output_dir")
    if configured not in (None, ""):
        return os.path.abspath(str(configured))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    scene_name = str(config.get("scene_name") or os.path.basename(os.path.abspath(str(config["rgb_dir"]))) or "rgb")
    return os.path.abspath(
        os.path.join(
            "logs",
            "object_segmentation_track_sampled_rgb",
            scene_name,
            timestamp,
        )
    )


def normalize_range(total: int, start_index: int, end_index: int):
    if total <= 0:
        raise ValueError("No RGB frames found.")
    start = max(0, int(start_index))
    end = total if int(end_index) < 0 else min(total, int(end_index))
    if start >= total:
        raise ValueError(f"sampling.start_index={start} is out of range for {total} frames.")
    if end <= start:
        raise ValueError(
            f"Invalid sampled range: start_index={start}, end_index={end}. "
            "`sampling.end_index` is exclusive."
        )
    return start, end


def sample_frame_ids(frame_ids: np.ndarray, stride: int, num_uniform_frames: int) -> np.ndarray:
    if frame_ids.size == 0:
        raise ValueError("No candidate frames remain after applying the start/end range.")
    if int(num_uniform_frames) > 0:
        n = int(num_uniform_frames)
        if n >= int(frame_ids.size):
            return frame_ids.copy()
        sample_local_ids = np.linspace(0, int(frame_ids.size) - 1, num=n, dtype=np.int32)
        return frame_ids[sample_local_ids]
    stride = int(stride)
    if stride <= 0:
        raise ValueError("sampling.stride must be > 0 when sampling.num_uniform_frames <= 0.")
    return frame_ids[::stride]


def build_plan(total_frames: int, sampling_cfg) -> dict:
    start, end = normalize_range(
        total=total_frames,
        start_index=int(sampling_cfg.get("start_index", 0)),
        end_index=int(sampling_cfg.get("end_index", -1)),
    )
    candidate_frame_ids = np.arange(start, end, dtype=np.int32)
    sampled_frame_ids = sample_frame_ids(
        frame_ids=candidate_frame_ids,
        stride=int(sampling_cfg.get("stride", 1)),
        num_uniform_frames=int(sampling_cfg.get("num_uniform_frames", 0)),
    )
    sampled_local_ids = np.searchsorted(candidate_frame_ids, sampled_frame_ids).astype(np.int32)
    sampling_mode = "count" if int(sampling_cfg.get("num_uniform_frames", 0)) > 0 else "stride"
    return {
        "frame_ids": candidate_frame_ids,
        "frame_sampling_mode": sampling_mode,
        "uniform_frame_stride": int(sampling_cfg.get("stride", 1)),
        "num_uniform_frames": int(sampling_cfg.get("num_uniform_frames", 0)),
        "sampled_frame_ids": sampled_frame_ids,
        "sampled_frame_ids_local": sampled_local_ids,
    }


def _numeric_stem(path: str) -> int | None:
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    try:
        return int(stem)
    except ValueError:
        return None


def infer_sampled_fps(rgb_paths: list[str], sampled_frame_ids) -> float | None:
    sampled_ids = [int(fid) for fid in np.asarray(sampled_frame_ids, dtype=np.int32).tolist()]
    if len(sampled_ids) < 2:
        return None

    sampled_timestamps = []
    for fid in sampled_ids:
        if fid < 0 or fid >= len(rgb_paths):
            return None
        timestamp = _numeric_stem(rgb_paths[fid])
        if timestamp is None:
            return None
        sampled_timestamps.append(timestamp)

    diffs = [b - a for a, b in zip(sampled_timestamps, sampled_timestamps[1:]) if b > a]
    if not diffs:
        return None

    median_diff = float(statistics.median(diffs))
    for units_per_second in (1_000_000_000.0, 1_000_000.0, 1_000.0, 1.0):
        fps = units_per_second / median_diff
        if 1.0 <= fps <= 240.0:
            return float(fps)
    return None


def save_track_video(annotated_rgb_dir: str, output_dir: str, fps: float) -> str | None:
    if fps <= 0:
        return None
    if not os.path.isdir(annotated_rgb_dir):
        return None
    if shutil.which("ffmpeg") is None:
        return None

    first_frame = os.path.join(annotated_rgb_dir, "frame_000000.jpg")
    if not os.path.isfile(first_frame):
        return None

    output_path = os.path.join(output_dir, "track_tracked.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        os.path.join(annotated_rgb_dir, "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path


def main(argv: list[str]) -> int:
    if not argv or any(arg in {"-h", "--help"} for arg in argv):
        print(usage())
        return 0

    base_config = build_base_config()
    override_config = OmegaConf.from_dotlist(argv)
    merged_config = OmegaConf.merge(base_config, override_config)
    config_dir = os.path.join(get_path(), "pipelines", "lexi-sg")
    resolved_config = load_object_segmentation_service_config(merged_config, config_dir=config_dir)
    config = OmegaConf.create(resolved_config)

    try:
        rgb_dir = os.path.abspath(str(require_value(config, "rgb_dir")))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    if not os.path.isdir(rgb_dir):
        print(f"rgb_dir does not exist or is not a directory: {rgb_dir}", file=sys.stderr)
        return 2

    scene_name = str(config.get("scene_name") or os.path.basename(rgb_dir) or "rgb")

    output_dir = resolve_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)

    dataset = get_dataset(
        rgb_dir,
        input_image_max_side=int(config.get("input_image_max_side", -1)),
    )
    rgb_paths = list(dataset.get_rgb())[:: int(config.get("subsample", 1))]
    plan = build_plan(len(rgb_paths), config.get("sampling", {}))
    ctx = {
        "rgb_paths": rgb_paths,
    }

    with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as handle:
        yaml.safe_dump(OmegaConf.to_container(config, resolve=True), handle, sort_keys=False)

    service = ObjectSegmentationService(
        config=config,
        scene_name=scene_name,
        artifact_root=output_dir,
        color_seed=int(get_object_segmentation_value(config, "color_seed", 0)),
    )
    result = service.run_detect_and_track_from_context(ctx=ctx, plan=plan)
    inferred_fps = infer_sampled_fps(rgb_paths, result.sampled_frame_ids)
    video_path = None
    if inferred_fps is not None:
        try:
            video_path = save_track_video(result.annotated_rgb_dir, output_dir, inferred_fps)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: failed to save track video: {exc.stderr}", file=sys.stderr)

    summary = {
        "rgb_dir": rgb_dir,
        "scene_name": scene_name,
        "output_dir": output_dir,
        "sampled_frame_ids": [int(fid) for fid in result.sampled_frame_ids],
        "tracklet_ids": [int(obj_id) for obj_id in result.tracklet_ids],
        "detections_dir": result.detections_dir,
        "annotated_rgb_dir": result.annotated_rgb_dir,
        "raw_rgb_dir": result.raw_rgb_dir,
        "track_video_fps": None if inferred_fps is None else float(inferred_fps),
        "track_video_path": video_path,
        "log_path": service.log_path,
    }
    with open(os.path.join(output_dir, "run_summary.yaml"), "w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    print(f"Saved detect+track artifacts to: {output_dir}")
    print(f"Sampled {len(result.sampled_frame_ids)} RGB frames.")
    print(f"Annotated RGB images: {result.annotated_rgb_dir}")
    print(f"Tracked detections: {result.detections_dir}")
    if video_path:
        print(f"Track video ({inferred_fps:.3f} FPS): {video_path}")
    elif inferred_fps is None:
        print("Track video was not saved: unable to infer FPS from sampled RGB timestamps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
