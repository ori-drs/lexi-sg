#!/usr/bin/env python3
#
# Runs the object segmentation pipeline sequentially for all scenes in an explicit dataset.
#
# Usage:
#   python scripts/run_object_segmentation_all.py <dataset>
#   python scripts/run_object_segmentation_all.py <dataset> /abs/path/to/output_root
#   python scripts/run_object_segmentation_all.py <dataset> /abs/path/to/output_root <scene1> <scene2>
#   python scripts/run_object_segmentation_all.py --gpus 0,1 <dataset>
#   python scripts/run_object_segmentation_all.py --gpus 0,1 <dataset> /abs/path/to/output_root -- stage=all num_uniform_frames=240
#   python scripts/run_object_segmentation_all.py <dataset> /abs/path/to/output_root <scene1> -- tasks.mask_selection.overlap_thres=0.85
#
# Datasets:
#   hm3d
#   replica
#   scannetpp
#
# Optional env vars:
#   PIPELINE_MODULE=lexisg.tasks.object_segmentation.pipeline
#   EXTRA_OVERRIDES="num_uniform_frames=240 stage=all"
#   HM3D_DATA_PATH=/workspace/datasets/hm3d
#   REPLICA_DATA_PATH=/workspace/datasets/Replica
#   REPLICA_EVAL_GT_PATH=/path/to/Replica
#   SCANNETPP_DATA_PATH=/workspace/datasets/scannetpp
#
# Hydra-style overrides:
#   Any args after `--` are passed through directly as Hydra overrides.
# GPU scheduling:
#   Use `--gpus 0,1,...` to run scenes in parallel, one worker per GPU ID.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import queue
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_MODULE_DEFAULT = "lexisg.tasks.object_segmentation.pipeline"
EXTRA_OVERRIDES_ENV = "EXTRA_OVERRIDES"

DATASET_CONFIG = {
    "hm3d": {
        "data_env": "HM3D_DATA_PATH",
        "data_path_default": "/workspace/datasets/hm3d",
        "output_prefix": "exp_object_segmentation_hm3d",
        "data_label": "HM3D data path",
        "discover": "discover_hm3d_scenes",
        "missing_message": [
            "No HM3D scenes found.",
            "Expected scene directories containing rgb/depth and pose or poses.",
        ],
        "example_scenes": "00824-Dd4bFSTQ8gi 00829-QaLdnwvtxbs",
    },
    "replica": {
        "data_env": "REPLICA_DATA_PATH",
        "data_path_default": "/workspace/datasets/Replica",
        "output_prefix": "exp_object_segmentation_replica",
        "data_label": "Replica data path",
        "discover": "discover_replica_scenes",
        "missing_message": ["No Replica scenes found."],
        "example_scenes": "room0 office0",
    },
    "scannetpp": {
        "data_env": "SCANNETPP_DATA_PATH",
        "data_path_default": "/workspace/datasets/scannetpp",
        "output_prefix": "exp_object_segmentation_scannetpp",
        "data_label": "ScanNet++ data path",
        "discover": "discover_scannetpp_scenes",
        "missing_message": [
            "No ScanNet++ scenes found.",
            "Expected scene directories with an iphone subdirectory.",
        ],
        "example_scenes": "0a7cc12c0e fd361ab85f",
    },
}


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams
        self._lock = Lock()

    def write(self, text: str) -> None:
        with self._lock:
            for stream in self.streams:
                stream.write(text)
                stream.flush()

    def flush(self) -> None:
        with self._lock:
            for stream in self.streams:
                stream.flush()


@dataclass
class ParsedArgs:
    dataset: str
    output_root: Path
    scenes: list[str]
    cli_overrides: list[str]
    gpu_ids: list[str]


def usage() -> str:
    script = Path(__file__).name
    datasets = ", ".join(DATASET_CONFIG)
    return "\n".join(
        [
            f"Usage: python scripts/{script} <dataset> [output_root] [scene1 scene2 ...] [-- override=value ...]",
            "",
            f"Datasets: {datasets}",
            "",
            "Examples:",
            f"  python scripts/{script} hm3d",
            f"  python scripts/{script} replica /abs/path/to/output_root room0 office0",
            f"  python scripts/{script} --gpus 0,1 replica /abs/path/to/output_root room0 office0",
            f"  python scripts/{script} scannetpp /abs/path/to/output_root -- stage=all num_uniform_frames=240",
        ]
    )


def parse_args(argv: list[str]) -> ParsedArgs:
    if not argv or argv[0] in {"-h", "--help"}:
        print(usage())
        raise SystemExit(0)

    tail = argv[:]
    gpu_ids: list[str] = []
    positional: list[str] = []
    i = 0
    while i < len(tail):
        arg = tail[i]
        if arg == "--":
            break
        if arg == "--gpus":
            if i + 1 >= len(tail):
                print("Missing value for --gpus", file=sys.stderr)
                print(usage(), file=sys.stderr)
                raise SystemExit(2)
            gpu_ids = parse_gpu_ids(tail[i + 1])
            i += 2
            continue
        positional.append(arg)
        i += 1

    cli_overrides = tail[i + 1 :] if i < len(tail) and tail[i] == "--" else []
    if not positional:
        print(usage())
        raise SystemExit(0)

    dataset = positional[0].strip().lower()
    if dataset not in DATASET_CONFIG:
        print(f"Unsupported dataset: {dataset!r}", file=sys.stderr)
        print(usage(), file=sys.stderr)
        raise SystemExit(2)

    default_output_root = REPO_ROOT / "logs" / f"{DATASET_CONFIG[dataset]['output_prefix']}_{datetime.now():%Y%m%d_%H%M%S}"
    dataset_positionals = positional[1:]
    if dataset_positionals:
        output_root = Path(dataset_positionals[0]).expanduser().resolve(strict=False)
        scenes = dataset_positionals[1:]
    else:
        output_root = default_output_root.resolve(strict=False)
        scenes = []

    return ParsedArgs(
        dataset=dataset,
        output_root=output_root,
        scenes=scenes,
        cli_overrides=cli_overrides,
        gpu_ids=gpu_ids,
    )


def parse_gpu_ids(raw: str) -> list[str]:
    gpu_ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not gpu_ids:
        print("Expected at least one GPU ID in --gpus", file=sys.stderr)
        raise SystemExit(2)
    return gpu_ids


def available_cuda_device_count() -> int | None:
    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return None


def validate_gpu_ids(gpu_ids: list[str]) -> None:
    if not gpu_ids:
        return

    invalid_format = [gpu_id for gpu_id in gpu_ids if not gpu_id.isdigit()]
    if invalid_format:
        print(
            "Invalid GPU IDs in --gpus; expected comma-separated non-negative integers: "
            + ", ".join(invalid_format),
            file=sys.stderr,
        )
        raise SystemExit(2)

    duplicates = sorted({gpu_id for gpu_id in gpu_ids if gpu_ids.count(gpu_id) > 1}, key=int)
    if duplicates:
        print(
            "Duplicate GPU IDs in --gpus are not allowed: " + ", ".join(duplicates),
            file=sys.stderr,
        )
        raise SystemExit(2)

    device_count = available_cuda_device_count()
    if device_count == 0:
        print("CUDA is not available, but --gpus was provided.", file=sys.stderr)
        raise SystemExit(2)
    if device_count is None:
        print("Unable to validate --gpus against available CUDA devices.", file=sys.stderr)
        raise SystemExit(2)

    invalid_ids = [gpu_id for gpu_id in gpu_ids if int(gpu_id) >= device_count]
    if invalid_ids:
        print(
            f"Invalid GPU IDs in --gpus: {', '.join(invalid_ids)}. "
            f"Available CUDA device indices: 0..{device_count - 1}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def discover_hm3d_scenes(root: Path) -> list[str]:
    if not root.is_dir():
        return []

    scenes: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if any(part in {"rgb", "depth", "pose", "poses"} for part in path.parts):
            continue
        if not (path / "rgb").is_dir():
            continue
        if not (path / "depth").is_dir():
            continue
        if not ((path / "pose").is_dir() or (path / "poses").is_dir()):
            continue
        scenes.add(path.name)
    return sorted(scenes, key=natural_sort_key)


def discover_replica_scenes(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        [path.name for path in root.iterdir() if path.is_dir() and re.fullmatch(r"(room|office)\d+", path.name)],
        key=natural_sort_key,
    )


def discover_scannetpp_scenes(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir() and (path / "iphone").is_dir()], key=natural_sort_key)


def natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def stream_process(cmd: list[str], tee: Tee, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        chunks.append(line)
        tee.write(line)
    rc = proc.wait()
    return rc, "".join(chunks)


def run_python_module(module_name: str, module_args: Iterable[str], tee, env: dict[str, str]) -> int:
    cmd = [sys.executable, "-u", "-m", module_name, *module_args]
    rc, _ = stream_process(cmd, tee, env)
    return rc


class PrefixedTee:
    def __init__(self, tee: Tee, prefix: str):
        self.tee = tee
        self.prefix = prefix

    def write(self, text: str) -> None:
        if not text:
            return
        lines = text.splitlines(keepends=True)
        for line in lines:
            if line.endswith("\n"):
                self.tee.write(f"{self.prefix}{line}")
            else:
                self.tee.write(f"{self.prefix}{line}\n")

    def flush(self) -> None:
        self.tee.flush()


def dataset_data_path(dataset: str) -> str:
    cfg = DATASET_CONFIG[dataset]
    return os.environ.get(cfg["data_env"], cfg["data_path_default"])


def env_overrides() -> list[str]:
    raw = os.environ.get(EXTRA_OVERRIDES_ENV, "").strip()
    return shlex.split(raw) if raw else []


def run() -> int:
    parsed = parse_args(sys.argv[1:])
    validate_gpu_ids(parsed.gpu_ids)
    dataset = parsed.dataset
    cfg = DATASET_CONFIG[dataset]
    data_path = dataset_data_path(dataset)
    pipeline_module = os.environ.get("PIPELINE_MODULE", PIPELINE_MODULE_DEFAULT)
    extra_overrides = env_overrides()

    parsed.output_root.mkdir(parents=True, exist_ok=True)
    run_log = parsed.output_root / "run.log"
    with run_log.open("a", encoding="utf-8") as log_fp:
        tee = Tee(sys.stdout, log_fp)
        env = dict(os.environ)
        env["PYTHONPATH"] = "."

        discover_fn = globals()[cfg["discover"]]
        scenes = parsed.scenes or discover_fn(Path(data_path))
        if not scenes:
            for line in cfg["missing_message"]:
                tee.write(f"{line}\n")
            tee.write(f"Checked: {data_path}\n")
            tee.write("Provide scenes explicitly, e.g.:\n")
            tee.write(
                f"  python scripts/{Path(__file__).name} {dataset} \"{parsed.output_root}\" {cfg['example_scenes']}\n"
            )
            return 1

        tee.write(f"Output root: {parsed.output_root}\n")
        tee.write(f"Run log: {run_log}\n")
        tee.write(f"{cfg['data_label']}: {data_path}\n")
        if dataset == "replica":
            replica_eval_gt_path = os.environ.get("REPLICA_EVAL_GT_PATH", "")
            if replica_eval_gt_path:
                tee.write(f"Replica eval GT path override: {replica_eval_gt_path}\n")
            else:
                tee.write("Replica eval GT path override: <none; use OpenLex3D config default>\n")
        tee.write(f"Python: {sys.executable}\n")
        tee.write(f"Pipeline module: {pipeline_module}\n")
        tee.write(f"Dataset: {dataset}\n")
        tee.write(f"Scenes ({len(scenes)}): {' '.join(scenes)}\n")
        if parsed.gpu_ids:
            tee.write(f"GPU workers ({len(parsed.gpu_ids)}): {' '.join(parsed.gpu_ids)}\n")
        else:
            tee.write("GPU workers: <sequential default>\n")
        extra_overrides_raw = os.environ.get(EXTRA_OVERRIDES_ENV, "").strip()
        if extra_overrides_raw:
            tee.write(f"Extra overrides from env: {extra_overrides_raw}\n")
        if parsed.cli_overrides:
            tee.write(f"Extra overrides from CLI: {' '.join(parsed.cli_overrides)}\n")
        tee.write("\n")

        def run_scene(scene: str, gpu_id: str | None) -> tuple[str, int]:
            if dataset == "scannetpp" and not (Path(data_path) / scene / "iphone").is_dir():
                tee.write(f"[SKIP] scene={scene} missing {Path(data_path) / scene / 'iphone'}\n")
                return scene, 0

            scene_log_dir = parsed.output_root / scene
            scene_log_dir.mkdir(parents=True, exist_ok=True)
            gpu_suffix = f" gpu={gpu_id}" if gpu_id is not None else ""
            tee.write(f"[RUN] scene={scene} log_dir={scene_log_dir}{gpu_suffix}\n")

            module_args = [
                f"dataset={dataset}",
                f"scene_name={scene}",
                f"data_path={data_path}",
                f"log_dir={scene_log_dir}",
            ]
            if dataset == "replica":
                replica_eval_gt_path = os.environ.get("REPLICA_EVAL_GT_PATH", "")
                if replica_eval_gt_path:
                    module_args.append(f"tasks.eval.replica_path={replica_eval_gt_path}")
            module_args.extend(extra_overrides)
            module_args.extend(parsed.cli_overrides)

            scene_env = dict(env)
            if gpu_id is not None:
                scene_env["CUDA_VISIBLE_DEVICES"] = gpu_id
            scene_tee = PrefixedTee(tee, f"[{scene}] ") if parsed.gpu_ids else tee
            rc = run_python_module(pipeline_module, module_args, scene_tee, scene_env)
            if rc != 0:
                tee.write(f"[FAIL] scene={scene} rc={rc}\n")
                return scene, rc

            tee.write(f"[DONE] scene={scene}\n\n")
            return scene, 0

        results: list[tuple[str, int]] = []
        if parsed.gpu_ids:
            pending: queue.Queue[str] = queue.Queue()
            for scene in scenes:
                pending.put(scene)

            def gpu_worker(gpu_id: str) -> None:
                while True:
                    try:
                        scene = pending.get_nowait()
                    except queue.Empty:
                        return
                    results.append(run_scene(scene, gpu_id))

            with ThreadPoolExecutor(max_workers=len(parsed.gpu_ids)) as executor:
                for future in [executor.submit(gpu_worker, gpu_id) for gpu_id in parsed.gpu_ids]:
                    future.result()
        else:
            for scene in scenes:
                results.append(run_scene(scene, gpu_id=None))

        failed = sorted(scene for scene, rc in results if rc != 0)
        if failed:
            tee.write(f"Completed with {len(failed)} failed scene(s): {' '.join(failed)}\n")
            return 1
        tee.write("Completed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
