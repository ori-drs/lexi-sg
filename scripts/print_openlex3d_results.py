#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import yaml

METRICS = ["synonyms", "depictions", "vis_sim", "clutter", "missing", "incorrect"]
DATASETS = ("replica", "scannetpp", "hm3d")
CSV_FIELDS = ["scene", *METRICS, "status", "results_yaml"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a clean OpenLex3D summary from a per-scene log root."
    )
    parser.add_argument("log_dir", type=Path, help="Root log directory containing per-scene subdirectories")
    parser.add_argument("scenes", nargs="*", help="Optional explicit scene list")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Override automatic dataset detection",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional output CSV path for per-scene rows plus a __mean__ summary row",
    )
    return parser


def discover_scenes(log_dir: Path) -> list[str]:
    return sorted((p.name for p in log_dir.iterdir() if p.is_dir()), key=natural_key)


def natural_key(text: str) -> list[tuple[int, object]]:
    parts = re.split(r"(\d+)", text)
    key: list[tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        # Tag each chunk so Python never has to compare raw ints against strs.
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def candidate_paths(scene_log_dir: Path, dataset: str, scene: str) -> list[Path]:
    current_base = scene_log_dir / "openlex3d_eval" / scene
    legacy_base = scene_log_dir / "eval" / "openlex3d_eval"
    return [
        current_base / f"{dataset}_top_5" / scene / "results.yaml",
        current_base / "top_5" / dataset / scene / "results.yaml",
        legacy_base / "clip" / f"{dataset}_top_5" / scene / "results.yaml",
        legacy_base / "clip" / "top_5" / dataset / scene / "results.yaml",
    ]


def detect_from_results(log_dir: Path, scenes: Iterable[str]) -> str | None:
    counts = {dataset: 0 for dataset in DATASETS}
    for scene in scenes:
        scene_log_dir = log_dir / scene
        if not scene_log_dir.is_dir():
            continue
        for dataset in DATASETS:
            if any(path.is_file() for path in candidate_paths(scene_log_dir, dataset, scene)):
                counts[dataset] += 1

    best_dataset, best_count = max(counts.items(), key=lambda item: item[1])
    return best_dataset if best_count > 0 else None


def detect_from_scene_names(scenes: Iterable[str]) -> str | None:
    scene_list = list(scenes)
    if not scene_list:
        return None
    if all(re.fullmatch(r"(?:room|office)\d+", scene) for scene in scene_list):
        return "replica"
    if all(re.fullmatch(r"[0-9a-f]{10}", scene) for scene in scene_list):
        return "scannetpp"
    if all(re.fullmatch(r"\d+", scene) for scene in scene_list):
        return "hm3d"
    if all("-" in scene for scene in scene_list):
        return "hm3d"
    return None


def detect_dataset(log_dir: Path, scenes: list[str]) -> str:
    dataset = detect_from_results(log_dir, scenes)
    if dataset is not None:
        return dataset

    dataset = detect_from_scene_names(scenes)
    if dataset is not None:
        return dataset

    raise SystemExit(
        "Could not infer dataset from the log directory. Pass --dataset explicitly."
    )


def load_row(log_dir: Path, dataset: str, scene: str) -> dict[str, object]:
    scene_log_dir = log_dir / scene
    if not scene_log_dir.is_dir():
        return {"scene": scene, "status": "missing_log_dir", "path": "-"}

    path = next((p for p in candidate_paths(scene_log_dir, dataset, scene) if p.is_file()), None)
    if path is None:
        return {"scene": scene, "status": "missing", "path": "-"}

    try:
        data = yaml.safe_load(path.read_text()) or {}
        freq = data.get("freq") or {}
        row: dict[str, object] = {"scene": scene, "status": "ok", "path": str(path)}
        for metric in METRICS:
            row[metric] = freq.get(metric)
        return row
    except Exception:
        return {"scene": scene, "status": "parse_error", "path": str(path)}


def fmt_metric(row: dict[str, object], metric: str) -> str:
    value = row.get(metric)
    return "-" if value is None else f"{float(value):.6f}"


def fmt_mean(value: float | None, width: int) -> str:
    return "-".rjust(width) if value is None else f"{value:>{width}.6f}"


def mean_metric(rows: list[dict[str, object]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return None if not values else sum(values) / len(values)


def mean_metric_all(rows: list[dict[str, object]], metric: str) -> float | None:
    if not rows:
        return None
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return None if not values else sum(values) / len(rows)


def scene_column_width(dataset: str, scenes: list[str]) -> int:
    min_width = 24 if dataset == "hm3d" else 12
    return max(min_width, max((len(scene) for scene in scenes), default=min_width))


def make_mean_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    mean_row: dict[str, object] = {
        "scene": "__mean__",
        "status": "summary",
        "path": "-",
    }
    for metric in METRICS:
        mean_row[metric] = mean_metric_all(rows, metric)
    return mean_row


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mean_row = make_mean_row(rows)
    csv_rows = list(rows)
    if mean_row is not None:
        csv_rows.append(mean_row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(
                {
                    "scene": row.get("scene", ""),
                    "synonyms": row.get("synonyms", ""),
                    "vis_sim": row.get("vis_sim", ""),
                    "clutter": row.get("clutter", ""),
                    "incorrect": row.get("incorrect", ""),
                    "missing": row.get("missing", ""),
                    "depictions": row.get("depictions", ""),
                    "status": row.get("status", ""),
                    "results_yaml": row.get("path", ""),
                }
            )



def print_summary(log_dir: Path, dataset: str, scenes: list[str], rows: list[dict[str, object]]) -> None:
    width = scene_column_width(dataset, scenes)

    print(f"Output root: {log_dir}")
    print(f"Dataset: {dataset}")
    print()
    print(
        f"{'scene':<{width}} {'synonyms':>10} {'depictions':>12} {'vis_sim':>10} "
        f"{'clutter':>10} {'missing':>10} {'incorrect':>10} {'status':>14}"
    )
    for row in rows:
        print(
            f"{str(row['scene']):<{width}} "
            f"{fmt_metric(row, 'synonyms'):>10} "
            f"{fmt_metric(row, 'depictions'):>12} "
            f"{fmt_metric(row, 'vis_sim'):>10} "
            f"{fmt_metric(row, 'clutter'):>10} "
            f"{fmt_metric(row, 'missing'):>10} "
            f"{fmt_metric(row, 'incorrect'):>10} "
            f"{str(row['status']):>14}"
        )

    ok_rows = [row for row in rows if row["status"] == "ok"]
    print()
    print(f"Scenes: {len(rows)} total, {len(ok_rows)} with results")
    if rows:
        means = {metric: mean_metric_all(rows, metric) for metric in METRICS}
        print(
            f"{'Mean:':<{width}} "
            f"{fmt_mean(means['synonyms'], 10)} "
            f"{fmt_mean(means['depictions'], 12)} "
            f"{fmt_mean(means['vis_sim'], 10)} "
            f"{fmt_mean(means['clutter'], 10)} "
            f"{fmt_mean(means['missing'], 10)} "
            f"{fmt_mean(means['incorrect'], 10)}"
        )

    bad_rows = [row for row in rows if row["status"] != "ok"]
    if bad_rows:
        print()
        print("Non-ok scenes:")
        for row in bad_rows:
            print(f"  {row['scene']}: {row['status']}")


def main() -> int:
    args = build_parser().parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    scenes = args.scenes or discover_scenes(log_dir)
    if not scenes:
        raise SystemExit(f"No per-scene log directories found under {log_dir}")

    dataset = args.dataset or detect_dataset(log_dir, scenes)
    rows = [load_row(log_dir, dataset, scene) for scene in scenes]
    print_summary(log_dir, dataset, scenes, rows)

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        write_csv(csv_path, rows)
        print()
        print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
