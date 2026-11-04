#!/usr/bin/env python3
"""Absolute trajectory error of a pipeline run against per-frame ground-truth poses, via evo.

The pipeline writes output/pose.txt as `<frame-stem> x y z qx qy qz qw`. evo's TUM
reader needs numeric timestamps, so both the estimate and the ground truth are
rewritten with the frame index (trailing number of the stem) as the timestamp,
restricted to the frames the pipeline actually produced.

Ground truth: <data_path>/pose/ (HM3D) or <data_path>/closed_loop_poses_tum.txt (AOD).

Usage:
  python scripts/eval_ate.py --output_dir output --data_path /workspace/datasets/hm3d/00824 \\
      --pose_convention opengl

Reports translation APE with SE(3) alignment and with Sim(3) alignment. Use
the Sim(3) number for image-only runs, whose scale is only up to a factor;
SE(3) is the fair comparison once metric priors (depth/poses) are supplied.
"""

import argparse
import json
import os
import re
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation as R

_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def frame_index(stem):
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"no trailing frame number in {stem!r}")
    return int(m.group(1))


def load_estimate(pose_txt):
    rows = {}
    with open(pose_txt) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 8:
                continue
            rows[parts[0]] = [float(v) for v in parts[1:]]
    return rows


GT_TUM_NAME = "closed_loop_poses_tum.txt"


def find_gt(data_path):
    for candidate in (os.path.join(data_path, "pose"), os.path.join(data_path, GT_TUM_NAME)):
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(f"no ground truth under {data_path}: expected pose/ or {GT_TUM_NAME}")


def load_gt(gt_path, stems, convention):
    if os.path.isdir(gt_path):
        return load_gt_matrices(gt_path, stems, convention)
    rows = load_estimate(gt_path)
    return {s: rows[s] for s in stems if s in rows}


def load_gt_matrices(pose_dir, stems, convention):
    rows = {}
    for stem in stems:
        path = os.path.join(pose_dir, stem + ".txt")
        if not os.path.exists(path):
            continue
        T = np.loadtxt(path).reshape(4, 4)
        if convention == "opengl":
            T = T @ _OPENGL_TO_OPENCV
        q = R.from_matrix(T[:3, :3]).as_quat()  # x y z w
        rows[stem] = [*T[:3, 3], *q]
    return rows


def write_tum(rows, path):
    with open(path, "w") as f:
        for stem in sorted(rows, key=frame_index):
            f.write(f"{frame_index(stem)} " + " ".join(f"{v:.8f}" for v in rows[stem]) + "\n")


def run_evo(gt_file, est_file, correct_scale):
    """Translation APE via evo's API: SE(3) alignment, or Sim(3) if correct_scale."""
    from evo.core import metrics, sync
    from evo.tools import file_interface

    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.01)
    traj_est.align(traj_ref, correct_scale=correct_scale)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    return {k: float(v) for k, v in ape.get_all_statistics().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True, help="pipeline output folder containing pose.txt")
    ap.add_argument("--data_path", required=True, help=f"dataset root containing pose/ or {GT_TUM_NAME}")
    ap.add_argument("--gt", default=None, help="ground truth (pose/ directory or TUM file); overrides the search under --data_path")
    ap.add_argument("--pose_convention", default="opencv", choices=["opencv", "opengl"])
    ap.add_argument("--save", default=None, help="write results JSON here (default: <output_dir>/ate.json)")
    args = ap.parse_args()

    est = load_estimate(os.path.join(args.output_dir, "pose.txt"))
    gt = load_gt(args.gt or find_gt(args.data_path), est.keys(), args.pose_convention)
    common = sorted(set(est) & set(gt), key=frame_index)
    if not common:
        raise SystemExit("no frames in common between estimate and ground truth")

    with tempfile.TemporaryDirectory() as tmp:
        est_file, gt_file = os.path.join(tmp, "est.tum"), os.path.join(tmp, "gt.tum")
        write_tum({s: est[s] for s in common}, est_file)
        write_tum({s: gt[s] for s in common}, gt_file)
        results = {
            "frames": len(common),
            "ate_trans_se3": run_evo(gt_file, est_file, correct_scale=False),
            "ate_trans_sim3": run_evo(gt_file, est_file, correct_scale=True),
        }

    print(f"frames evaluated: {results['frames']}")
    for key in ("ate_trans_se3", "ate_trans_sim3"):
        s = results[key]
        print(f"{key:15s} rmse={s.get('rmse', float('nan')):.4f} m  mean={s.get('mean', float('nan')):.4f}  "
              f"median={s.get('median', float('nan')):.4f}  max={s.get('max', float('nan')):.4f}")
    save = args.save or os.path.join(args.output_dir, "ate.json")
    with open(save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {save}")


if __name__ == "__main__":
    main()
