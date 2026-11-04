#!/usr/bin/env python3
"""Room segmentation evaluation on HM3DSem-Walks, replicating HOV-SG's `evaluate_rooms`.

Each predicted room cloud is compared top-down (BEV) against each ground-truth
region polygon from scene_info.json, at the GT floor height, downsampled to a
5 cm voxel grid, exactly as in HOV-SG (hovsg/eval/hm3dsem_evaluator.py):

  - association  = max(frac of pred within 5 cm of gt, frac of gt within 5 cm of pred)
  - hydra_prec   = mean over pred rooms of max_j (#gt points near pred_i) / |pred_i|
  - hydra_recall = mean over gt rooms  of max_i (#pred points near gt_j) / |gt_j|
  - AP           = trapz over precision/recall of Hungarian-matched rooms at IoU
                   thresholds 0.0..1.0 (HOV-SG sorts recall but not precision here;
                   kept as-is for comparability), and acc/prec/recall at index 6 of
                   that sweep, which HOV-SG labels "@IoU=0.5" (it is threshold 0.6).

Predicted clouds come from <output_dir>/original/*.pcd (world frame). That frame
is arbitrary (and, for monocular runs, only known up to scale), so the clouds are
first aligned to the ground-truth frame with the Sim(3) fitted between the run's
pose.txt and the GT poses -- the same alignment eval_ate.py uses.

Usage:
  python scripts/eval_rooms.py --output_dir output --data_path /workspace/datasets/hm3d/00824 \\
      --pose_convention opengl [--exclude_label doorway]
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile

import faiss
import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ate import frame_index, load_estimate, load_gt, write_tum  # noqa: E402


# --- HOV-SG helpers, copied verbatim in behaviour -------------------------------

def find_intersection_share(map_points, obj_points, radius=0.05):
    if map_points.shape[0] == 0 or obj_points.shape[0] == 0:
        return 0
    _, indices = cKDTree(obj_points).query(map_points, k=1, distance_upper_bound=radius, p=2, workers=-1)
    indices = indices[indices != obj_points.shape[0]]
    return indices.shape[0] / obj_points.shape[0]


def find_overlapping_ratio_faiss(pcd1, pcd2, radius=0.02):
    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return 0
    index1, index2 = faiss.IndexFlatL2(3), faiss.IndexFlatL2(3)
    index1.add(pcd1.astype(np.float32))
    index2.add(pcd2.astype(np.float32))
    D1, _ = index2.search(pcd1.astype(np.float32), k=1)
    D2, _ = index1.search(pcd2.astype(np.float32), k=1)
    return np.max([np.sum(D1 < radius**2) / pcd1.shape[0], np.sum(D2 < radius**2) / pcd2.shape[0]])


# --- alignment to the GT frame --------------------------------------------------

def sim3_to_gt(output_dir, data_path, convention):
    """(scale, R, t) mapping the run's world frame onto the GT world frame."""
    from evo.core import sync
    from evo.tools import file_interface

    est = load_estimate(os.path.join(output_dir, "pose.txt"))
    gt = load_gt(os.path.join(data_path, "pose"), est.keys(), convention)
    common = sorted(set(est) & set(gt), key=frame_index)
    with tempfile.TemporaryDirectory() as tmp:
        ef, gf = os.path.join(tmp, "est.tum"), os.path.join(tmp, "gt.tum")
        write_tum({s: est[s] for s in common}, ef)
        write_tum({s: gt[s] for s in common}, gf)
        traj_ref = file_interface.read_tum_trajectory_file(gf)
        traj_est = file_interface.read_tum_trajectory_file(ef)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.01)
    R, t, s = traj_est.align(traj_ref, correct_scale=True)
    return float(s), np.asarray(R), np.asarray(t)


# --- evaluation ------------------------------------------------------------------

def evaluate(pred_clouds, gt_rooms, gt_floors, voxel=0.05, radius=0.05):
    n_pred, n_gt = len(pred_clouds), len(gt_rooms)
    assoc = np.zeros((n_pred, n_gt))
    over_pred = np.zeros((n_pred, n_gt))
    over_gt = np.zeros((n_pred, n_gt))

    for j, gt in enumerate(gt_rooms):
        on_some_floor = any(fl["lower"] < gt["mean_height"] < fl["upper"] for fl in gt_floors) or not gt_floors
        if not on_some_floor:
            continue
        gt_bev = o3d.geometry.PointCloud()
        gt_bev.points = o3d.utility.Vector3dVector(np.asarray(gt["bev_region_points"], dtype=np.float64))
        gt_bev = gt_bev.voxel_down_sample(voxel)
        gt_pts = np.asarray(gt_bev.points)

        for i, pts in enumerate(pred_clouds):
            pred_mean_height = float(pts[:, 1].min() + (pts[:, 1].max() - pts[:, 1].min()) / 2)
            if not (gt["min_height"] < pred_mean_height < gt["max_height"]):
                continue
            flat = pts.copy()
            flat[:, 1] = gt["min_height"]  # project onto the GT floor plane
            pred_bev = o3d.geometry.PointCloud()
            pred_bev.points = o3d.utility.Vector3dVector(flat)
            pred_pts = np.asarray(pred_bev.voxel_down_sample(voxel).points)

            assoc[i, j] = find_overlapping_ratio_faiss(pred_pts, gt_pts, radius)
            over_pred[i, j] = min(find_intersection_share(gt_pts, pred_pts, radius), 1.0)
            over_gt[i, j] = min(find_intersection_share(pred_pts, gt_pts, radius), 1.0)

    hydra_prec = float(np.mean(over_pred.max(axis=1))) if n_pred else 0.0
    hydra_recall = float(np.mean(over_gt.max(axis=0))) if n_gt else 0.0

    row, col = linear_sum_assignment(assoc, maximize=True)
    acc_v, prec_v, rec_v = [], [], []
    for thresh in np.linspace(0.0, 1.0, 11, endpoint=True):
        TP = int(np.sum(assoc[row, col] > thresh))
        FP, FN, TN = n_pred - TP, n_gt - TP, 0
        prec_v.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        rec_v.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        acc_v.append((TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0)
    rec_sorted = sorted(rec_v)  # HOV-SG sorts recall only; kept for comparability
    ap = float(np.trapz(prec_v, rec_sorted))

    return {
        "acc@IoU=0.5": acc_v[6], "ap": ap, "prec@IoU=0.5": prec_v[6], "recall@IoU=0.5": rec_v[6],
        "gt": n_gt, "pred": n_pred, "hydra_prec": hydra_prec, "hydra_recall": hydra_recall,
        "matched_iou": [float(assoc[r, c]) for r, c in zip(row, col)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_path", required=True, help="scene root with scene_info.json and pose/")
    ap.add_argument("--pose_convention", default="opencv", choices=["opencv", "opengl"])
    ap.add_argument("--exclude_label", action="append", default=[],
                    help="skip predicted nodes with this label prefix (e.g. doorway); repeatable")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    info = json.load(open(os.path.join(args.data_path, "scene_info.json")))
    gt_rooms = list(info["regions"].values()) if isinstance(info["regions"], dict) else info["regions"]
    levels = info.get("levels", [])
    gt_floors = [{"lower": l["lower"], "upper": l["upper"]} for l in
                 (levels.values() if isinstance(levels, dict) else levels) if "lower" in l]

    s, R, t = sim3_to_gt(args.output_dir, args.data_path, args.pose_convention)
    print(f"aligned to GT frame with Sim(3): scale={s:.4f}")

    files = sorted(glob.glob(os.path.join(args.output_dir, "original", "*.pcd")),
                   key=lambda f: int(re.search(r"_(\d+)\.pcd$", f).group(1)))
    pred, names = [], []
    for f in files:
        label = os.path.basename(f).rsplit("_", 1)[0]
        if any(label.startswith(x) for x in args.exclude_label):
            continue
        pts = np.asarray(o3d.io.read_point_cloud(f).points)
        pred.append((s * (R @ pts.T)).T + t)
        names.append(os.path.basename(f)[:-4])

    m = evaluate(pred, gt_rooms, gt_floors)
    print("----- Room Evaluation (HOV-SG protocol) -----")
    for k in ("gt", "pred", "hydra_prec", "hydra_recall", "ap", "prec@IoU=0.5", "recall@IoU=0.5", "acc@IoU=0.5"):
        v = m[k]
        print(f"{k:16s} {v:.4f}" if isinstance(v, float) else f"{k:16s} {v}")
    m["pred_names"] = names
    save = args.save or os.path.join(args.output_dir, "rooms.json")
    json.dump(m, open(save, "w"), indent=2)
    print(f"saved {save}")


if __name__ == "__main__":
    main()
