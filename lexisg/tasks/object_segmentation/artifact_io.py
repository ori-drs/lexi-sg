from __future__ import annotations

import gzip
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import open3d as o3d


def save_pickle_gz(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_gz(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def pcd_to_payload(pcd: o3d.geometry.PointCloud):
    return {
        "points": np.asarray(pcd.points),
        "colors": np.asarray(pcd.colors),
    }


def pcd_from_payload(payload) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(payload.get("points", np.zeros((0, 3), dtype=np.float64)))
    colors = np.asarray(payload.get("colors", np.zeros((0, 3), dtype=np.float64)))
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors.size > 0 and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def voxelize_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    voxels = np.floor(points / float(voxel_size)).astype(np.int32)
    return np.unique(voxels, axis=0)


def voxel_rows_view(voxel_indices: np.ndarray) -> np.ndarray:
    voxel_indices = np.ascontiguousarray(voxel_indices, dtype=np.int32)
    if voxel_indices.ndim != 2 or voxel_indices.shape[1] != 3:
        return np.zeros((0,), dtype=np.dtype((np.void, np.dtype(np.int32).itemsize * 3)))
    row_dtype = np.dtype((np.void, voxel_indices.dtype.itemsize * voxel_indices.shape[1]))
    return voxel_indices.view(row_dtype).reshape(-1)


def load_object_pcd_artifact(path: str) -> o3d.geometry.PointCloud:
    payload = np.load(path)
    return pcd_from_payload(payload)


def load_object_voxel_artifact(path: str) -> Tuple[np.ndarray, float]:
    payload = np.load(path)
    voxel_indices = np.asarray(payload.get("voxel_indices", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32)
    voxel_size_arr = np.asarray(payload.get("voxel_size", np.asarray([0.0], dtype=np.float32))).reshape(-1)
    voxel_size = float(voxel_size_arr[0]) if voxel_size_arr.size > 0 else 0.0
    return voxel_indices, voxel_size


def save_track_result(
    stage_dir: str,
    result: Dict,
    voxel_size: float | None = None,
):
    del voxel_size
    payload = {
        "sampled_frame_ids": [int(fid) for fid in result.get("sampled_frame_ids", [])],
        "detections_dir": str(result.get("detections_dir", "")),
        "tracklet_ids": [int(obj_id) for obj_id in result.get("tracklet_ids", [])],
    }
    path = os.path.join(stage_dir, "track_result.pkl.gz")
    save_pickle_gz(path, payload)
    return path


def load_track_result(path: str):
    payload = load_pickle_gz(path)
    return {
        "sampled_frame_ids": [int(fid) for fid in payload.get("sampled_frame_ids", [])],
        "detections_dir": str(payload.get("detections_dir", "")),
        "tracklet_ids": [int(obj_id) for obj_id in payload.get("tracklet_ids", [])],
    }


def save_merge_result(stage_dir: str, result: Dict):
    payload = {
        "masks": {
            int(fid): {int(obj_id): np.asarray(mask).astype(bool) for obj_id, mask in per_frame.items()}
            for fid, per_frame in result.get("masks", {}).items()
        },
        "obj_pcds_world": {
            int(obj_id): pcd_to_payload(pcd)
            for obj_id, pcd in result.get("obj_pcds_world", {}).items()
        },
        "merge_map": {
            int(src): int(dst) for src, dst in result.get("merge_map", {}).items()
        },
    }
    path = os.path.join(stage_dir, "merge_result.pkl.gz")
    save_pickle_gz(path, payload)
    return path


def load_merge_result(path: str):
    payload = load_pickle_gz(path)
    return {
        "masks": {
            int(fid): {int(obj_id): np.asarray(mask).astype(bool) for obj_id, mask in per_frame.items()}
            for fid, per_frame in payload.get("masks", {}).items()
        },
        "obj_pcds_world": {
            int(obj_id): pcd_from_payload(pcd_payload)
            for obj_id, pcd_payload in payload.get("obj_pcds_world", {}).items()
        },
        "merge_map": {
            int(src): int(dst) for src, dst in payload.get("merge_map", {}).items()
        },
    }


def save_clip_result(stage_dir: str, result: Dict):
    payload = {
        "masks": {
            int(fid): {int(obj_id): np.asarray(mask).astype(bool) for obj_id, mask in per_frame.items()}
            for fid, per_frame in result.get("masks", {}).items()
        },
        "best_fid_per_obj": {
            int(obj_id): int(fid) for obj_id, fid in result.get("best_fid_per_obj", {}).items()
        },
        "best_fids_per_obj": {
            int(obj_id): [int(fid) for fid in fids]
            for obj_id, fids in result.get("best_fids_per_obj", {}).items()
        },
        "obj_clip_feats": {
            int(obj_id): np.asarray(feat, dtype=np.float32)
            for obj_id, feat in result.get("obj_clip_feats", {}).items()
        },
        "obj_clip_crops": {
            int(obj_id): [(int(fid), int(rank), np.asarray(crop)) for fid, rank, crop in crops]
            for obj_id, crops in result.get("obj_clip_crops", {}).items()
        },
        "obj_pcds_world": {
            int(obj_id): pcd_to_payload(pcd)
            for obj_id, pcd in result.get("obj_pcds_world", {}).items()
        },
        "merge_map": {
            int(src): int(dst) for src, dst in result.get("merge_map", {}).items()
        },
    }
    path = os.path.join(stage_dir, "clip_result.pkl.gz")
    save_pickle_gz(path, payload)
    return path


def load_clip_result(path: str):
    payload = load_pickle_gz(path)
    return {
        "masks": {
            int(fid): {int(obj_id): np.asarray(mask).astype(bool) for obj_id, mask in per_frame.items()}
            for fid, per_frame in payload.get("masks", {}).items()
        },
        "best_fid_per_obj": {
            int(obj_id): int(fid) for obj_id, fid in payload.get("best_fid_per_obj", {}).items()
        },
        "best_fids_per_obj": {
            int(obj_id): [int(fid) for fid in fids]
            for obj_id, fids in payload.get("best_fids_per_obj", {}).items()
        },
        "obj_clip_feats": {
            int(obj_id): np.asarray(feat, dtype=np.float32)
            for obj_id, feat in payload.get("obj_clip_feats", {}).items()
        },
        "obj_clip_crops": {
            int(obj_id): [(int(fid), int(rank), np.asarray(crop)) for fid, rank, crop in crops]
            for obj_id, crops in payload.get("obj_clip_crops", {}).items()
        },
        "obj_pcds_world": {
            int(obj_id): pcd_from_payload(pcd_payload)
            for obj_id, pcd_payload in payload.get("obj_pcds_world", {}).items()
        },
        "merge_map": {
            int(src): int(dst) for src, dst in payload.get("merge_map", {}).items()
        },
    }
