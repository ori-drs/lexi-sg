from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import open3d as o3d
from PIL import Image
from loguru import logger

from lexisg.utils.config import get_object_segmentation_geometry_config, get_object_segmentation_stage_config
import lexisg.utils.geometry as geometry


def estimate_numpy_nbytes(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        total = 0
        for item in value:
            if isinstance(item, np.ndarray):
                total += int(item.nbytes)
        return total
    return 0


def load_images_for_frame_ids(rgb_paths: List[str], frame_ids) -> np.ndarray:
    images = [np.array(Image.open(rgb_paths[int(fid)]).convert("RGB")) for fid in np.asarray(frame_ids, dtype=np.int32)]
    return np.stack(images, axis=0)


def save_shared_geometry_debug_pcd(shared_geometry: Dict, save_path: str):
    points = np.asarray(shared_geometry.get("points", np.zeros((0, 3), dtype=np.float32)))
    pcd = o3d.geometry.PointCloud()
    if points.size > 0:
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)


def attach_shared_geometry_to_predictions(
    predictions: Dict,
    shared_geometry_voxel_size: float,
    max_backproj_depth: float | None,
    run_count: int,
    artifact_root: str | None = None,
):
    shared_geometry = geometry.build_shared_pointcloud_lookup_from_predictions(
        predictions,
        depth_scale=1.0,
        max_depth=max_backproj_depth,
        include_colors=False,
        voxel_size=shared_geometry_voxel_size,
    )
    predictions["shared_geometry"] = shared_geometry
    points = np.asarray(shared_geometry.get("points", np.zeros((0, 3), dtype=np.float32)))
    pixel_to_index = np.asarray(shared_geometry.get("pixel_to_index", np.zeros((0, 0, 0), dtype=np.int32)))
    frame_raw_point_counts = np.asarray(shared_geometry.get("frame_raw_point_counts", np.zeros((0,), dtype=np.int32)))
    frame_count = int(shared_geometry.get("frame_count", pixel_to_index.shape[0]))
    raw_point_count = int(shared_geometry.get("raw_point_count", int(shared_geometry.get("valid_pixels", 0))))
    logger.info(
        f"Shared geometry prepared for run_count={run_count}: "
        f"raw_points={raw_point_count} voxel_points={int(shared_geometry.get('voxel_count', points.shape[0]))} "
        f"frames={frame_count}"
    )
    logger.info(
        f"Shared geometry memory for run_count={run_count}: "
        f"points={points.nbytes / (1024 ** 2):.1f} MiB "
        f"lookup={pixel_to_index.nbytes / (1024 ** 2):.1f} MiB "
        f"metadata={frame_raw_point_counts.nbytes / (1024 ** 2):.3f} MiB"
    )
    if artifact_root is not None:
        save_path = os.path.join(artifact_root, "shared_geometry_chunk_local.ply")
        save_shared_geometry_debug_pcd(shared_geometry, save_path)


def prepare_predictions_for_sample_ids(
    config,
    ctx: Dict,
    sample_global_ids,
    candidate_frame_ids,
    all_selected_global_ids,
    run_count: int,
    artifact_root: str | None = None,
    build_shared_geometry: bool = True,
):
    sample_global_ids = np.asarray(sample_global_ids, dtype=np.int32)
    candidate_frame_ids = np.asarray(candidate_frame_ids, dtype=np.int32)
    all_selected_global_ids = np.asarray(all_selected_global_ids, dtype=np.int32)

    rgb_batch = [ctx["rgb_paths"][int(i)] for i in sample_global_ids]
    depth_batch = [ctx["depth_paths"][int(i)] for i in sample_global_ids]
    pose_batch = [ctx["poses"][int(i)] for i in sample_global_ids]

    logger.info(f"Running RGB-D forward pass for run_count={run_count}.")
    depth_scale = ctx["depth_scale"]
    intrinsics = ctx["intrinsics"]
    predictions = {
        "images": np.stack([np.array(Image.open(p)) for p in rgb_batch], axis=0),
        "depth": np.stack([np.array(Image.open(p)) / depth_scale for p in depth_batch], axis=0),
        "extrinsic": np.stack(pose_batch, axis=0),
        "intrinsic": np.stack([intrinsics] * len(rgb_batch), axis=0),
        "T_world_local": np.eye(4),
        "aligned_pose": np.stack(pose_batch, axis=0)[-1],
    }
    predictions["frame_ids_global"] = sample_global_ids.copy()
    predictions["candidate_frame_ids_global"] = candidate_frame_ids.copy()
    predictions["all_selected_frame_ids_global"] = all_selected_global_ids.copy()
    predictions["rgb_paths_all"] = list(ctx["rgb_paths"])
    logger.info(f"RGB-D forward pass finished for run_count={run_count}.")
    logger.info(
        f"Memory estimate after RGB-D forward pass: "
        f"images={estimate_numpy_nbytes(predictions.get('images')) / (1024 ** 2):.1f} MiB "
        f"depth={estimate_numpy_nbytes(predictions.get('depth')) / (1024 ** 2):.1f} MiB"
    )

    if build_shared_geometry:
        geometry_cfg = get_object_segmentation_geometry_config(config)
        merge_cfg = get_object_segmentation_stage_config(config, "merge")
        max_backproj_depth = float(merge_cfg.get("max_backproj_depth", -1.0))
        if max_backproj_depth <= 0:
            max_backproj_depth = None
        attach_shared_geometry_to_predictions(
            predictions,
            shared_geometry_voxel_size=float(geometry_cfg.get("shared_geometry_voxel_size", 0.01)),
            max_backproj_depth=max_backproj_depth,
            run_count=run_count,
            artifact_root=artifact_root,
        )
    return predictions


def prepare_tmp_image_dir(root_dir: str) -> str:
    tmp_img_dir = os.path.join(root_dir, "tmp_images")
    os.makedirs(tmp_img_dir, exist_ok=True)
    for name in os.listdir(tmp_img_dir):
        path = os.path.join(tmp_img_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    return tmp_img_dir


def object_color(color_seed: int, obj_id: int):
    rng = np.random.default_rng(int(color_seed) + int(obj_id))
    return rng.random(3)


def paint_object_pcd(color_seed: int, obj_id: int, pcd: o3d.geometry.PointCloud):
    if pcd is None or len(pcd.points) == 0:
        return pcd
    pcd.paint_uniform_color(object_color(color_seed, obj_id).tolist())
    return pcd
