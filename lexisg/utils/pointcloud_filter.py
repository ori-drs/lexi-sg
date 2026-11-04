from __future__ import annotations

import numpy as np
import open3d as o3d

from lexisg.utils.config import get_object_segmentation_stage_config


class PerFramePointCloudFilter:
    def __init__(self, config):
        merge_cfg = get_object_segmentation_stage_config(config, "merge")
        cfg = dict(merge_cfg.get("object_pcd_per_frame_filter", {}) or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.voxel_size = float(cfg.get("voxel_size", 0.01))

        dbscan_cfg = dict(cfg.get("dbscan", {}) or {})
        self.dbscan_enabled = bool(dbscan_cfg.get("enabled", False))
        self.dbscan_eps = float(dbscan_cfg.get("eps", 0.02))
        self.dbscan_min_points = int(dbscan_cfg.get("min_points", 10))
        self.dbscan_min_cluster_size = int(dbscan_cfg.get("min_cluster_size", self.dbscan_min_points))
        self.dbscan_keep_largest_cluster = bool(dbscan_cfg.get("keep_largest_cluster", True))

    def apply(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if pcd is None:
            return o3d.geometry.PointCloud()

        filtered = o3d.geometry.PointCloud(pcd)
        if len(filtered.points) == 0 or not self.enabled:
            return filtered

        if self.voxel_size > 0:
            filtered = filtered.voxel_down_sample(self.voxel_size)
        if len(filtered.points) == 0 or not self.dbscan_enabled:
            return filtered

        labels = np.asarray(
            filtered.cluster_dbscan(
                eps=self.dbscan_eps,
                min_points=self.dbscan_min_points,
                print_progress=False,
            ),
            dtype=np.int32,
        )
        if labels.size == 0:
            return filtered

        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            return o3d.geometry.PointCloud()

        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        if self.dbscan_keep_largest_cluster:
            kept_labels = [int(unique_labels[int(np.argmax(counts))])]
        else:
            kept_labels = [
                int(label)
                for label, count in zip(unique_labels.tolist(), counts.tolist())
                if int(count) >= self.dbscan_min_cluster_size
            ]
            if not kept_labels:
                kept_labels = [int(unique_labels[int(np.argmax(counts))])]

        keep_mask = np.isin(labels, np.asarray(kept_labels, dtype=np.int32))
        keep_indices = np.flatnonzero(keep_mask).astype(np.int32)
        if keep_indices.size == 0:
            return o3d.geometry.PointCloud()
        return filtered.select_by_index(keep_indices.tolist())
