from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import open3d as o3d
from loguru import logger

from lexisg.utils.config import get_object_segmentation_stage_config
import lexisg.utils.geometry as geometry
from lexisg.tasks.object_segmentation import artifact_io
from lexisg.tasks.object_segmentation.common import (
    load_images_for_frame_ids,
    paint_object_pcd,
    prepare_predictions_for_sample_ids,
)
from lexisg.utils.visualization import save_global_mask_overlays
from lexisg.utils.pointcloud_filter import PerFramePointCloudFilter


@dataclass
class MergeBundle:
    masks: Dict[int, Dict[int, np.ndarray]]
    obj_pcds_world: Dict[int, o3d.geometry.PointCloud] = field(default_factory=dict)
    obj_geometry: Dict[int, Dict] = field(default_factory=dict)
    best_fid_per_obj: Dict[int, int] = field(default_factory=dict)
    best_fids_per_obj: Dict[int, List[int]] = field(default_factory=dict)
    obj_clip_feats: Dict[int, np.ndarray] = field(default_factory=dict)
    obj_clip_crops: Dict[int, List[Tuple[int, int, np.ndarray]]] = field(default_factory=dict)
    merge_map: Dict[int, int] = field(default_factory=dict)


@dataclass
class MergeStrategyContext:
    paint_object_pcd: Callable[[int, o3d.geometry.PointCloud], o3d.geometry.PointCloud]
    load_object_pcd_artifact: Callable[[str], o3d.geometry.PointCloud]
    load_object_voxel_artifact: Callable[[str], Tuple[np.ndarray, float]]
    voxelize_points: Callable[[np.ndarray, float], np.ndarray]
    voxel_rows_view: Callable[[np.ndarray], np.ndarray]
    l2_normalize: Callable[[np.ndarray], np.ndarray]


def l2_normalize(feat: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(feat))
    if denom <= 1e-12:
        return feat
    return feat / denom


class ObjectGeometryBuilder:
    def __init__(self, config):
        self.config = config
        merge_cfg = get_object_segmentation_stage_config(config, "merge")
        self.max_backproj_depth = float(merge_cfg.get("max_backproj_depth", -1.0))
        if self.max_backproj_depth <= 0:
            self.max_backproj_depth = None
        self.per_frame_filter = PerFramePointCloudFilter(config)

    def build_local_object_pcds(self, predictions, masks_local: Dict[int, Dict[int, np.ndarray]]):
        pcd_per_obj = defaultdict(o3d.geometry.PointCloud)
        for fid_local, per_frame in sorted(masks_local.items()):
            fid_local = int(fid_local)
            for obj_id, mask in sorted(per_frame.items()):
                mask = np.asarray(mask).astype(bool)
                if mask.sum() == 0:
                    continue
                mask_pcd = geometry.backproject_mask_from_predictions(
                    predictions,
                    fid_local,
                    mask,
                    depth_scale=1.0,
                    max_depth=self.max_backproj_depth,
                    erode_iters=1,
                )
                mask_pcd = self.per_frame_filter.apply(mask_pcd)
                if len(mask_pcd.points) == 0:
                    continue
                pcd_per_obj[int(obj_id)] += mask_pcd
        return {int(obj_id): pcd for obj_id, pcd in pcd_per_obj.items()}


class BaseMergeStrategy:
    name = "base"

    def is_enabled(self) -> bool:
        return True

    def run(self, bundle: MergeBundle) -> Tuple[MergeBundle, Dict[int, int]]:
        raise NotImplementedError


class OverlapMergeStrategy(BaseMergeStrategy):
    name = "overlap"

    def __init__(self, cfg: Dict, ctx: MergeStrategyContext):
        self.cfg = cfg or {}
        self.ctx = ctx

    def is_enabled(self) -> bool:
        return float(self.cfg.get("overlap_threshold", 0.3)) >= 0.0

    def _materialize_pcds(self, bundle: MergeBundle) -> Dict[int, o3d.geometry.PointCloud]:
        if bundle.obj_pcds_world:
            return bundle.obj_pcds_world
        if not bundle.obj_geometry:
            return {}
        return {
            int(obj_id): self.ctx.paint_object_pcd(
                int(obj_id),
                self.ctx.load_object_pcd_artifact(meta["pcd_path"]),
            )
            for obj_id, meta in bundle.obj_geometry.items()
        }

    def run(self, bundle: MergeBundle) -> Tuple[MergeBundle, Dict[int, int]]:
        if not self.is_enabled():
            return MergeBundle(
                masks=bundle.masks,
                obj_pcds_world=self._materialize_pcds(bundle),
                obj_geometry=dict(bundle.obj_geometry or {}),
                best_fid_per_obj=dict(bundle.best_fid_per_obj or {}),
                best_fids_per_obj=dict(bundle.best_fids_per_obj or {}),
                obj_clip_feats=dict(bundle.obj_clip_feats or {}),
                obj_clip_crops=dict(bundle.obj_clip_crops or {}),
                merge_map=dict(bundle.merge_map or {}),
            ), {}

        object_ids = sorted({int(tid) for per_frame in bundle.masks.values() for tid in per_frame.keys()})
        if not object_ids:
            return bundle, {}

        parent = {tid: tid for tid in object_ids}
        unions_3d = []

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            root = min(ra, rb)
            other = rb if root == ra else ra
            parent[other] = root
            unions_3d.append((a, b, root))

        overlap_cfg = self.cfg
        overlap_thr = float(overlap_cfg.get("overlap_threshold", 0.3))
        if overlap_thr >= 0.0:
            voxel_size = float(overlap_cfg.get("voxel_size", 0.05))
            min_points = int(overlap_cfg.get("min_points", 50))

            voxel_infos = {}
            if bundle.obj_geometry:
                for tid, meta in bundle.obj_geometry.items():
                    tid = int(tid)
                    if int(meta.get("point_count", 0)) < min_points:
                        continue
                    voxel_indices, saved_voxel_size = self.ctx.load_object_voxel_artifact(meta["voxel_path"])
                    if voxel_indices.shape[0] == 0:
                        continue
                    if saved_voxel_size > 0 and abs(saved_voxel_size - voxel_size) > 1e-6:
                        logger.warning(
                            f"Voxel size mismatch for object {tid}: saved={saved_voxel_size}, requested={voxel_size}. "
                            "Using saved voxel indices."
                        )
                    voxel_infos[tid] = {
                        "rows": self.ctx.voxel_rows_view(voxel_indices),
                        "bounds_min": np.asarray(meta.get("voxel_bounds_min", voxel_indices.min(axis=0)), dtype=np.int32),
                        "bounds_max": np.asarray(meta.get("voxel_bounds_max", voxel_indices.max(axis=0)), dtype=np.int32),
                    }
            else:
                for tid, pcd in bundle.obj_pcds_world.items():
                    tid = int(tid)
                    pts = np.asarray(pcd.points)
                    if pts.shape[0] < min_points:
                        continue
                    voxel_indices = self.ctx.voxelize_points(pts, voxel_size)
                    if voxel_indices.shape[0] == 0:
                        continue
                    voxel_infos[tid] = {
                        "rows": self.ctx.voxel_rows_view(voxel_indices),
                        "bounds_min": voxel_indices.min(axis=0),
                        "bounds_max": voxel_indices.max(axis=0),
                    }

            tids = sorted(voxel_infos.keys())
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    a, b = tids[i], tids[j]
                    info_a, info_b = voxel_infos[a], voxel_infos[b]
                    if np.any(info_a["bounds_max"] < info_b["bounds_min"]) or np.any(info_b["bounds_max"] < info_a["bounds_min"]):
                        continue
                    inter = int(np.intersect1d(info_a["rows"], info_b["rows"], assume_unique=True).shape[0])
                    if inter == 0:
                        continue
                    union_vox = int(info_a["rows"].shape[0] + info_b["rows"].shape[0] - inter)
                    iou = inter / max(union_vox, 1)
                    if iou >= overlap_thr:
                        union(a, b)

        groups = defaultdict(list)
        for tid in object_ids:
            groups[find(tid)].append(tid)

        merged_groups = {int(k): sorted(int(x) for x in v) for k, v in groups.items() if len(v) > 1}
        logger.info(
            "Post-merge summary: "
            f"objects_before={len(object_ids)}, "
            f"objects_after={len(groups)}, "
            f"unions_3d={len(unions_3d)}, "
            f"merged_groups={len(merged_groups)}"
        )
        if unions_3d:
            logger.debug(f"Post-merge 3D unions: {unions_3d}")
        if merged_groups:
            logger.info(f"Post-merge groups: {merged_groups}")

        if all(len(v) == 1 for v in groups.values()):
            return MergeBundle(
                masks=bundle.masks,
                obj_pcds_world=self._materialize_pcds(bundle),
                obj_geometry=dict(bundle.obj_geometry or {}),
                best_fid_per_obj=dict(bundle.best_fid_per_obj or {}),
                best_fids_per_obj=dict(bundle.best_fids_per_obj or {}),
                obj_clip_feats=dict(bundle.obj_clip_feats or {}),
                obj_clip_crops=dict(bundle.obj_clip_crops or {}),
                merge_map=dict(bundle.merge_map or {}),
            ), {}

        merge_map = {}
        for root, members in groups.items():
            for tid in members:
                merge_map[int(tid)] = int(root)

        merged_masks = {}
        for fid, per_frame in bundle.masks.items():
            out = {}
            for tid, mask in per_frame.items():
                rid = merge_map.get(int(tid), int(tid))
                if rid in out:
                    out[rid] = np.logical_or(out[rid], np.asarray(mask).astype(bool))
                else:
                    out[rid] = np.asarray(mask).astype(bool)
            merged_masks[int(fid)] = out

        merged_pcd_per_obj = {}
        for rid, members in groups.items():
            merged = o3d.geometry.PointCloud()
            for tid in members:
                if tid in bundle.obj_pcds_world:
                    merged += bundle.obj_pcds_world[tid]
                elif bundle.obj_geometry and tid in bundle.obj_geometry:
                    merged += self.ctx.load_object_pcd_artifact(bundle.obj_geometry[tid]["pcd_path"])
            merged_pcd_per_obj[int(rid)] = self.ctx.paint_object_pcd(int(rid), merged)

        return (
            MergeBundle(
                masks=merged_masks,
                obj_pcds_world=merged_pcd_per_obj,
                obj_geometry={},
                best_fid_per_obj=dict(bundle.best_fid_per_obj or {}),
                best_fids_per_obj=dict(bundle.best_fids_per_obj or {}),
                obj_clip_feats=dict(bundle.obj_clip_feats or {}),
                obj_clip_crops=dict(bundle.obj_clip_crops or {}),
                merge_map=dict(bundle.merge_map or {}),
            ),
            merge_map,
        )


class MergeStage:
    def __init__(self, config, scene_name: str, color_seed: int, track_dir: str, merge_dir: str):
        self.config = config
        self.scene_name = str(scene_name)
        self.color_seed = int(color_seed)
        self.track_dir = str(track_dir)
        self.merge_dir = str(merge_dir)
        self.geometry_builder = ObjectGeometryBuilder(config)
        merge_cfg = get_object_segmentation_stage_config(self.config, "merge")
        self.object_pcd_voxel_size = float(merge_cfg.get("object_pcd_voxel_size", 0.01))

    def _object_color(self, obj_id: int):
        rng = np.random.default_rng(self.color_seed + int(obj_id))
        return rng.random(3)

    def _voxelize_object_pcd(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if pcd is None or len(pcd.points) == 0:
            return pcd
        if self.object_pcd_voxel_size <= 0:
            return pcd
        return pcd.voxel_down_sample(self.object_pcd_voxel_size)

    def _load_track_result(self):
        result_path = os.path.join(self.track_dir, "track_result.pkl.gz")
        if not os.path.exists(result_path):
            raise ValueError(f"Missing track stage artifact: {result_path}")
        return artifact_io.load_track_result(result_path)

    @staticmethod
    def _load_frame_detections(detections_dir: str, fid_global: int):
        path = os.path.join(detections_dir, f"frame_{int(fid_global):06d}.pkl.gz")
        if not os.path.exists(path):
            return None
        return artifact_io.load_pickle_gz(path)

    def _iter_frame_detections(self, track_result):
        detections_dir = str(track_result["detections_dir"])
        for fid_global in [int(fid) for fid in track_result.get("sampled_frame_ids", [])]:
            dets = self._load_frame_detections(detections_dir, int(fid_global))
            if dets is None or len(dets) == 0 or dets.mask is None:
                yield int(fid_global), []
                continue
            masks = np.asarray(dets.mask).astype(bool)
            class_ids = np.asarray(dets.class_id, dtype=np.int32)
            entries = []
            for idx in range(len(class_ids)):
                entries.append(
                    {
                        "obj_id": int(class_ids[idx]),
                        "mask": masks[idx],
                    }
                )
            yield int(fid_global), entries

    def _materialize_tracked_masks(self, track_result):
        materialized_masks = defaultdict(dict)
        for fid_global, entries in self._iter_frame_detections(track_result):
            if not entries:
                continue
            per_frame = materialized_masks[int(fid_global)]
            for entry in entries:
                per_frame[int(entry["obj_id"])] = np.asarray(entry["mask"]).astype(bool)
        return {int(fid): dict(per_frame) for fid, per_frame in sorted(materialized_masks.items())}

    def _build_world_pcds_from_masks(self, ctx: Dict, plan: Dict, merged_masks: Dict[int, Dict[int, np.ndarray]]):
        sampled_frame_ids = np.asarray(plan["sampled_frame_ids"], dtype=np.int32)
        predictions = prepare_predictions_for_sample_ids(
            self.config,
            ctx=ctx,
            sample_global_ids=sampled_frame_ids,
            candidate_frame_ids=plan["frame_ids"],
            all_selected_global_ids=plan["sampled_frame_ids"],
            run_count=0,
            artifact_root=None,
            build_shared_geometry=True,
        )
        masks_local = {}
        for fid_local, fid_global in enumerate(sampled_frame_ids.tolist()):
            per_frame = merged_masks.get(int(fid_global), {})
            if per_frame:
                masks_local[int(fid_local)] = {
                    int(obj_id): np.asarray(mask).astype(bool)
                    for obj_id, mask in per_frame.items()
                }
        if not masks_local:
            return {}
        local_pcds = self.geometry_builder.build_local_object_pcds(predictions, masks_local)
        aggregated_pcds: Dict[int, o3d.geometry.PointCloud] = {}
        T_world_local = predictions["T_world_local"]
        for obj_id, pcd in local_pcds.items():
            obj_id = int(obj_id)
            pcd_world = o3d.geometry.PointCloud(pcd)
            pcd_world.transform(T_world_local)
            pcd_world = self._voxelize_object_pcd(pcd_world)
            if len(pcd_world.points) > 0:
                pcd_world.paint_uniform_color(self._object_color(obj_id).tolist())
            aggregated_pcds[obj_id] = pcd_world
        return aggregated_pcds

    def _build_world_pcds_from_predictions(self, predictions: Dict, sampled_frame_ids, merged_masks: Dict[int, Dict[int, np.ndarray]]):
        sampled_frame_ids = np.asarray(sampled_frame_ids, dtype=np.int32)
        masks_local = {}
        for fid_local, fid_value in enumerate(sampled_frame_ids.tolist()):
            per_frame = merged_masks.get(int(fid_value), {})
            if per_frame:
                masks_local[int(fid_local)] = {
                    int(obj_id): np.asarray(mask).astype(bool)
                    for obj_id, mask in per_frame.items()
                }
        if not masks_local:
            return {}

        local_pcds = self.geometry_builder.build_local_object_pcds(predictions, masks_local)
        aggregated_pcds: Dict[int, o3d.geometry.PointCloud] = {}
        T_world_local = np.asarray(predictions.get("T_world_local", np.eye(4)), dtype=np.float64)
        for obj_id, pcd in local_pcds.items():
            obj_id = int(obj_id)
            pcd_world = o3d.geometry.PointCloud(pcd)
            pcd_world.transform(T_world_local)
            pcd_world = self._voxelize_object_pcd(pcd_world)
            if len(pcd_world.points) > 0:
                pcd_world.paint_uniform_color(self._object_color(obj_id).tolist())
            aggregated_pcds[obj_id] = pcd_world
        return aggregated_pcds

    def _run_3d_merge(self, masks, pcd_per_obj):
        merge_cfg = get_object_segmentation_stage_config(self.config, "merge")
        overlap_cfg = dict(merge_cfg.get("overlap_merge", {}))
        strategy = OverlapMergeStrategy(
            overlap_cfg,
            MergeStrategyContext(
                paint_object_pcd=lambda obj_id, pcd: paint_object_pcd(self.color_seed, obj_id, pcd),
                load_object_pcd_artifact=artifact_io.load_object_pcd_artifact,
                load_object_voxel_artifact=artifact_io.load_object_voxel_artifact,
                voxelize_points=artifact_io.voxelize_points,
                voxel_rows_view=artifact_io.voxel_rows_view,
                l2_normalize=lambda feat: feat,
            ),
        )
        bundle, merge_map = strategy.run(
            MergeBundle(
                masks=masks,
                obj_pcds_world=pcd_per_obj,
            )
        )
        return bundle.masks, bundle.obj_pcds_world, merge_map

    def run(self, ctx: Dict, plan: Dict):
        merge_cfg = get_object_segmentation_stage_config(self.config, "merge")
        save_visualizations = bool(merge_cfg.get("save_visualizations", True))
        track_result = self._load_track_result()
        tracked_masks = self._materialize_tracked_masks(track_result)
        tracked_pcds = self._build_world_pcds_from_masks(ctx, plan, tracked_masks)

        merged_masks, merged_pcds, merge_map_3d = self._run_3d_merge(
            masks=tracked_masks,
            pcd_per_obj=tracked_pcds,
        )
        result_path = artifact_io.save_merge_result(
            self.merge_dir,
            {
                "masks": merged_masks,
                "obj_pcds_world": merged_pcds,
                "merge_map": merge_map_3d,
            },
        )
        if save_visualizations:
            save_global_mask_overlays(
                images={
                    int(fid): np.asarray(image, dtype=np.uint8)
                    for fid, image in zip(
                        np.asarray(plan["sampled_frame_ids"], dtype=np.int32).tolist(),
                        load_images_for_frame_ids(ctx["rgb_paths"], plan["sampled_frame_ids"]),
                    )
                },
                masks=merged_masks,
                save_dir=os.path.join(self.merge_dir, "visualizations", "merge_merged"),
                frame_ids=np.asarray(plan["sampled_frame_ids"], dtype=np.int32),
            )
        return {
            "result_path": result_path,
            "masks": merged_masks,
            "obj_pcds_world": merged_pcds,
            "merge_map": merge_map_3d,
        }

    def run_with_predictions(self, predictions: Dict, sampled_frame_ids):
        merge_cfg = get_object_segmentation_stage_config(self.config, "merge")
        save_visualizations = bool(merge_cfg.get("save_visualizations", True))
        track_result = self._load_track_result()
        tracked_masks = self._materialize_tracked_masks(track_result)
        tracked_pcds = self._build_world_pcds_from_predictions(
            predictions,
            sampled_frame_ids=sampled_frame_ids,
            merged_masks=tracked_masks,
        )

        merged_masks, merged_pcds, merge_map_3d = self._run_3d_merge(
            masks=tracked_masks,
            pcd_per_obj=tracked_pcds,
        )
        result_path = artifact_io.save_merge_result(
            self.merge_dir,
            {
                "masks": merged_masks,
                "obj_pcds_world": merged_pcds,
                "merge_map": merge_map_3d,
            },
        )
        if save_visualizations and predictions.get("images") is not None:
            save_global_mask_overlays(
                images={
                    int(fid): np.asarray(predictions["images"][fid_local], dtype=np.uint8)
                    for fid_local, fid in enumerate(np.asarray(sampled_frame_ids, dtype=np.int32).tolist())
                },
                masks=merged_masks,
                save_dir=os.path.join(self.merge_dir, "visualizations", "merge_merged"),
                frame_ids=np.asarray(sampled_frame_ids, dtype=np.int32),
            )
        return {
            "result_path": result_path,
            "masks": merged_masks,
            "obj_pcds_world": merged_pcds,
            "merge_map": merge_map_3d,
        }
