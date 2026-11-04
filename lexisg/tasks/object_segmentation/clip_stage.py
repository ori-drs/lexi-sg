from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from loguru import logger

from lexisg.utils.config import get_object_segmentation_stage_config
from lexisg.models.clip_handler import CLIP
from lexisg.tasks.object_segmentation import artifact_io
from lexisg.tasks.object_segmentation.merge_stage import BaseMergeStrategy, MergeBundle, MergeStrategyContext, l2_normalize
from lexisg.tasks.object_segmentation.common import paint_object_pcd


def rank_frame_ids_per_object(
    masks: Dict[int, Dict[int, np.ndarray]],
    weights: Dict[str, float],
) -> Dict[int, List[int]]:
    w_center = float(weights.get("center", 1.0))
    w_area = float(weights.get("area", 1.0))
    w_fill = float(weights.get("fill_ratio", 1.0))

    per_obj = {}
    for fid, per_obj_masks in masks.items():
        for obj_id, mask in per_obj_masks.items():
            mask_bool = np.asarray(mask).astype(bool)
            area = int(mask_bool.sum())
            if area <= 0:
                continue

            ys, xs = np.where(mask_bool)
            if ys.size == 0:
                continue

            h, w = mask_bool.shape
            cx_img = (w - 1) * 0.5
            cy_img = (h - 1) * 0.5
            cx_mask = float(xs.mean())
            cy_mask = float(ys.mean())
            dist = float(np.hypot(cx_mask - cx_img, cy_mask - cy_img))
            max_dist = float(np.hypot(cx_img, cy_img))
            center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 0.0
            center_score = float(np.clip(center_score, 0.0, 1.0))

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bbox_w = max(float(x2 - x1), 1.0)
            bbox_h = max(float(y2 - y1), 1.0)
            bbox_area = bbox_w * bbox_h
            fill_ratio = float(area / bbox_area) if bbox_area > 0 else 0.0

            per_obj.setdefault(int(obj_id), []).append(
                {
                    "fid": int(fid),
                    "center": center_score,
                    "area": float(area),
                    "fill_ratio": float(fill_ratio),
                }
            )

    ranked = {}
    for obj_id, vals in per_obj.items():
        if not vals:
            continue

        center_vals = np.array([v["center"] for v in vals], dtype=np.float32)
        area_vals = np.array([v["area"] for v in vals], dtype=np.float32)
        fill_vals = np.array([v["fill_ratio"] for v in vals], dtype=np.float32)

        def _normalize(x: np.ndarray):
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            if x_max - x_min < 1e-12:
                return np.ones_like(x, dtype=np.float32)
            return (x - x_min) / (x_max - x_min)

        score = (
            (w_center * _normalize(center_vals))
            + (w_area * _normalize(area_vals))
            + (w_fill * _normalize(fill_vals))
        )
        order = np.argsort(-score)
        ranked[int(obj_id)] = [int(vals[i]["fid"]) for i in order]
    return ranked


def merge_object_features(
    obj_clip_feats: Dict[int, np.ndarray],
    obj_clip_crops: Dict[int, List[Tuple[int, int, np.ndarray]]],
    merge_map: Dict[int, int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[Tuple[int, int, np.ndarray]]]]:
    if not merge_map:
        return obj_clip_feats, obj_clip_crops

    grouped_feats = defaultdict(list)
    grouped_crops = defaultdict(list)
    for obj_id, feat in obj_clip_feats.items():
        grouped_feats[int(merge_map.get(int(obj_id), int(obj_id)))].append(np.asarray(feat).reshape(-1))
    for obj_id, crops in obj_clip_crops.items():
        grouped_crops[int(merge_map.get(int(obj_id), int(obj_id)))].extend(list(crops))

    merged_feats = {}
    for obj_id, feats in grouped_feats.items():
        feat_avg = np.mean(np.stack(feats, axis=0), axis=0)
        merged_feats[int(obj_id)] = l2_normalize(feat_avg).astype(np.float32)

    merged_crops = {}
    for obj_id, crops in grouped_crops.items():
        merged_crops[int(obj_id)] = sorted(crops, key=lambda item: (int(item[1]), int(item[0])))
    return merged_feats, merged_crops


def filter_objects_without_clip(bundle: MergeBundle) -> Tuple[MergeBundle, List[int]]:
    keep_ids = {int(obj_id) for obj_id in bundle.obj_clip_feats.keys()}
    removed_ids = sorted(int(obj_id) for obj_id in bundle.obj_pcds_world.keys() if int(obj_id) not in keep_ids)

    filtered_masks = {}
    for fid, per_frame in bundle.masks.items():
        kept = {
            int(obj_id): mask
            for obj_id, mask in per_frame.items()
            if int(obj_id) in keep_ids
        }
        if kept:
            filtered_masks[int(fid)] = kept

    return (
        MergeBundle(
            masks=filtered_masks,
            obj_pcds_world={
                int(obj_id): pcd
                for obj_id, pcd in bundle.obj_pcds_world.items()
                if int(obj_id) in keep_ids
            },
            obj_geometry={
                int(obj_id): meta
                for obj_id, meta in (bundle.obj_geometry or {}).items()
                if int(obj_id) in keep_ids
            },
            best_fid_per_obj={
                int(obj_id): int(fid)
                for obj_id, fid in (bundle.best_fid_per_obj or {}).items()
                if int(obj_id) in keep_ids
            },
            best_fids_per_obj={
                int(obj_id): [int(fid) for fid in fids]
                for obj_id, fids in (bundle.best_fids_per_obj or {}).items()
                if int(obj_id) in keep_ids
            },
            obj_clip_feats=bundle.obj_clip_feats,
            obj_clip_crops={
                int(obj_id): list(crops)
                for obj_id, crops in (bundle.obj_clip_crops or {}).items()
                if int(obj_id) in keep_ids
            },
            merge_map=dict(bundle.merge_map or {}),
        ),
        removed_ids,
    )


class ClipAwareWorldMergeStrategy(BaseMergeStrategy):
    name = "clip_aware_world"

    def __init__(self, cfg: Dict, ctx: MergeStrategyContext):
        self.cfg = cfg or {}
        self.ctx = ctx

    def is_enabled(self) -> bool:
        return float(self.cfg.get("overlap_threshold", 0.3)) >= 0.0

    def run(self, bundle: MergeBundle) -> Tuple[MergeBundle, Dict[int, int]]:
        if not self.is_enabled():
            return bundle, {}

        object_ids = sorted(
            int(obj_id)
            for obj_id in bundle.obj_pcds_world.keys()
            if int(obj_id) in bundle.obj_clip_feats
        )
        if not object_ids:
            return bundle, {}

        cfg = self.cfg
        voxel_size = float(cfg.get("voxel_size", 0.05))
        overlap_thr = float(cfg.get("overlap_threshold", 0.3))
        min_points = int(cfg.get("min_points", 50))
        clip_sim_thr = float(cfg.get("clip_similarity_threshold", 0.7))

        parent = {tid: tid for tid in object_ids}
        unions = []

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
            unions.append((a, b, root))

        voxel_sets = {}
        for tid in object_ids:
            pts = np.asarray(bundle.obj_pcds_world[tid].points)
            if pts.shape[0] < min_points:
                continue
            vox = np.floor(pts / voxel_size).astype(np.int32)
            voxel_sets[tid] = set(map(tuple, vox))

        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                a, b = object_ids[i], object_ids[j]
                va, vb = voxel_sets.get(a), voxel_sets.get(b)
                if not va or not vb:
                    continue
                inter = len(va.intersection(vb))
                if inter == 0:
                    continue
                union_vox = len(va.union(vb))
                iou = inter / max(union_vox, 1)
                if iou < overlap_thr:
                    continue
                feat_a = self.ctx.l2_normalize(np.asarray(bundle.obj_clip_feats[a]).reshape(-1))
                feat_b = self.ctx.l2_normalize(np.asarray(bundle.obj_clip_feats[b]).reshape(-1))
                clip_sim = float(np.dot(feat_a, feat_b))
                if clip_sim_thr >= 0 and clip_sim < clip_sim_thr:
                    continue
                union(a, b)

        groups = defaultdict(list)
        for tid in object_ids:
            groups[find(tid)].append(tid)
        if all(len(v) == 1 for v in groups.values()):
            return bundle, {}

        merge_map = {}
        for root, members in groups.items():
            for tid in members:
                merge_map[int(tid)] = int(root)
        logger.info(
            "CLIP-aware 3D merge summary: "
            f"objects_before={len(object_ids)}, "
            f"objects_after={len(groups)}, "
            f"unions={len(unions)}, "
            f"clip_similarity_threshold={clip_sim_thr}"
        )

        merged_pcds = {}
        for root, members in groups.items():
            merged = o3d.geometry.PointCloud()
            for tid in members:
                merged += bundle.obj_pcds_world[tid]
            merged_pcds[int(root)] = self.ctx.paint_object_pcd(int(root), merged)

        merged_feats, merged_crops = merge_object_features(bundle.obj_clip_feats, bundle.obj_clip_crops, merge_map)

        merged_best_fid = {}
        for tid, fid in (bundle.best_fid_per_obj or {}).items():
            rid = int(merge_map.get(int(tid), int(tid)))
            if rid not in merged_best_fid:
                merged_best_fid[rid] = int(fid)

        merged_best_fids = defaultdict(list)
        for tid, fids in (bundle.best_fids_per_obj or {}).items():
            rid = int(merge_map.get(int(tid), int(tid)))
            merged_best_fids[rid].extend(int(fid) for fid in fids)

        merged_masks = {}
        for fid, per_frame in (bundle.masks or {}).items():
            out = {}
            for tid, mask in per_frame.items():
                rid = int(merge_map.get(int(tid), int(tid)))
                if rid in out:
                    out[rid] = np.logical_or(out[rid], np.asarray(mask).astype(bool))
                else:
                    out[rid] = np.asarray(mask).astype(bool)
            if out:
                merged_masks[int(fid)] = out

        return (
            MergeBundle(
                masks=merged_masks,
                obj_pcds_world=merged_pcds,
                obj_geometry={},
                best_fid_per_obj=merged_best_fid,
                best_fids_per_obj={k: sorted(dict.fromkeys(v)) for k, v in merged_best_fids.items()},
                obj_clip_feats=merged_feats,
                obj_clip_crops=merged_crops,
                merge_map=dict(bundle.merge_map or {}),
            ),
            merge_map,
        )


class ClipStage:
    def __init__(
        self,
        config,
        scene_name: str,
        color_seed: int,
        merge_dir: str,
        clip_dir: str,
        feature_model=None,
        feature_encoder: str = "clip",
    ):
        self.config = config
        self.scene_name = str(scene_name)
        self.color_seed = int(color_seed)
        self.merge_dir = str(merge_dir)
        self.clip_dir = str(clip_dir)
        clip_cfg = get_object_segmentation_stage_config(self.config, "clip")
        self.mask_pad_pixel = int(clip_cfg.get("mask_pad_pixel", 20))
        self.mask_pad_ratio = clip_cfg.get("mask_pad_ratio", -1)
        self.mask_mode = str(clip_cfg.get("mask_mode", "original"))
        self.clip_top_n_frames = int(clip_cfg.get("top_n_frames", 5))
        self.feature_encoder = str(feature_encoder).lower()
        self.owns_model = feature_model is None  # don't release a borrowed model
        self.clip = feature_model

    @staticmethod
    def _l2_normalize(feat: np.ndarray) -> np.ndarray:
        denom = float(np.linalg.norm(feat))
        if denom <= 1e-12:
            return feat
        return feat / denom

    def _get_clip(self):
        if self.clip is None:
            self.clip = CLIP()
        return self.clip

    def _release_clip(self):
        if self.clip is None or not self.owns_model:
            return
        try:
            if hasattr(self.clip, "release"):
                self.clip.release()
        except Exception:
            pass
        self.clip = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _extract_clip_feature_from_mask(self, image: np.ndarray, mask: np.ndarray, clip_model: CLIP):
        mask_bool = mask.astype(bool)
        if mask_bool.sum() == 0:
            return None, None
        ys, xs = np.where(mask_bool)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        h, w, _ = image.shape

        use_ratio = self.mask_pad_ratio is not None and float(self.mask_pad_ratio) > 0
        use_pixel = self.mask_pad_pixel is not None and int(self.mask_pad_pixel) >= 0
        if use_ratio and use_pixel:
            raise ValueError("Use either mask_pad_ratio or mask_pad_pixel, not both.")
        if use_ratio:
            box_h = max(1, ymax - ymin)
            box_w = max(1, xmax - xmin)
            pad_y = int(box_h * (float(self.mask_pad_ratio) - 1) / 2)
            pad_x = int(box_w * (float(self.mask_pad_ratio) - 1) / 2)
        elif use_pixel:
            pad_y = int(self.mask_pad_pixel)
            pad_x = int(self.mask_pad_pixel)
        else:
            pad_y = 0
            pad_x = 0

        ymin = max(0, ymin - pad_y)
        ymax = min(h, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(w, xmax + pad_x)

        if self.mask_mode == "white_bg":
            masked_rgb = np.ones_like(image) * 255
        elif self.mask_mode == "black_bg":
            masked_rgb = np.zeros_like(image)
        else:
            masked_rgb = image.copy()
        masked_rgb[mask_bool, :] = image[mask_bool, :]
        crop = masked_rgb[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            return None, None
        feat = clip_model.get_img_feat(crop).squeeze(0)
        return feat, crop

    def _run_clip_merge(self, bundle: MergeBundle):
        strategy = ClipAwareWorldMergeStrategy(
            get_object_segmentation_stage_config(self.config, "merge").get("clip_aware_overlap_merge", {}),
            MergeStrategyContext(
                paint_object_pcd=lambda obj_id, pcd: paint_object_pcd(self.color_seed, obj_id, pcd),
                load_object_pcd_artifact=artifact_io.load_object_pcd_artifact,
                load_object_voxel_artifact=artifact_io.load_object_voxel_artifact,
                voxelize_points=artifact_io.voxelize_points,
                voxel_rows_view=artifact_io.voxel_rows_view,
                l2_normalize=self._l2_normalize,
            ),
        )
        return strategy.run(bundle)

    def _save_openlex3d_outputs(
        self,
        obj_pcds_world: Dict[int, o3d.geometry.PointCloud],
        obj_clip_feats: Dict[int, np.ndarray],
        save_dir: str,
    ):
        valid_ids = [
            obj_id
            for obj_id in sorted(obj_clip_feats.keys())
            if obj_id in obj_pcds_world and len(obj_pcds_world[obj_id].points) > 0
        ]
        if not valid_ids:
            logger.warning("No valid objects for OpenLex3D output; skipping save.")
            return
        first_feat = np.asarray(obj_clip_feats[valid_ids[0]]).reshape(-1)
        n_dim = first_feat.shape[0]
        all_points, all_colors, index_list, embedding_rows = [], [], [], []
        obj_row = 0
        for obj_id in valid_ids:
            pcd = obj_pcds_world[obj_id]
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            if points.shape[0] == 0 or colors.shape != points.shape:
                continue
            feat = np.asarray(obj_clip_feats[obj_id]).reshape(-1).astype(np.float32)
            if feat.shape[0] != n_dim:
                continue
            keep_mask = np.all(np.isfinite(colors), axis=1) & (~np.all(colors >= 0.95, axis=1)) & (~np.all(colors <= 1e-6, axis=1))
            if not np.any(keep_mask):
                continue
            points = points[keep_mask]
            colors = colors[keep_mask]
            embedding_rows.append(feat)
            all_points.append(points)
            all_colors.append(colors)
            index_list.append(np.full((points.shape[0],), obj_row, dtype=np.int32))
            obj_row += 1
        if not all_points:
            logger.warning("No points aggregated for OpenLex3D output; skipping save.")
            return
        np.save(os.path.join(save_dir, "embeddings.npy"), np.stack(embedding_rows, axis=0).astype(np.float32))
        np.save(os.path.join(save_dir, "index.npy"), np.concatenate(index_list, axis=0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        o3d.io.write_point_cloud(os.path.join(save_dir, "point_cloud.pcd"), pcd)

    def _finalize_outputs(self, obj_pcds_world, obj_clip_feats, obj_clip_crops):
        merged_pcd = o3d.geometry.PointCloud()
        for pcd in obj_pcds_world.values():
            merged_pcd += pcd
        save_dir = os.path.join(self.clip_dir, "segmented_pointcloud")
        feat_save_dir = os.path.join(self.clip_dir, "clip_features")
        crop_save_dir = os.path.join(self.clip_dir, "clip_crops")
        openlex3d_dir = os.path.join(self.clip_dir, "openlex3d_output", self.scene_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(feat_save_dir, exist_ok=True)
        os.makedirs(crop_save_dir, exist_ok=True)
        os.makedirs(openlex3d_dir, exist_ok=True)
        clip_cfg = get_object_segmentation_stage_config(self.config, "clip")
        if bool(clip_cfg.get("save_outputs", True)):
            for tracker_id, pcd in obj_pcds_world.items():
                if len(pcd.points) == 0:
                    continue
                o3d.io.write_point_cloud(os.path.join(save_dir, f"object_{int(tracker_id):04d}.ply"), pcd)
            if len(merged_pcd.points) > 0:
                o3d.io.write_point_cloud(os.path.join(save_dir, "merged.ply"), merged_pcd)
            for obj_id, feat in obj_clip_feats.items():
                np.save(os.path.join(feat_save_dir, f"object_{int(obj_id):04d}.npy"), feat)
            for obj_id, crops in obj_clip_crops.items():
                obj_crop_dir = os.path.join(crop_save_dir, f"object_{int(obj_id):04d}")
                os.makedirs(obj_crop_dir, exist_ok=True)
                for fid, rank, crop in crops:
                    Image.fromarray(crop).save(os.path.join(obj_crop_dir, f"rank_{int(rank):02d}_fid_{int(fid):06d}.jpg"))
        if bool(clip_cfg.get("save_openlex3d", True)):
            if self.feature_encoder == "clip":
                self._save_openlex3d_outputs(obj_pcds_world, obj_clip_feats, openlex3d_dir)
            else:
                logger.info(
                    f"Skipping OpenLex3D export: features are {self.feature_encoder}, "
                    "but OpenLex3D eval expects CLIP ViT-H-14 embeddings."
                )
        return {
            "obj_pcds_world": obj_pcds_world,
            "merged_pcd": merged_pcd,
            "save_dir": save_dir,
            "openlex3d_dir": openlex3d_dir,
        }

    def run(self, ctx: Dict):
        merge_result_path = os.path.join(self.merge_dir, "merge_result.pkl.gz")
        if not os.path.exists(merge_result_path):
            raise ValueError(f"Missing merge stage artifact: {merge_result_path}")
        merged = artifact_io.load_merge_result(merge_result_path)

        merged_masks = merged["masks"]
        merged_pcds = merged["obj_pcds_world"]
        weights = get_object_segmentation_stage_config(self.config, "track").get("best_frame", {}).get("weights", {"center": 1.0, "area": 1.0, "fill_ratio": 1.0})
        best_fids_per_obj = rank_frame_ids_per_object(merged_masks, weights)
        best_fid_per_obj = {obj_id: fids[0] for obj_id, fids in best_fids_per_obj.items() if fids}

        obj_clip_feats = {}
        obj_clip_crops = {}
        clip_model = self._get_clip()
        for obj_id, ranked_fids in best_fids_per_obj.items():
            feats = []
            crops = []
            for rank, fid_global in enumerate(ranked_fids[: max(self.clip_top_n_frames, 1)], start=1):
                mask = merged_masks.get(int(fid_global), {}).get(int(obj_id))
                if mask is None:
                    continue
                image = np.array(Image.open(ctx["rgb_paths"][int(fid_global)]).convert("RGB"))
                feat, crop = self._extract_clip_feature_from_mask(image, mask, clip_model)
                if feat is None:
                    continue
                feats.append(np.asarray(feat).reshape(-1))
                crops.append((int(fid_global), int(rank), crop))
            if not feats:
                continue
            feat_avg = np.mean(np.stack(feats, axis=0), axis=0)
            obj_clip_feats[int(obj_id)] = self._l2_normalize(feat_avg).astype(np.float32)
            obj_clip_crops[int(obj_id)] = crops
        self._release_clip()

        filtered_bundle, removed_no_clip = filter_objects_without_clip(
            MergeBundle(
                masks=merged_masks,
                obj_pcds_world=merged_pcds,
                best_fid_per_obj=best_fid_per_obj,
                best_fids_per_obj=best_fids_per_obj,
                obj_clip_feats=obj_clip_feats,
                obj_clip_crops=obj_clip_crops,
            )
        )
        if removed_no_clip:
            logger.info(f"Removed {len(removed_no_clip)} objects without CLIP features: {removed_no_clip}")

        merged_bundle, clip_merge_map = self._run_clip_merge(filtered_bundle)
        best_fids_per_obj = rank_frame_ids_per_object(merged_bundle.masks, weights)
        best_fid_per_obj = {obj_id: fids[0] for obj_id, fids in best_fids_per_obj.items() if fids}

        final_result = self._finalize_outputs(
            obj_pcds_world=merged_bundle.obj_pcds_world,
            obj_clip_feats=merged_bundle.obj_clip_feats,
            obj_clip_crops=merged_bundle.obj_clip_crops,
        )
        artifact_io.save_clip_result(
            self.clip_dir,
            {
                "masks": merged_bundle.masks,
                "best_fid_per_obj": best_fid_per_obj,
                "best_fids_per_obj": best_fids_per_obj,
                "obj_clip_feats": merged_bundle.obj_clip_feats,
                "obj_clip_crops": merged_bundle.obj_clip_crops,
                "obj_pcds_world": merged_bundle.obj_pcds_world,
                "merge_map": clip_merge_map,
            },
        )
        return {
            "masks": merged_bundle.masks,
            "best_fid_per_obj": best_fid_per_obj,
            "best_fids_per_obj": best_fids_per_obj,
            "obj_clip_feats": merged_bundle.obj_clip_feats,
            "obj_clip_crops": merged_bundle.obj_clip_crops,
            "obj_pcds_world": merged_bundle.obj_pcds_world,
            "merge_map": clip_merge_map,
            **final_result,
        }
