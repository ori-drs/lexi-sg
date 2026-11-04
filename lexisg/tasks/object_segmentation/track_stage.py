from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from typing import Dict

import numpy as np
from PIL import Image
from supervision import Detections

from lexisg.utils.config import get_object_segmentation_stage_config
from lexisg.models.bidirectional_track_handler import BidirectionalTrackHandler
from lexisg.tasks.object_segmentation import artifact_io
from lexisg.tasks.object_segmentation.common import load_images_for_frame_ids, prepare_tmp_image_dir
from lexisg.utils.visualization import save_global_mask_overlays
from lexisg.utils.sv_compat import mask_to_xyxy


class TrackStage:
    def __init__(self, config, detect_dir: str, track_dir: str):
        self.config = config
        self.detect_dir = str(detect_dir)
        self.track_dir = str(track_dir)
        self.tracker = BidirectionalTrackHandler(self.config)

    @staticmethod
    def _load_detection_candidates(manifest_path: str):
        payload = artifact_io.load_pickle_gz(manifest_path)
        return list(payload.get("candidates", []))

    @staticmethod
    def _make_detection_mask_loader(detections_dir: str):
        frame_cache = OrderedDict()
        max_cache_frames = 4

        def _load_frame_dets(fid_local: int):
            fid_local = int(fid_local)
            cached = frame_cache.get(fid_local)
            if cached is not None:
                frame_cache.move_to_end(fid_local)
                return cached
            dets = artifact_io.load_pickle_gz(os.path.join(detections_dir, f"frame_{fid_local:06d}.pkl.gz"))
            frame_cache[fid_local] = dets
            frame_cache.move_to_end(fid_local)
            while len(frame_cache) > max_cache_frames:
                frame_cache.popitem(last=False)
            return dets

        def _load_mask(fid_local: int, det_idx: int):
            dets = _load_frame_dets(fid_local)
            if dets is None or dets.mask is None or int(det_idx) >= len(dets.mask):
                return None
            return np.asarray(dets.mask[int(det_idx)], dtype=bool)

        return _load_mask

    @staticmethod
    def _detections_from_masks(per_frame: Dict[int, np.ndarray]):
        if not per_frame:
            return None
        masks = [
            (int(obj_id), np.asarray(mask, dtype=bool))
            for obj_id, mask in sorted(per_frame.items())
            if np.asarray(mask, dtype=bool).sum() > 0
        ]
        if not masks:
            return None
        class_ids = np.asarray([obj_id for obj_id, _ in masks], dtype=np.int32)
        mask_array = np.stack([mask for _, mask in masks], axis=0)
        return Detections(
            xyxy=mask_to_xyxy(mask_array),
            mask=mask_array,
            class_id=class_ids.copy(),
            tracker_id=class_ids.copy(),
            confidence=np.ones((mask_array.shape[0],), dtype=np.float32),
        )

    def _save_tracked_detections(self, sampled_frame_ids: np.ndarray, masks_local: Dict[int, Dict[int, np.ndarray]]):
        detections_dir = os.path.join(self.track_dir, "tracked_detections")
        os.makedirs(detections_dir, exist_ok=True)
        tracklet_ids = set()
        for fid_local, fid_global in enumerate(np.asarray(sampled_frame_ids, dtype=np.int32).tolist()):
            per_frame = {
                int(obj_id): np.asarray(mask, dtype=bool)
                for obj_id, mask in masks_local.get(int(fid_local), {}).items()
            }
            tracklet_ids.update(int(obj_id) for obj_id in per_frame.keys())
            artifact_io.save_pickle_gz(
                os.path.join(detections_dir, f"frame_{int(fid_global):06d}.pkl.gz"),
                self._detections_from_masks(per_frame),
            )
        return detections_dir, sorted(int(obj_id) for obj_id in tracklet_ids)

    @staticmethod
    def _save_track_result(stage_dir: str, result: Dict):
        return artifact_io.save_track_result(stage_dir, result)

    def run(self, ctx: Dict, plan: Dict):
        stage_cfg = get_object_segmentation_stage_config(self.config, "track")
        save_visualizations = bool(stage_cfg.get("save_visualizations", True))
        sampled_frame_ids = np.asarray(plan["sampled_frame_ids"], dtype=np.int32)
        runtime_dir = os.path.join(self.track_dir, "track_runtime")
        images = load_images_for_frame_ids(ctx["rgb_paths"], sampled_frame_ids)
        tracking_inputs = {
            "images": images,
            "frame_ids_global": sampled_frame_ids.copy(),
        }
        tmp_img_dir = prepare_tmp_image_dir(runtime_dir)
        for i, img in enumerate(images):
            Image.fromarray(img).save(os.path.join(tmp_img_dir, f"{i:03d}.jpg"))

        candidates = self._load_detection_candidates(os.path.join(self.detect_dir, "detect_candidates.pkl.gz"))
        load_candidate_mask = self._make_detection_mask_loader(os.path.join(self.detect_dir, "detections"))
        masks_local, _ = self.tracker.track_from_candidates(
            tracking_inputs=tracking_inputs,
            tmp_img_dir=tmp_img_dir,
            candidates=candidates,
            load_candidate_mask=load_candidate_mask,
        )

        tracked_detections_dir, tracklet_ids = self._save_tracked_detections(sampled_frame_ids, masks_local)
        if save_visualizations:
            save_global_mask_overlays(
                images={
                    int(fid_global): np.asarray(images[fid_local], dtype=np.uint8)
                    for fid_local, fid_global in enumerate(sampled_frame_ids.tolist())
                },
                masks={
                    int(sampled_frame_ids[fid_local]): {
                        int(obj_id): np.asarray(mask, dtype=bool)
                        for obj_id, mask in per_frame.items()
                    }
                    for fid_local, per_frame in masks_local.items()
                },
                save_dir=os.path.join(self.track_dir, "visualizations", "track_tracked"),
                frame_ids=sampled_frame_ids,
                draw_boxes=False,
                draw_labels=False,
            )
        self._save_track_result(
            self.track_dir,
            {
                "sampled_frame_ids": sampled_frame_ids.tolist(),
                "detections_dir": tracked_detections_dir,
                "tracklet_ids": tracklet_ids,
            },
        )
        shutil.rmtree(runtime_dir, ignore_errors=True)
