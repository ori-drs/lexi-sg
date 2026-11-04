from __future__ import annotations

import os
from typing import Dict

import numpy as np
from loguru import logger

from lexisg.utils.config import get_object_segmentation_stage_config
from lexisg.models.detect_handler import SAMDetector
from lexisg.tasks.object_segmentation.common import (
    load_images_for_frame_ids,
)
from lexisg.tasks.object_segmentation import artifact_io
from lexisg.utils.visualization import save_detection_overlay_frame


class DetectStage:
    def __init__(self, config, scene_name: str, stage_dir: str):
        self.config = config
        self.scene_name = str(scene_name)
        self.stage_dir = str(stage_dir)

    @staticmethod
    def _candidate_record(fid: int, det_idx: int, dets):
        conf = float(dets.confidence[det_idx]) if dets.confidence is not None else 0.0
        area = int(np.asarray(dets.mask[det_idx]).sum()) if dets.mask is not None else 0
        return {
            "fid": int(fid),
            "det_idx": int(det_idx),
            "confidence": float(conf),
            "area": area,
            "score": float(conf),
        }

    def run(self, ctx: Dict, plan: Dict):
        detector = SAMDetector(self.config)
        try:
            stage_cfg = get_object_segmentation_stage_config(self.config, "detect")
            save_visualizations = bool(stage_cfg.get("save_visualizations", True))
            sampled_frame_ids = np.asarray(plan["sampled_frame_ids"], dtype=np.int32)
            images = load_images_for_frame_ids(ctx["rgb_paths"], sampled_frame_ids)
            detections_dir = os.path.join(self.stage_dir, "detections")
            os.makedirs(detections_dir, exist_ok=True)

            candidates = []
            for fid_local, image in enumerate(images):
                fid_global = int(sampled_frame_ids[fid_local])
                cache_key = f"{self.scene_name}_frame{fid_global:06d}"
                dets = detector.predict(image, cache_key=cache_key)
                artifact_io.save_pickle_gz(
                    os.path.join(detections_dir, f"frame_{int(fid_local):06d}.pkl.gz"),
                    dets,
                )
                if save_visualizations:
                    save_detection_overlay_frame(
                        image,
                        dets,
                        save_dir=os.path.join(self.stage_dir, "visualizations", "detect_raw"),
                        frame_index=int(fid_global),
                    )
                if dets is None or len(dets) == 0 or dets.mask is None:
                    continue
                for det_idx in range(len(dets)):
                    candidates.append(self._candidate_record(int(fid_local), int(det_idx), dets))

            candidates.sort(key=lambda item: item["score"], reverse=True)
            artifact_io.save_pickle_gz(
                os.path.join(self.stage_dir, "detect_candidates.pkl.gz"),
                {"version": 3, "mode": "selected_frames_only", "candidates": candidates},
            )
            logger.info(
                f"Detect stage saved frames={len(sampled_frame_ids)} candidates={len(candidates)}"
            )
        finally:
            detector.cleanup_after_run()
