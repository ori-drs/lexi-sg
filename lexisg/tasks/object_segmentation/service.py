from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Dict, List

import numpy as np
import open3d as o3d
from PIL import Image

from lexisg.tasks.object_segmentation import artifact_io
from lexisg.tasks.object_segmentation.clip_stage import ClipStage
from lexisg.tasks.object_segmentation.detect_stage import DetectStage
from lexisg.tasks.object_segmentation.eval_stage import EvalStage
from lexisg.tasks.object_segmentation.merge_stage import MergeStage
from lexisg.tasks.object_segmentation.track_stage import TrackStage

logger = logging.getLogger(__name__)


@dataclass
class ObjectSegmentationResult:
    sampled_frame_ids: List[int] = field(default_factory=list)
    masks: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    obj_pcds_world: Dict[int, o3d.geometry.PointCloud] = field(default_factory=dict)
    merge_map: Dict[int, int] = field(default_factory=dict)
    best_fid_per_obj: Dict[int, int] = field(default_factory=dict)
    best_fids_per_obj: Dict[int, List[int]] = field(default_factory=dict)
    obj_clip_feats: Dict[int, np.ndarray] = field(default_factory=dict)
    obj_clip_crops: Dict[int, List] = field(default_factory=dict)
    eval_dir: str | None = None


@dataclass
class ObjectSegmentationTrackResult:
    sampled_frame_ids: List[int] = field(default_factory=list)
    detections_dir: str | None = None
    tracklet_ids: List[int] = field(default_factory=list)
    annotated_rgb_dir: str | None = None
    raw_rgb_dir: str | None = None


class ObjectSegmentationService:
    """Reusable object segmentation orchestration for scene- or room-level callers."""

    def __init__(
        self,
        config,
        scene_name: str,
        artifact_root: str,
        color_seed: int,
        dataset: str | None = None,
        data_path: str | None = None,
        feature_model=None,
        feature_encoder: str = "clip",
    ):
        self.config = config
        self.feature_model = feature_model
        self.feature_encoder = str(feature_encoder)
        self.scene_name = str(scene_name)
        self.artifact_root = str(artifact_root)
        self.color_seed = int(color_seed)
        self.dataset = None if dataset is None else str(dataset)
        self.data_path = None if data_path is None else str(data_path)
        self.log_path = self._configure_logging()

    def _configure_logging(self):
        os.makedirs(self.artifact_root, exist_ok=True)
        log_path = os.path.join(self.artifact_root, "object_segmentation.log")
        logger_name = f"{__name__}.{self.scene_name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True
        file_handler_exists = any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(getattr(handler, "baseFilename", "")) == os.path.abspath(log_path)
            for handler in self.logger.handlers
        )
        if not file_handler_exists:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.info("Object segmentation file logging enabled at %s", log_path)
        return log_path

    def _build_stages(self):
        return {
            "detect": self._build_detect_stage(),
            "track": self._build_track_stage(),
            "merge": MergeStage(
                self.config,
                self.scene_name,
                self.color_seed,
                self.artifact_root,
                self.artifact_root,
            ),
            "clip": ClipStage(
                self.config,
                self.scene_name,
                self.color_seed,
                self.artifact_root,
                self.artifact_root,
                feature_model=self.feature_model,
                feature_encoder=self.feature_encoder,
            ),
            "eval": None
            if self.dataset is None or self.data_path is None
            else EvalStage(
                self.config,
                self.dataset,
                self.scene_name,
                self.data_path,
                self.artifact_root,
                self.artifact_root,
            ),
        }

    def _build_detect_stage(self):
        return DetectStage(self.config, self.scene_name, self.artifact_root)

    def _build_track_stage(self):
        return TrackStage(self.config, self.artifact_root, self.artifact_root)

    def _materialize_prediction_images(self, predictions: Dict):
        images = predictions.get("images")
        if images is None:
            raise ValueError("run_from_predictions requires predictions['images'] to be present.")
        image_dir = os.path.join(self.artifact_root, "model_space_rgb")
        os.makedirs(image_dir, exist_ok=True)
        rgb_paths = []
        for frame_idx, image in enumerate(images):
            save_path = os.path.join(image_dir, f"frame_{int(frame_idx):06d}.png")
            Image.fromarray(np.asarray(image, dtype=np.uint8)).save(save_path)
            rgb_paths.append(save_path)
        return rgb_paths

    def run_from_context(
        self,
        ctx: Dict,
        plan: Dict,
        include_clip: bool = True,
        include_eval: bool = False,
    ) -> ObjectSegmentationResult:
        self.logger.info(
            "Starting object segmentation from context: scene=%s artifact_root=%s include_clip=%s include_eval=%s sampled_frames=%d",
            self.scene_name,
            self.artifact_root,
            include_clip,
            include_eval,
            len(plan.get("sampled_frame_ids", [])),
        )
        stages = self._build_stages()
        self.logger.info("Running detect stage")
        stages["detect"].run(ctx, plan)
        self.logger.info("Running track stage")
        stages["track"].run(ctx, plan)
        self.logger.info("Running merge stage")
        merge_result = stages["merge"].run(ctx, plan)

        result = ObjectSegmentationResult(
            sampled_frame_ids=[int(fid) for fid in plan.get("sampled_frame_ids", [])],
            masks=dict(merge_result.get("masks", {})),
            obj_pcds_world=dict(merge_result.get("obj_pcds_world", {})),
            merge_map=dict(merge_result.get("merge_map", {})),
        )

        if include_clip:
            self.logger.info("Running clip stage")
            clip_result = stages["clip"].run(ctx) or {}
            result.masks = dict(clip_result.get("masks", result.masks))
            result.obj_pcds_world = dict(clip_result.get("obj_pcds_world", result.obj_pcds_world))
            result.merge_map = dict(clip_result.get("merge_map", result.merge_map))
            result.best_fid_per_obj = dict(clip_result.get("best_fid_per_obj", {}))
            result.best_fids_per_obj = dict(clip_result.get("best_fids_per_obj", {}))
            result.obj_clip_feats = dict(clip_result.get("obj_clip_feats", {}))
            result.obj_clip_crops = dict(clip_result.get("obj_clip_crops", {}))

        if include_eval and stages["eval"] is not None:
            self.logger.info("Running eval stage")
            eval_result = stages["eval"].run() or {}
            result.eval_dir = eval_result.get("eval_dir")

        self.logger.info(
            "Completed object segmentation from context: sampled_frames=%d objects=%d merged_objects=%d eval_dir=%s",
            len(result.sampled_frame_ids),
            len(result.obj_pcds_world),
            len(result.merge_map),
            result.eval_dir,
        )

        return result

    def run_detect_and_track_from_context(
        self,
        ctx: Dict,
        plan: Dict,
    ) -> ObjectSegmentationTrackResult:
        sampled_frame_ids = [int(fid) for fid in plan.get("sampled_frame_ids", [])]
        self.logger.info(
            "Starting object segmentation detect+track from context: scene=%s artifact_root=%s sampled_frames=%d",
            self.scene_name,
            self.artifact_root,
            len(sampled_frame_ids),
        )
        self.logger.info("Running detect stage")
        self._build_detect_stage().run(ctx, plan)
        self.logger.info("Running track stage")
        self._build_track_stage().run(ctx, plan)
        track_payload = artifact_io.load_track_result(
            os.path.join(self.artifact_root, "track_result.pkl.gz")
        )
        result = ObjectSegmentationTrackResult(
            sampled_frame_ids=[int(fid) for fid in track_payload.get("sampled_frame_ids", [])],
            detections_dir=str(track_payload.get("detections_dir", "")),
            tracklet_ids=[int(obj_id) for obj_id in track_payload.get("tracklet_ids", [])],
            annotated_rgb_dir=os.path.join(
                self.artifact_root,
                "visualizations",
                "track_tracked",
            ),
            raw_rgb_dir=os.path.join(
                self.artifact_root,
                "visualizations",
                "track_tracked",
                "raw_rgb",
            ),
        )
        self.logger.info(
            "Completed object segmentation detect+track from context: sampled_frames=%d tracklets=%d annotated_rgb_dir=%s",
            len(result.sampled_frame_ids),
            len(result.tracklet_ids),
            result.annotated_rgb_dir,
        )
        return result

    def run_from_predictions(
        self,
        predictions: Dict,
        include_clip: bool = True,
        include_eval: bool = False,
    ) -> ObjectSegmentationResult:
        rgb_paths = self._materialize_prediction_images(predictions)
        sampled_frame_ids = np.arange(len(rgb_paths), dtype=np.int32)
        self.logger.info(
            "Starting object segmentation from predictions: scene=%s artifact_root=%s include_clip=%s include_eval=%s frames=%d",
            self.scene_name,
            self.artifact_root,
            include_clip,
            include_eval,
            len(rgb_paths),
        )
        ctx = {
            "rgb_paths": list(rgb_paths),
        }
        plan = {
            "frame_ids": sampled_frame_ids.copy(),
            "sampled_frame_ids": sampled_frame_ids.copy(),
            "sampled_frame_ids_local": sampled_frame_ids.copy(),
            "frame_sampling_mode": "explicit",
        }

        stages = self._build_stages()
        self.logger.info("Running detect stage")
        stages["detect"].run(ctx, plan)
        self.logger.info("Running track stage")
        stages["track"].run(ctx, plan)
        self.logger.info("Running merge stage with predictions")
        merge_result = stages["merge"].run_with_predictions(
            predictions=predictions,
            sampled_frame_ids=sampled_frame_ids,
        )

        result = ObjectSegmentationResult(
            sampled_frame_ids=[int(fid) for fid in sampled_frame_ids.tolist()],
            masks=dict(merge_result.get("masks", {})),
            obj_pcds_world=dict(merge_result.get("obj_pcds_world", {})),
            merge_map=dict(merge_result.get("merge_map", {})),
        )

        if include_clip:
            self.logger.info("Running clip stage")
            clip_result = stages["clip"].run(ctx) or {}
            result.masks = dict(clip_result.get("masks", result.masks))
            result.obj_pcds_world = dict(clip_result.get("obj_pcds_world", result.obj_pcds_world))
            result.merge_map = dict(clip_result.get("merge_map", result.merge_map))
            result.best_fid_per_obj = dict(clip_result.get("best_fid_per_obj", {}))
            result.best_fids_per_obj = dict(clip_result.get("best_fids_per_obj", {}))
            result.obj_clip_feats = dict(clip_result.get("obj_clip_feats", {}))
            result.obj_clip_crops = dict(clip_result.get("obj_clip_crops", {}))

        if include_eval and stages["eval"] is not None:
            self.logger.info("Running eval stage")
            eval_result = stages["eval"].run() or {}
            result.eval_dir = eval_result.get("eval_dir")

        self.logger.info(
            "Completed object segmentation from predictions: frames=%d objects=%d merged_objects=%d eval_dir=%s",
            len(result.sampled_frame_ids),
            len(result.obj_pcds_world),
            len(result.merge_map),
            result.eval_dir,
        )

        return result
