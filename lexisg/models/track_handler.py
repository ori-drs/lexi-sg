import gc
from contextlib import nullcontext

import numpy as np
import torch
from loguru import logger

from lexisg.utils.config import get_object_segmentation_stage_config
from lexisg.models.detect_handler import SAMDetector
import lexisg.utils.geometry as geometry


class SAMTracker:
    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device()
        self.tracking_state = None
        voxel_consistency_cfg = self._tracking_cfg().get("voxel_consistency", {})
        self.voxel_consistency_enabled = bool(voxel_consistency_cfg.get("enabled", False))
        self.voxel_consistency_min_voxel_intersection = int(
            voxel_consistency_cfg.get("min_voxel_intersection", 1)
        )
        logger.info(f"Initializing SAMTracker on device={self.device}")
        self.video_predictor = self._load_video_predictor(method=self._get_tracking_method())
        logger.info(f"SAMTracker ready: method={self._get_tracking_method()} model={self._get_tracking_model_name()}")

    @staticmethod
    def _resolve_device():
        if torch.cuda.is_available():
            try:
                _ = torch.cuda.device_count()
                _ = torch.zeros(1, device="cuda")
                return "cuda"
            except Exception as exc:
                logger.warning(f"CUDA unavailable at runtime; falling back to CPU: {exc}")
        return "cpu"

    @staticmethod
    def _prepare_sam2_hydra():
        SAMDetector._prepare_sam2_hydra()

    def _tracking_cfg(self):
        return get_object_segmentation_stage_config(self.config, "track")

    def _video_tracking_cfg(self):
        track_cfg = self._tracking_cfg()
        return track_cfg.get("tracking", track_cfg)

    def _get_tracking_method(self):
        track_cfg = self._video_tracking_cfg()
        if "method" in track_cfg:
            return track_cfg["method"]
        return self._tracking_cfg().get("method", "sam2")

    def _get_tracking_model_name(self):
        track_cfg = self._video_tracking_cfg()
        if "model" in track_cfg:
            return track_cfg["model"]
        return self._tracking_cfg()["tracking_model"]

    def _sam2_tracking_cfg(self):
        return self._video_tracking_cfg().get("sam2", {})

    def _load_video_predictor(self, method: str):
        if "edgetam" in method:
            return self._load_video_predictor_edgetam()
        if "efficienttam" in method:
            return self._load_video_predictor_efficienttam()
        if "sam2" in method:
            return self._load_video_predictor_sam()
        raise ValueError(f"Unsupported tracking model: {method}")

    def _load_video_predictor_sam(self):
        self._prepare_sam2_hydra()
        candidates = ["sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large"]
        tracking_model = self._get_tracking_model_name()
        assert tracking_model in candidates, f"Unsupported SAM2 model: {tracking_model}"
        print(f"Using SAM2 tracking model: {tracking_model}")
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        sam2_cfg = self._sam2_tracking_cfg()
        return SAM2VideoPredictor.from_pretrained(
            f"facebook/{tracking_model}",
            device=self.device,
            apply_postprocessing=sam2_cfg.get("apply_postprocessing", True),
            vos_optimized=sam2_cfg.get("vos_optimized", False),
        ).to(self.device)

    def _load_video_predictor_edgetam(self):
        from sam2.build_sam import build_sam2_video_predictor

        model_cfg = "configs/edgetam.yaml"
        checkpoint = "../thirdparty/EdgeTAM/checkpoints/edgetam.pt"
        return build_sam2_video_predictor(model_cfg, checkpoint, device=self.device)

    def _load_video_predictor_efficienttam(self):
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

        candidates = [
            "efficienttam_s_1", "efficienttam_s_2", "efficienttam_s", "efficienttam_s_512x512",
            "efficienttam_ti_1", "efficienttam_ti_2", "efficienttam_ti", "efficienttam_ti_512x512",
        ]
        tracking_model = self._get_tracking_model_name()
        assert tracking_model in candidates, f"Unsupported EfficientTAM model: {tracking_model}"
        model_cfg = f"configs/efficienttam/{tracking_model}.yaml"
        checkpoint = f"../thirdparty/EfficientTAM/checkpoints/{tracking_model}.pt"
        return build_efficienttam_video_predictor(model_cfg, ckpt_path=checkpoint, device=self.device)

    def init_tracking(self, rgb_paths: str):
        logger.info(f"Initializing tracker state from image dir: {rgb_paths}")
        sam2_cfg = self._sam2_tracking_cfg()
        self.tracking_state = self.video_predictor.init_state(
            rgb_paths,
            offload_state_to_cpu=bool(sam2_cfg.get("offload_state_to_cpu", True)),
            offload_video_to_cpu=bool(sam2_cfg.get("offload_video_to_cpu", True)),
        )
        logger.info("Tracker state initialization finished.")
        return self.tracking_state

    def get_ious(self, obj_id: int, frame_idx: int):
        if self.tracking_state is None:
            raise ValueError("Tracking state is not initialized.")
        obj_idx = self.tracking_state["obj_id_to_idx"][obj_id]
        out = self.tracking_state["output_dict_per_obj"][obj_idx]
        key = "cond_frame_outputs" if frame_idx in out["cond_frame_outputs"] else "non_cond_frame_outputs"
        return out[key][frame_idx]["ious"].to(torch.float32).detach().cpu().numpy().squeeze()

    def _autocast_context(self):
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def propagate_one_object(
        self,
        obj_id: int,
        start_fid: int,
        mask_bool: np.ndarray,
        conf_thres: float,
        patience: int,
        max_track_count: int = 10,
        shared_geometry=None,
    ):
        self.video_predictor.reset_state(self.tracking_state)
        dets_per_fid = {}
        seed_voxel_ids = None
        if self.voxel_consistency_enabled and shared_geometry is not None:
            seed_voxel_ids = set(
                geometry.masked_point_indices_from_lookup(
                    shared_geometry,
                    int(start_fid),
                    np.asarray(mask_bool).astype(bool),
                ).tolist()
            )
        with torch.inference_mode(), self._autocast_context():
            from supervision import Detections
            from lexisg.utils.sv_compat import mask_to_xyxy

            mask_tensor = torch.tensor(mask_bool, dtype=torch.bool)
            _, obj_ids, _ = self.video_predictor.add_new_mask(self.tracking_state, start_fid, obj_id, mask_tensor)
            for reverse in [False, True]:
                no_mask = 0
                track_count = 0
                for fid, obj_ids, m_t in self.video_predictor.propagate_in_video(
                    self.tracking_state, start_frame_idx=start_fid, reverse=reverse
                ):
                    if track_count >= max_track_count and max_track_count > 0:
                        break
                    track_count += 1
                    mask = (m_t > 0.0).squeeze(1).cpu().numpy()
                    if mask.sum() == 0:
                        no_mask += 1
                        if no_mask >= patience:
                            break
                        continue
                    no_mask = 0
                    ious = [self.get_ious(oid, fid) for oid in obj_ids]
                    dets = Detections(
                        xyxy=mask_to_xyxy(mask),
                        mask=mask,
                        class_id=np.array(obj_ids, int),
                        confidence=np.array(ious, float),
                    )
                    if dets.confidence[0] < conf_thres:
                        no_mask += 1
                        continue
                    if seed_voxel_ids is not None:
                        curr_voxel_ids = set(
                            geometry.masked_point_indices_from_lookup(
                                shared_geometry,
                                int(fid),
                                np.asarray(mask[0]).astype(bool),
                            ).tolist()
                        )
                        if len(seed_voxel_ids & curr_voxel_ids) < self.voxel_consistency_min_voxel_intersection:
                            no_mask += 1
                            logger.debug(
                                f"Rejected propagated mask for obj={int(obj_id)} fid={int(fid)} due to "
                                f"shared-voxel inconsistency: intersection={len(seed_voxel_ids & curr_voxel_ids)} "
                                f"min_required={self.voxel_consistency_min_voxel_intersection}"
                            )
                            if no_mask >= patience:
                                break
                            continue
                    dets_per_fid[fid] = dets
        return dict(sorted(dets_per_fid.items()))

    def cleanup_after_run(self):
        try:
            if self.tracking_state is not None and self.video_predictor is not None:
                try:
                    self.video_predictor.reset_state(self.tracking_state)
                except Exception:
                    pass
                self.tracking_state = None
        finally:
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass


class SAM:
    """Compatibility facade for older call sites that expect one combined handler."""

    def __init__(self, config, load_tracking_model: bool = True):
        self.detector = SAMDetector(config)
        self.tracker = SAMTracker(config) if load_tracking_model else None

    def predict(self, image, cache_key: str = None):
        return self.detector.predict(image, cache_key=cache_key)

    def init_tracking(self, rgb_paths: str):
        if self.tracker is None:
            self.tracker = SAMTracker(self.detector.config)
        return self.tracker.init_tracking(rgb_paths)

    def get_ious(self, obj_id: int, frame_idx: int):
        if self.tracker is None:
            raise ValueError("Tracking model is not loaded.")
        return self.tracker.get_ious(obj_id, frame_idx)

    def propagate_one_object(self, *args, **kwargs):
        if self.tracker is None:
            self.tracker = SAMTracker(self.detector.config)
        return self.tracker.propagate_one_object(*args, **kwargs)

    def cleanup_after_run(self):
        self.detector.cleanup_after_run()
        if self.tracker is not None:
            self.tracker.cleanup_after_run()
