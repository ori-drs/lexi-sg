import gc

import numpy as np
import torch
from loguru import logger
from PIL import Image
from supervision import Detections

from lexisg.utils.config import get_object_segmentation_stage_config
from lexisg.models.grounded_sam_wrapper import GroundedSAMPipeline
from lexisg.utils.detection import filter_overlapping_detections, resize_detections
from lexisg.utils.sv_compat import xywh_to_xyxy


class SAMDetector:
    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device()
        logger.info(f"Initializing SAMDetector on device={self.device}")
        self.mask_generator = self._load_mask_generator(method=self._get_detection_method())
        logger.info(f"SAMDetector ready: method={self._get_detection_method()} model={self._get_detection_model_name()}")

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
        try:
            from hydra import initialize_config_module
            from hydra.core.global_hydra import GlobalHydra

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            initialize_config_module("sam2", version_base="1.2")
        except Exception:
            pass

    def _detection_cfg(self):
        return get_object_segmentation_stage_config(self.config, "detect")

    def _tracking_cfg(self):
        return get_object_segmentation_stage_config(self.config, "track")

    def _amg_cfg(self):
        return self._detection_cfg().get("amg", {})

    def _post_filter_cfg(self):
        return self._detection_cfg().get("post_filter", {})

    def _get_detection_method(self):
        det_cfg = self._detection_cfg()
        if "method" in det_cfg:
            return det_cfg["method"]
        return self._tracking_cfg().get("method", "amg")

    def _get_detection_model_name(self):
        det_cfg = self._detection_cfg()
        if "model" in det_cfg:
            return det_cfg["model"]
        return self._tracking_cfg().get("detection_model", self._get_detection_method())

    def _load_mask_generator(self, method: str):
        if "grounded_sam" in method or method == "gsa":
            return self._load_mask_generator_grounded_sam()
        if method == "amg":
            return self._load_mask_generator_amg()
        raise ValueError(f"Unsupported detection model: {method}")

    def _load_mask_generator_grounded_sam(self):
        det_cfg = self._detection_cfg()
        gsa_cfg = det_cfg.get("grounded_sam", {})
        return GroundedSAMPipeline(
            device=self.device,
            enable_ram=gsa_cfg.get("enable_ram", True),
            gsa_repo_root=gsa_cfg.get("gsa_path"),
            ram_repo_root=gsa_cfg.get("ram_repo_root"),
            ram_checkpoint_path=gsa_cfg.get("ram_checkpoint_path"),
            grounding_dino_config_path=gsa_cfg.get("grounding_dino_config_path"),
            grounding_dino_checkpoint_path=gsa_cfg.get("grounding_dino_checkpoint_path"),
            sam2_model_id=gsa_cfg.get("sam2_model_id", "facebook/sam2-hiera-large"),
            sam2_config_path=gsa_cfg.get("sam2_config_path"),
            sam2_checkpoint_path=gsa_cfg.get("sam2_checkpoint_path"),
            sam2_apply_postprocessing=gsa_cfg.get("sam2_apply_postprocessing", True),
            sam2_iou_threshold=gsa_cfg.get("sam2_iou_threshold", None),
            confidence_threshold=gsa_cfg.get("confidence_threshold", None),
            box_threshold=gsa_cfg.get("box_threshold", 0.25),
            text_threshold=gsa_cfg.get("text_threshold", 0.25),
            nms_threshold=gsa_cfg.get("nms_threshold", 0.5),
            max_box_area_ratio=gsa_cfg.get("max_box_area_ratio", 0.8),
            add_background_classes=gsa_cfg.get("add_background_classes", False),
            add_classes=gsa_cfg.get("add_classes"),
            remove_substrings=gsa_cfg.get("remove_substrings"),
        )

    def _load_mask_generator_amg(self):
        self._prepare_sam2_hydra()
        candidates = ["sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large"]
        detection_model = self._get_detection_model_name()
        assert detection_model in candidates, f"Unsupported SAM2 model: {detection_model}"
        print(f"Using SAM2 detection model: {detection_model}")
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        amg_cfg = self._amg_cfg()
        pred_iou_thresh = amg_cfg.get(
            "pred_iou_thresh",
            self._tracking_cfg().get("conf_threshold", 0.8),
        )
        amg_kwargs = {
            "points_per_side": amg_cfg.get("points_per_side", 12),
            "points_per_batch": amg_cfg.get("points_per_batch", 64),
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": amg_cfg.get("stability_score_thresh", 0.95),
            "stability_score_offset": amg_cfg.get("stability_score_offset", 1.0),
            "mask_threshold": amg_cfg.get("mask_threshold", 0.0),
            "box_nms_thresh": amg_cfg.get("box_nms_thresh", 0.7),
            "crop_n_layers": amg_cfg.get("crop_n_layers", 0),
            "crop_nms_thresh": amg_cfg.get("crop_nms_thresh", 0.7),
            "crop_overlap_ratio": amg_cfg.get("crop_overlap_ratio", 512 / 1500),
            "crop_n_points_downscale_factor": amg_cfg.get("crop_n_points_downscale_factor", 1),
            "point_grids": amg_cfg.get("point_grids", None),
            "min_mask_region_area": amg_cfg.get("min_mask_region_area", 0),
            "output_mode": amg_cfg.get("output_mode", "binary_mask"),
            "use_m2m": amg_cfg.get("use_m2m", False),
            "multimask_output": amg_cfg.get("multimask_output", True),
        }
        sam2_kwargs = {
            "device": self.device,
            "apply_postprocessing": amg_cfg.get("apply_postprocessing", True),
        }
        return SAM2AutomaticMaskGenerator.from_pretrained(
            f"facebook/{detection_model}",
            **sam2_kwargs,
            **amg_kwargs,
        )

    def _cleanup_mask_generator_only(self):
        try:
            if hasattr(self.mask_generator, "cleanup"):
                try:
                    self.mask_generator.cleanup()
                except Exception:
                    pass
            predictor = getattr(self.mask_generator, "predictor", None)
            if predictor is not None and hasattr(predictor, "reset_predictor"):
                try:
                    predictor.reset_predictor()
                except Exception:
                    pass
        finally:
            gc.collect()

    def _clear_cuda_cache(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _reload_mask_generator_on_device(self, device: str):
        self._cleanup_mask_generator_only()
        self.device = str(device)
        self.mask_generator = self._load_mask_generator(method=self._get_detection_method())
        logger.info(f"Reloaded detector on device={self.device}")

    def dets_from_sam(self, sam_result) -> Detections:
        masks_sorted = sorted(sam_result, key=lambda x: x["area"], reverse=True)
        if not masks_sorted:
            return Detections.empty()
        xywh = np.array([m["bbox"] for m in masks_sorted])
        mask = np.array([m["segmentation"] for m in masks_sorted])
        conf = np.array([m["predicted_iou"] for m in masks_sorted])
        class_id = np.arange(len(masks_sorted), dtype=int)
        return Detections(xyxy=xywh_to_xyxy(xywh), mask=mask, confidence=conf, class_id=class_id)

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _resize_image_to_max_side(image: np.ndarray, max_side: int):
        max_side = int(max_side)
        if max_side <= 0:
            return image, False
        h, w = image.shape[:2]
        curr_max = max(h, w)
        if curr_max <= max_side:
            return image, False

        scale = float(max_side) / float(curr_max)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = np.ascontiguousarray(np.array(Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR)))
        return resized, True

    def _predict_raw(self, image) -> Detections:
        method = self._get_detection_method()
        with torch.inference_mode():
            if "grounded_sam" in method or method == "gsa":
                return self.mask_generator.predict(image)
            raw = self.mask_generator.generate(image)
            return self.dets_from_sam(raw)

    def _predict_with_optional_resize(self, image, max_side: int) -> Detections:
        h_orig, w_orig = image.shape[:2]
        resized_image, resized = self._resize_image_to_max_side(image, int(max_side))
        dets = self._predict_raw(resized_image)
        if resized:
            dets = resize_detections(dets, h_orig, w_orig)
        return dets

    def _apply_detection_post_filter(self, dets: Detections, cache_key: str | None = None) -> Detections:
        if dets is None or len(dets) == 0 or getattr(dets, "mask", None) is None:
            return dets
        post_cfg = self._post_filter_cfg()
        ios_threshold = float(post_cfg.get("mask_ios_threshold", 0.8))
        if ios_threshold < 0.0:
            return dets
        det_count_before = int(len(dets))
        filtered = filter_overlapping_detections(
            dets,
            iou_threshold=0.0,
            inner_overlap_threshold=ios_threshold,
        )
        removed = det_count_before - int(len(filtered))
        if removed > 0:
            logger.info(
                "Detection post-filter removed {} overlapping masks by IoS "
                "(threshold={:.3f}, cache_key={})",
                removed,
                ios_threshold,
                cache_key if cache_key is not None else "n/a",
            )
        return filtered

    def predict(self, image, cache_key: str | None = None) -> Detections:
        amg_cfg = self._amg_cfg()
        det_method = self._get_detection_method()
        gsa_cfg = self._detection_cfg().get("grounded_sam", {})
        max_side = int(amg_cfg.get("max_side", 0))
        oom_retry_max_side = int(amg_cfg.get("oom_retry_max_side", 1280))
        try:
            dets = self._predict_with_optional_resize(image, max_side=max_side)
        except Exception as exc:
            if self.device != "cuda" or not self._is_cuda_oom(exc):
                raise
            retry_side = int(oom_retry_max_side)
            if retry_side <= 0:
                raise
            logger.warning(
                f"CUDA OOM during detection; retrying with resized input "
                f"(cache_key={cache_key}, max_side={retry_side}): {exc}"
            )
            gc.collect()
            self._clear_cuda_cache()
            try:
                dets = self._predict_with_optional_resize(image, max_side=retry_side)
            except Exception as retry_exc:
                use_cpu_fallback = (
                    ("grounded_sam" in det_method or det_method == "gsa")
                    and bool(gsa_cfg.get("cpu_fallback_on_oom", True))
                )
                if not use_cpu_fallback or not self._is_cuda_oom(retry_exc):
                    raise
                cpu_fallback_max_side = int(gsa_cfg.get("cpu_fallback_max_side", retry_side))
                logger.warning(
                    f"CUDA OOM persisted for grounded_sam; reloading on CPU "
                    f"(cache_key={cache_key}, max_side={cpu_fallback_max_side}): {retry_exc}"
                )
                self._reload_mask_generator_on_device("cpu")
                dets = self._predict_with_optional_resize(image, max_side=cpu_fallback_max_side)

        dets = self._apply_detection_post_filter(dets, cache_key=cache_key)
        return dets

    def cleanup_after_run(self):
        try:
            self._cleanup_mask_generator_only()
        finally:
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                self._clear_cuda_cache()
