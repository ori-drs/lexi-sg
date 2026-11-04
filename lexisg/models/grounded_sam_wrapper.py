from __future__ import annotations

import gc
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
import torchvision.transforms as T
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from loguru import logger

from lexisg.utils.sv_compat import mask_to_xyxy


DEFAULT_RAM_ADD_CLASSES = ["other item"]
DEFAULT_RAM_REQUIRED_CLASSES = ["object"]
DEFAULT_RAM_REMOVE_SUBSTRINGS = [
    "room",
    "kitchen",
    "office",
    "house",
    "home",
    "building",
    "corner",
    "shadow",
    "carpet",
    "photo",
    "shade",
    "stall",
    "space",
    "aquarium",
    "apartment",
    "image",
    "city",
    "blue",
    "skylight",
    "hallway",
    "bureau",
    "modern",
    "salon",
    "doorway",
    "wall lamp",
    "wood floor",
]
BACKGROUND_CLASSES = ["wall", "floor", "ceiling"]


def _resolve_optional_path(explicit_path: str | os.PathLike[str] | None, env_var: str) -> str | None:
    if explicit_path is not None:
        return str(Path(explicit_path).expanduser())
    value = os.environ.get(env_var)
    if value:
        return str(Path(value).expanduser())
    return None


# lexisg/models/grounded_sam_wrapper.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate_repo_roots(explicit_repo_root: str | os.PathLike[str] | None) -> list[Path]:
    candidates: list[Path] = []
    env_repo = os.environ.get("GSA_PATH")
    if explicit_repo_root:
        candidates.append(Path(explicit_repo_root).expanduser())
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    # Default location for this repo's own layout.
    candidates.append(_REPO_ROOT / "thirdparty" / "Grounded-Segment-Anything")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _candidate_ram_roots(explicit_repo_root: str | os.PathLike[str] | None) -> list[Path]:
    candidates: list[Path] = []
    env_repo = os.environ.get("RAM_REPO_ROOT")
    if explicit_repo_root:
        candidates.append(Path(explicit_repo_root).expanduser())
    if env_repo:
        candidates.append(Path(env_repo).expanduser())
    candidates.append(_REPO_ROOT / "thirdparty" / "recognize-anything")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_required_path(
    explicit_path: str | os.PathLike[str] | None,
    env_var: str,
    candidates: Iterable[Path],
) -> str:
    resolved = _resolve_optional_path(explicit_path, env_var)
    if resolved:
        return resolved

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise ValueError(f"Missing required path. Provide it explicitly or set {env_var}.")


def _normalize_class_list(classes: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in classes:
        value = item.strip()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _classes_from_ram_tags(
    tag_string: str,
    add_classes: Iterable[str] | None = None,
    remove_substrings: Iterable[str] | None = None,
    add_background_classes: bool = False,
) -> list[str]:
    classes = _normalize_class_list(tag_string.replace(" |", ",").split(","))

    for item in add_classes or DEFAULT_RAM_ADD_CLASSES:
        if item not in classes:
            classes.append(item)
    for item in DEFAULT_RAM_REQUIRED_CLASSES:
        if item not in classes:
            classes.append(item)

    blocked = list(remove_substrings or DEFAULT_RAM_REMOVE_SUBSTRINGS)
    if add_background_classes:
        for item in BACKGROUND_CLASSES:
            if item not in classes:
                classes.append(item)
    else:
        blocked.extend(BACKGROUND_CLASSES)

    filtered = [
        cls for cls in classes if not any(blocked_item in cls.lower() for blocked_item in blocked)
    ]
    return _normalize_class_list(filtered)


def _rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _bootstrap_repo_paths(gsa_repo_root: str | os.PathLike[str] | None, ram_repo_root: str | os.PathLike[str] | None) -> None:
    for repo_root in _candidate_repo_roots(gsa_repo_root):
        groundingdino_root = repo_root / "GroundingDINO"
        if groundingdino_root.exists():
            path_str = str(groundingdino_root)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    for repo_root in _candidate_ram_roots(ram_repo_root):
        if repo_root.exists():
            path_str = str(repo_root)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _prepare_sam2_hydra() -> None:
    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")
    except Exception:
        pass


def _normalize_sam2_model_id(model_id: str | None) -> str:
    value = str(model_id or "facebook/sam2-hiera-large").strip()
    if "/" not in value:
        value = f"facebook/{value}"
    return value


def _empty_detections(height: int = 0, width: int = 0) -> sv.Detections:
    mask = np.zeros((0, int(height), int(width)), dtype=bool)
    empty_int = np.empty((0,), dtype=np.int32)
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        mask=mask,
        confidence=np.empty((0,), dtype=np.float32),
        class_id=empty_int.copy(),
    )


def _slice_detections(detections: sv.Detections, keep: np.ndarray) -> sv.Detections:
    keep = np.asarray(keep, dtype=bool)
    if detections is None or len(detections) == 0 or keep.size == 0 or not np.any(keep):
        if detections is not None and getattr(detections, "mask", None) is not None and detections.mask.ndim == 3:
            h, w = detections.mask.shape[1:]
            return _empty_detections(h, w)
        return _empty_detections()

    xyxy = np.asarray(detections.xyxy)[keep].astype(np.float32)
    mask = None if getattr(detections, "mask", None) is None else np.asarray(detections.mask)[keep].astype(bool)
    confidence = (
        None
        if getattr(detections, "confidence", None) is None
        else np.asarray(detections.confidence)[keep].astype(np.float32)
    )
    class_id = (
        None
        if getattr(detections, "class_id", None) is None
        else np.asarray(detections.class_id)[keep].astype(np.int32)
    )
    data = {}
    for key, value in getattr(detections, "data", {}).items():
        arr = np.asarray(value)
        try:
            data[key] = arr[keep]
        except Exception:
            data[key] = value

    sliced = sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        data=data,
    )
    tracker_id = getattr(detections, "tracker_id", None)
    if tracker_id is not None:
        sliced.tracker_id = np.asarray(tracker_id)[keep]
    return sliced


@dataclass
class RAMOutput:
    tags_english: str
    tags_chinese: str


class RAMTagger:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        image_size: int = 384,
        gsa_repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.device = device
        self.image_size = image_size
        checkpoint_candidates = [
            repo_root / "ram_swin_large_14m.pth"
            for repo_root in _candidate_repo_roots(gsa_repo_root)
        ] + [_REPO_ROOT / "checkpoints" / "ram_swin_large_14m.pth"]
        self.checkpoint_path = _resolve_required_path(
            checkpoint_path,
            "RAM_CHECKPOINT_PATH",
            checkpoint_candidates,
        )

        try:
            from ram.models import ram
            from ram.inference import inference_ram
        except ImportError as exc:
            raise ImportError(
                "RAM is not installed. Install recognize-anything or expose its source tree."
            ) from exc

        self._inference_ram = inference_ram
        self._transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model = ram(
            pretrained=self.checkpoint_path,
            image_size=image_size,
            vit="swin_l",
        ).eval().to(device)

    def predict(self, image_rgb: np.ndarray) -> RAMOutput:
        pil_image = torchvision.transforms.functional.to_pil_image(image_rgb)
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
        tags_english, tags_chinese = self._inference_ram(tensor, self.model)
        return RAMOutput(tags_english=tags_english, tags_chinese=tags_chinese)


class GroundingDINOModel:
    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        gsa_repo_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.requested_device = device

        repo_roots = _candidate_repo_roots(gsa_repo_root)
        config_candidates = [
            repo_root / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
            for repo_root in repo_roots
        ]
        checkpoint_candidates = [
            repo_root / "groundingdino_swint_ogc.pth"
            for repo_root in repo_roots
        ] + [_REPO_ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"]

        resolved_config = _resolve_required_path(
            config_path,
            "GROUNDING_DINO_CONFIG_PATH",
            config_candidates,
        )
        resolved_checkpoint = _resolve_required_path(
            checkpoint_path,
            "GROUNDING_DINO_CHECKPOINT_PATH",
            checkpoint_candidates,
        )

        try:
            from groundingdino.util.inference import Model
        except ImportError as exc:
            raise ImportError(
                "GroundingDINO is not available. Install it or expose its source tree via GSA_PATH."
            ) from exc

        self.device = device if self._has_custom_ops() and device.startswith("cuda") else "cpu"
        if self.device != self.requested_device:
            logger.warning(
                "GroundingDINO custom ops are unavailable in this environment; forcing CPU fallback."
            )
        self.model = Model(
            model_config_path=resolved_config,
            model_checkpoint_path=resolved_checkpoint,
            device=self.device,
        )

    @staticmethod
    def _has_custom_ops() -> bool:
        try:
            importlib.import_module("groundingdino._C")
            return True
        except ImportError:
            return False

    def detect(
        self,
        image_rgb: np.ndarray,
        classes: Iterable[str],
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        max_box_area_ratio: float | None = None,
    ) -> sv.Detections:
        classes_list = list(classes)
        detections = self.model.predict_with_classes(
            image=_rgb_to_bgr(image_rgb),
            classes=classes_list,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(detections.xyxy) == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )

        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold,
        ).cpu().numpy()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        valid_idx = detections.class_id != -1
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]

        if max_box_area_ratio is not None and 0.0 < float(max_box_area_ratio) < 1.0 and len(detections.xyxy) > 0:
            image_area = float(image_rgb.shape[0] * image_rgb.shape[1])
            if image_area > 0.0:
                xyxy = detections.xyxy.astype(np.float32)
                box_w = np.clip(xyxy[:, 2] - xyxy[:, 0], a_min=0.0, a_max=None)
                box_h = np.clip(xyxy[:, 3] - xyxy[:, 1], a_min=0.0, a_max=None)
                keep = (box_w * box_h) / image_area <= float(max_box_area_ratio)
                removed = int(np.count_nonzero(~keep))
                if removed > 0:
                    logger.info(
                        f"GroundingDINO filtered {removed} oversized boxes "
                        f"(max_box_area_ratio={float(max_box_area_ratio):.3f})"
                    )
                detections.xyxy = detections.xyxy[keep]
                detections.confidence = detections.confidence[keep]
                detections.class_id = detections.class_id[keep]
        return detections


class SAM2Segmenter:
    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        apply_postprocessing: bool = True,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        iou_threshold: float | None = None,
    ) -> None:
        _prepare_sam2_hydra()

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "SAM2 is not installed. Install it or expose its source tree."
            ) from exc

        self.device = device
        self.apply_postprocessing = bool(apply_postprocessing)
        self.iou_threshold = None if iou_threshold is None else float(iou_threshold)
        if config_path and checkpoint_path:
            sam_model = build_sam2(
                config_file=config_path,
                ckpt_path=checkpoint_path,
                device=device,
                apply_postprocessing=self.apply_postprocessing,
            )
            self.predictor = SAM2ImagePredictor(sam_model)
            self.model_id = None
        else:
            self.model_id = _normalize_sam2_model_id(model_id)
            self.predictor = SAM2ImagePredictor.from_pretrained(
                self.model_id,
                device=device,
                apply_postprocessing=self.apply_postprocessing,
            )

    def predict_masks(
        self,
        image_rgb: np.ndarray,
        xyxy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.predictor.set_image(image_rgb)
        masks: list[np.ndarray] = []
        confidences: list[float] = []
        filtered_low_iou = 0
        for box in xyxy:
            predicted_masks, scores, _ = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if self.iou_threshold is not None and self.iou_threshold >= 0.0 and best_score < self.iou_threshold:
                filtered_low_iou += 1
                continue
            best_mask = np.asarray(predicted_masks[best_idx]).astype(bool)
            if not np.any(best_mask):
                continue
            masks.append(best_mask)
            confidences.append(best_score)

        if filtered_low_iou > 0:
            logger.info(
                f"SAM2 filtered {filtered_low_iou} boxes by predicted IoU "
                f"(threshold={self.iou_threshold:.3f})"
            )

        if not masks:
            empty_mask = np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
            empty_conf = np.empty((0,), dtype=np.float32)
            return empty_mask, empty_conf
        return np.asarray(masks, dtype=bool), np.asarray(confidences, dtype=np.float32)

    def segment(self, image_rgb: np.ndarray, detections: sv.Detections) -> sv.Detections:
        if detections is None or len(detections) == 0 or getattr(detections, "xyxy", None) is None:
            return _empty_detections(image_rgb.shape[0], image_rgb.shape[1])

        masks, confidences = self.predict_masks(image_rgb=image_rgb, xyxy=detections.xyxy)
        if masks.shape[0] == 0:
            return _empty_detections(image_rgb.shape[0], image_rgb.shape[1])

        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        keep = np.where(areas > 0)[0]
        if keep.size == 0:
            return _empty_detections(image_rgb.shape[0], image_rgb.shape[1])

        masks = masks[keep]
        confidences = confidences[keep]
        order = np.argsort(areas[keep])[::-1]
        masks = masks[order]
        confidences = confidences[order]
        xyxy = mask_to_xyxy(masks)
        gd_confidence = None
        if getattr(detections, "confidence", None) is not None:
            gd_confidence = np.asarray(detections.confidence)[keep][order].astype(np.float32)
        if getattr(detections, "class_id", None) is not None:
            class_id = np.asarray(detections.class_id)[keep][order].astype(np.int32)
        else:
            class_id = np.arange(len(masks), dtype=np.int32)
        return sv.Detections(
            xyxy=xyxy.astype(np.float32),
            mask=masks.astype(bool),
            confidence=(
                gd_confidence
                if gd_confidence is not None
                else confidences.astype(np.float32)
            ),
            class_id=class_id,
            data={
                "grounding_dino_confidence": (
                    gd_confidence
                    if gd_confidence is not None
                    else np.zeros((len(masks),), dtype=np.float32)
                ),
                "sam2_predicted_iou": confidences.astype(np.float32),
            },
        )


class GroundedSAMPipeline:
    def __init__(
        self,
        device: str = "cuda",
        enable_ram: bool = True,
        gsa_repo_root: str | os.PathLike[str] | None = None,
        ram_repo_root: str | os.PathLike[str] | None = None,
        ram_checkpoint_path: str | None = None,
        grounding_dino_config_path: str | None = None,
        grounding_dino_checkpoint_path: str | None = None,
        sam2_model_id: str | None = None,
        sam2_config_path: str | None = None,
        sam2_checkpoint_path: str | None = None,
        sam2_apply_postprocessing: bool = True,
        sam2_iou_threshold: float | None = None,
        confidence_threshold: float | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        max_box_area_ratio: float | None = 0.8,
        add_background_classes: bool = False,
        add_classes: Iterable[str] | None = None,
        remove_substrings: Iterable[str] | None = None,
    ) -> None:
        _bootstrap_repo_paths(gsa_repo_root=gsa_repo_root, ram_repo_root=ram_repo_root)

        self.device = device
        self.enable_ram = enable_ram
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.max_box_area_ratio = max_box_area_ratio
        self.confidence_threshold = None if confidence_threshold is None else float(confidence_threshold)
        self.add_background_classes = add_background_classes
        self.add_classes = list(add_classes) if add_classes is not None else None
        self.remove_substrings = list(remove_substrings) if remove_substrings is not None else None

        self.ram = (
            RAMTagger(
                checkpoint_path=ram_checkpoint_path,
                device=device,
                gsa_repo_root=gsa_repo_root,
            )
            if enable_ram
            else None
        )
        self.grounding_dino = GroundingDINOModel(
            config_path=grounding_dino_config_path,
            checkpoint_path=grounding_dino_checkpoint_path,
            device=device,
            gsa_repo_root=gsa_repo_root,
        )
        self.sam = SAM2Segmenter(
            model_id=sam2_model_id,
            device=device,
            apply_postprocessing=sam2_apply_postprocessing,
            config_path=sam2_config_path,
            checkpoint_path=sam2_checkpoint_path,
            iou_threshold=sam2_iou_threshold,
        )

    @staticmethod
    def _resolve_threshold(value: float | None, default: float) -> float:
        return default if value is None else value

    def _filter_by_confidence(self, detections: sv.Detections) -> sv.Detections:
        if detections is None or len(detections) == 0:
            return detections
        if self.confidence_threshold is None or self.confidence_threshold < 0.0:
            return detections
        conf = getattr(detections, "confidence", None)
        if conf is None:
            return detections
        keep = np.asarray(conf, dtype=np.float32) >= float(self.confidence_threshold)
        removed = int(np.count_nonzero(~keep))
        if removed > 0:
            logger.info(
                f"Grounded-SAM filtered {removed} detections by confidence "
                f"(threshold={self.confidence_threshold:.3f})"
            )
        return _slice_detections(detections, keep)

    def predict_tags(self, image_rgb: np.ndarray) -> list[str]:
        if self.ram is None:
            raise ValueError("RAM is disabled for this pipeline.")

        ram_output = self.ram.predict(image_rgb)
        return _classes_from_ram_tags(
            ram_output.tags_english,
            add_classes=self.add_classes,
            remove_substrings=self.remove_substrings,
            add_background_classes=self.add_background_classes,
        )

    def detect(
        self,
        image_rgb: np.ndarray,
        classes: Iterable[str],
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        nms_threshold: float | None = None,
    ) -> sv.Detections:
        classes_list = _normalize_class_list(classes)
        if not classes_list:
            raise ValueError("classes must not be empty.")

        detections = self.grounding_dino.detect(
            image_rgb=image_rgb,
            classes=classes_list,
            box_threshold=self._resolve_threshold(box_threshold, self.box_threshold),
            text_threshold=self._resolve_threshold(text_threshold, self.text_threshold),
            nms_threshold=self._resolve_threshold(nms_threshold, self.nms_threshold),
            max_box_area_ratio=self.max_box_area_ratio,
        )
        return self._filter_by_confidence(self.sam.segment(image_rgb=image_rgb, detections=detections))

    def predict(
        self,
        image_rgb: np.ndarray,
        classes: Iterable[str] | None = None,
        *,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
        nms_threshold: float | None = None,
    ) -> sv.Detections:
        classes_list = _normalize_class_list(classes) if classes is not None else None
        if classes_list is None:
            if self.ram is None:
                raise ValueError("classes were not provided and RAM is disabled.")
            classes_list = self.predict_tags(image_rgb)

        detections = self.grounding_dino.detect(
            image_rgb=image_rgb,
            classes=classes_list,
            box_threshold=self._resolve_threshold(box_threshold, self.box_threshold),
            text_threshold=self._resolve_threshold(text_threshold, self.text_threshold),
            nms_threshold=self._resolve_threshold(nms_threshold, self.nms_threshold),
            max_box_area_ratio=self.max_box_area_ratio,
        )
        return self._filter_by_confidence(self.sam.segment(image_rgb=image_rgb, detections=detections))

    def cleanup(self) -> None:
        try:
            predictor = getattr(self.sam, "predictor", None)
            if predictor is not None and hasattr(predictor, "reset_predictor"):
                try:
                    predictor.reset_predictor()
                except Exception:
                    pass
        finally:
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
