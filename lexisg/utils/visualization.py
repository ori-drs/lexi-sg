from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from supervision import BoxAnnotator, ColorPalette, Detections, MaskAnnotator

from lexisg.utils.sv_compat import mask_to_xyxy


def rect_intersection_area(a, b) -> int:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def annotate_labels_collision_aware(image: np.ndarray, detections: Detections, labels: List[str]):
    if detections is None or len(detections) == 0 or not labels:
        return image
    canvas = image.copy()
    h, w = canvas.shape[:2]
    occupied = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    pad_x = 6
    pad_y = 4
    margin = 2
    palette = ColorPalette.DEFAULT

    for idx in range(min(len(detections), len(labels))):
        label = str(labels[idx])
        x1, y1, x2, y2 = np.asarray(detections.xyxy[idx]).astype(int).tolist()
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        box_w = text_w + 2 * pad_x
        box_h = text_h + baseline + 2 * pad_y
        class_id = idx
        if getattr(detections, "class_id", None) is not None and len(detections.class_id) > idx:
            class_id = int(detections.class_id[idx])
        color = palette.by_idx(abs(int(class_id)))
        bg_color = (int(color.b), int(color.g), int(color.r))
        luminance = 0.2126 * float(color.r) + 0.7152 * float(color.g) + 0.0722 * float(color.b)
        text_color = (0, 0, 0) if luminance > 160.0 else (255, 255, 255)

        candidates = [
            (x1, y1 - box_h - margin),
            (x1, y2 + margin),
            (x2 - box_w, y1 - box_h - margin),
            (x2 - box_w, y2 + margin),
            (x1, y1 + margin),
            (x2 - box_w, y1 + margin),
        ]

        best_rect = None
        best_overlap = None
        for cand_x, cand_y in candidates:
            draw_x = min(max(int(cand_x), 0), max(w - box_w, 0))
            draw_y = min(max(int(cand_y), 0), max(h - box_h, 0))
            rect = (draw_x, draw_y, draw_x + box_w, draw_y + box_h)
            overlap = sum(rect_intersection_area(rect, prev) for prev in occupied)
            if best_rect is None or overlap < best_overlap:
                best_rect = rect
                best_overlap = overlap
                if overlap == 0:
                    break

        if best_rect is None:
            continue

        rx1, ry1, rx2, ry2 = best_rect
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), bg_color, thickness=-1)
        cv2.putText(
            canvas,
            label,
            (rx1 + pad_x, ry1 + pad_y + text_h),
            font,
            font_scale,
            text_color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        occupied.append(best_rect)
    return canvas


def save_detection_overlay_frame(image: np.ndarray, dets, save_dir: str, frame_index: int):
    os.makedirs(save_dir, exist_ok=True)
    raw_dir = os.path.join(save_dir, "raw_rgb")
    os.makedirs(raw_dir, exist_ok=True)
    mask_annotator = MaskAnnotator()
    box_annotator = BoxAnnotator()
    try:
        labels = []
        for det_idx in range(len(dets)):
            conf = float(dets.confidence[det_idx]) if dets.confidence is not None else None
            labels.append(f"{det_idx} {conf:.2f}" if conf is not None else f"{det_idx}")
        annotated = mask_annotator.annotate(scene=image.copy(), detections=dets)
        annotated = box_annotator.annotate(scene=annotated, detections=dets)
        annotated = annotate_labels_collision_aware(annotated, dets, labels)
        Image.fromarray(annotated).save(os.path.join(save_dir, f"{int(frame_index):03d}.jpg"))
        Image.fromarray(image).save(os.path.join(raw_dir, f"{int(frame_index):03d}.jpg"))
    except Exception as exc:
        logger.warning(f"Failed to save detection overlay for frame {int(frame_index)}: {exc}")


def save_global_mask_overlays(
    images: Dict[int, np.ndarray],
    masks: Dict[int, Dict[int, np.ndarray]],
    save_dir: str,
    frame_ids=None,
    draw_boxes: bool = True,
    draw_labels: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    raw_dir = os.path.join(save_dir, "raw_rgb")
    os.makedirs(raw_dir, exist_ok=True)
    mask_annotator = MaskAnnotator()
    box_annotator = BoxAnnotator() if draw_boxes else None

    if frame_ids is None:
        ordered_frame_ids = sorted(int(fid) for fid in masks.keys())
    else:
        ordered_frame_ids = [int(fid) for fid in np.asarray(frame_ids, dtype=np.int32).tolist()]

    for fid in ordered_frame_ids:
        try:
            image = images.get(int(fid))
            if image is None:
                logger.warning(f"Missing overlay image for frame {int(fid)}")
                continue
            image = np.asarray(image, dtype=np.uint8)
            per_frame = masks.get(int(fid), {})
            if not per_frame:
                Image.fromarray(image).save(os.path.join(save_dir, f"frame_{int(fid):06d}.jpg"))
                Image.fromarray(image).save(os.path.join(raw_dir, f"frame_{int(fid):06d}.jpg"))
                continue

            det_list = []
            for obj_id, mask in sorted(per_frame.items()):
                mask_bool = np.asarray(mask).astype(bool)
                if mask_bool.sum() == 0:
                    continue
                mask_arr = mask_bool[None, ...]
                det_list.append(
                    Detections(
                        xyxy=mask_to_xyxy(mask_arr),
                        mask=mask_arr,
                        class_id=np.array([int(obj_id)]),
                        confidence=np.array([1.0], dtype=float),
                    )
                )

            if not det_list:
                Image.fromarray(image).save(os.path.join(save_dir, f"frame_{int(fid):06d}.jpg"))
                Image.fromarray(image).save(os.path.join(raw_dir, f"frame_{int(fid):06d}.jpg"))
                continue

            dets = Detections.merge(det_list)
            annotated = mask_annotator.annotate(scene=image.copy(), detections=dets)
            if box_annotator is not None:
                annotated = box_annotator.annotate(scene=annotated, detections=dets)
            if draw_labels:
                annotated = annotate_labels_collision_aware(
                    annotated,
                    dets,
                    [str(int(cid)) for cid in np.asarray(dets.class_id).tolist()],
                )
            Image.fromarray(annotated).save(os.path.join(save_dir, f"frame_{int(fid):06d}.jpg"))
            Image.fromarray(image).save(os.path.join(raw_dir, f"frame_{int(fid):06d}.jpg"))
        except Exception as exc:
            logger.warning(f"Failed to save global merged mask overlay for frame {int(fid)}: {exc}")
