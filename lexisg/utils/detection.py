import numpy as np
import cv2
from supervision import Detections

def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    mask1 = mask1.squeeze() if mask1.ndim == 3 else mask1
    mask2 = mask2.squeeze() if mask2.ndim == 3 else mask2
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape for IoU calculation.")
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection, union

def sim_matrix_iou(dets1: Detections, dets2: Detections) -> np.ndarray:
    if dets1.mask is None or dets2.mask is None:
        raise ValueError("Both detections must have masks to compute intersection.")
    n1, n2 = len(dets1), len(dets2)
    iou_matrix = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            intersection, union = mask_iou(dets1.mask[i], dets2.mask[j])
            iou = intersection / union if union > 0 else 0.0
            iou_matrix[i, j] = iou
    return iou_matrix

def filter_overlapping_detections(
    detections: Detections,
    iou_threshold: float = 0.0,
    inner_overlap_threshold: float = 0.0,
) -> Detections:
    if detections.mask is None or len(detections) == 0:
        return detections

    # Sort by area (descending)
    areas = [np.count_nonzero(mask) for mask in detections.mask]
    sorted_indices = np.argsort(areas)[::-1]
    detections = detections[sorted_indices]

    # update idx_to_remove based on thresholds
    idx_to_remove = set()
    if iou_threshold:
        sim_mat_iou = np.triu(sim_matrix_iou(detections, detections), k=1)
        idx_to_remove.update(np.where(sim_mat_iou > iou_threshold)[1])
    if inner_overlap_threshold:
        masks = np.asarray(detections.mask).astype(bool)
        for i in range(len(detections)):
            if i in idx_to_remove:
                continue
            area_i = int(np.count_nonzero(masks[i]))
            if area_i <= 0:
                continue
            for j in range(i + 1, len(detections)):
                if j in idx_to_remove:
                    continue
                area_j = int(np.count_nonzero(masks[j]))
                if area_j <= 0:
                    continue
                intersection = int(np.logical_and(masks[i], masks[j]).sum())
                smaller = min(area_i, area_j)
                ios = intersection / smaller if smaller > 0 else 0.0
                if ios > inner_overlap_threshold:
                    # Detections are sorted by descending area, so remove the larger mask
                    # when it nearly covers the smaller one.
                    idx_to_remove.add(i if area_i >= area_j else j)
                    break

    # Remove overlapping detections
    return detections[~np.isin(np.arange(len(detections)), idx_to_remove)]


def _boxes_from_masks(masks) -> np.ndarray:
    """Derive xyxy boxes from boolean masks; returns (N, 4) float32."""
    if masks is None:
        return np.zeros((0, 4), dtype=np.float32)

    if isinstance(masks, list):
        arr = np.stack([m.astype(bool) for m in masks], axis=0) if len(masks) else np.zeros((0, 0, 0), bool)
    elif isinstance(masks, np.ndarray):
        if masks.ndim == 2:  # single mask
            arr = masks[None, ...].astype(bool)
        elif masks.ndim == 3:
            arr = masks.astype(bool)
        else:
            arr = np.zeros((0, 0, 0), bool)
    else:
        arr = np.zeros((0, 0, 0), bool)

    N, h, w = arr.shape
    boxes = np.zeros((N, 4), dtype=np.float32)

    for i in range(N):
        ys, xs = np.nonzero(arr[i])
        if ys.size > 0:
            boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return boxes


def resize_detections(
    dets: Detections,
    h_new: int,
    w_new: int
) -> Detections:
    """
    Rescale a Detections object to new size (h_new, w_new).
    - If masks are present, resizes them.
    - If boxes are present, rescales them.
    - If only masks are present, derives boxes from resized masks.
    """

    # Determine original size from masks or boxes
    if getattr(dets, "mask", None) is not None:
        m = dets.mask
        if isinstance(m, list):
            h_orig, w_orig = m[0].shape
        elif isinstance(m, np.ndarray) and m.ndim >= 2:
            h_orig, w_orig = m.shape[-2:]
        else:
            h_orig = w_orig = None
    elif getattr(dets, "xyxy", None) is not None and len(dets.xyxy) > 0:
        # if no masks, original size is not directly known — scaling factors must be estimated externally
        raise ValueError("Cannot infer original size without masks")
    else:
        return dets  # nothing to resize

    sx = w_new / w_orig
    sy = h_new / h_orig

    # Resize boxes if present
    new_xyxy = None
    if getattr(dets, "xyxy", None) is not None and len(dets.xyxy) > 0:
        new_xyxy = dets.xyxy.astype(np.float32).copy()
        new_xyxy[:, [0, 2]] *= sx
        new_xyxy[:, [1, 3]] *= sy
        new_xyxy[:, [0, 2]] = np.clip(new_xyxy[:, [0, 2]], 0, w_new - 1)
        new_xyxy[:, [1, 3]] = np.clip(new_xyxy[:, [1, 3]], 0, h_new - 1)

    # Resize masks if present
    new_masks = None
    if getattr(dets, "mask", None) is not None:
        if isinstance(m, list):
            new_masks = [
                cv2.resize(mask.astype(np.uint8), (w_new, h_new), interpolation=cv2.INTER_NEAREST).astype(bool)
                for mask in m
            ]
        elif isinstance(m, np.ndarray):
            if m.ndim == 3:  # (N, H, W)
                N = m.shape[0]
                new_masks = np.zeros((N, h_new, w_new), dtype=bool)
                for i in range(N):
                    new_masks[i] = cv2.resize(m[i].astype(np.uint8), (w_new, h_new), interpolation=cv2.INTER_NEAREST).astype(bool)
            elif m.ndim == 2:
                new_masks = cv2.resize(m.astype(np.uint8), (w_new, h_new), interpolation=cv2.INTER_NEAREST).astype(bool)

    # If boxes missing but masks exist, derive boxes
    if new_xyxy is None and new_masks is not None:
        new_xyxy = _boxes_from_masks(new_masks)

    # If no boxes at all, ensure empty array
    if new_xyxy is None:
        new_xyxy = np.zeros((0, 4), dtype=np.float32)

    return Detections(
        xyxy=new_xyxy,
        mask=new_masks,
        confidence=getattr(dets, "confidence", None),
        class_id=getattr(dets, "class_id", None),
        tracker_id=getattr(dets, "tracker_id", None)
    )
