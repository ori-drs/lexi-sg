import numpy as np


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] boxes to [x1, y1, x2, y2]."""
    xywh = np.asarray(xywh)
    if xywh.ndim == 1:
        xywh = xywh[None, :]
    out = np.zeros((xywh.shape[0], 4), dtype=np.float32)
    out[:, 0] = xywh[:, 0]
    out[:, 1] = xywh[:, 1]
    out[:, 2] = xywh[:, 0] + xywh[:, 2]
    out[:, 3] = xywh[:, 1] + xywh[:, 3]
    return out


def mask_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """Convert binary masks of shape (N, H, W) to [x1, y1, x2, y2]."""
    masks = np.asarray(masks).astype(bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    n = masks.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        ys, xs = np.where(masks[i])
        if xs.size == 0 or ys.size == 0:
            continue
        out[i, 0] = float(xs.min())
        out[i, 1] = float(ys.min())
        out[i, 2] = float(xs.max())
        out[i, 3] = float(ys.max())
    return out
