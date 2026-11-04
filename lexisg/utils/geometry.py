import numpy as np
import open3d as o3d
import cv2


def precompute_backprojection(depth_image, rgb_image, intrinsics, cam_pose, 
                              depth_scale=1.0, max_depth=None):
    """
    Precompute 3D world points and colors for all valid depth pixels, and build a pixel->point index map.

    Args:
        depth_image (np.ndarray, shape=(H,W)): Depth map (uint16 or float), same units as z*depth_scale.
        rgb_image (np.ndarray, shape=(H,W,3)): Corresponding color image (uint8 [0-255] or float [0-1]).
        intrinsics (np.ndarray, shape=(3,3)): Camera intrinsic matrix.
        cam_pose (np.ndarray, shape=(4,4)): Camera-to-world transform.
        depth_scale (float): Scale to convert integer depth to meters (default 1.0).

    Returns:
        dict containing:
            - 'points' (np.ndarray of shape (M,3)): 3D points in world frame.
            - 'colors' (np.ndarray of shape (M,3)): RGB colors normalized to [0,1].
            - 'pixel_to_index' (np.ndarray of shape (H,W)): int map, -1 for invalid, else index into points/colors.
    """
    H, W = depth_image.shape
    if rgb_image.shape[:2] != (H, W):
        raise ValueError(
            f"RGB/depth shape mismatch: depth={depth_image.shape}, rgb={rgb_image.shape}"
        )
    # Convert depth to float meters
    depth = depth_image.astype(np.float32)
    if depth_scale != 1.0:
        depth = depth / depth_scale
    valid = depth > 0
    if max_depth is not None:
        valid &= (depth <= max_depth)

    # Pixel coordinates of valid depths
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    us = us[valid]
    vs = vs[valid]
    ds = depth[valid]

    # Backproject into camera frame
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    xs = (us - cx) * ds / fx
    ys = (vs - cy) * ds / fy
    zs = ds
    pts_cam = np.stack([xs, ys, zs], axis=1)  # (M,3)

    # Transform to world frame
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
    pts_cam_h = np.hstack([pts_cam, ones])    # (M,4)
    pts_world_h = (cam_pose @ pts_cam_h.T).T  # (M,4)
    pts_world = pts_world_h[:, :3]

    # Extract colors and normalize
    cols = rgb_image.astype(np.float32)
    if rgb_image.dtype == np.uint8:
        cols = cols / 255.0
    cols = cols[vs, us]

    # Build pixel->index map
    pixel_to_index = -np.ones((H, W), dtype=int)
    linear_idx = np.arange(pts_world.shape[0], dtype=int)
    pixel_to_index[vs, us] = linear_idx

    return {
        'points': pts_world,
        'colors': cols,
        'pixel_to_index': pixel_to_index
    }

_SHARED_GEOM_PACK_BITS = 21
_SHARED_GEOM_PACK_BIAS = 1 << (_SHARED_GEOM_PACK_BITS - 1)
_SHARED_GEOM_PACK_LIMIT = 1 << _SHARED_GEOM_PACK_BITS


def _pack_voxel_rows_int64(voxel_rows: np.ndarray) -> np.ndarray:
    voxel_rows = np.asarray(voxel_rows, dtype=np.int64)
    if voxel_rows.size == 0:
        return np.zeros((0,), dtype=np.int64)
    shifted = voxel_rows + int(_SHARED_GEOM_PACK_BIAS)
    if np.any(shifted < 0) or np.any(shifted >= int(_SHARED_GEOM_PACK_LIMIT)):
        raise ValueError(
            "Voxel coordinates exceed packed shared-geometry range. "
            f"Supported per-axis voxel index range is "
            f"[{-_SHARED_GEOM_PACK_BIAS}, {_SHARED_GEOM_PACK_BIAS - 1}]."
        )
    return (
        (shifted[:, 0] << (2 * _SHARED_GEOM_PACK_BITS))
        | (shifted[:, 1] << _SHARED_GEOM_PACK_BITS)
        | shifted[:, 2]
    ).astype(np.int64, copy=False)


def _unpack_voxel_keys_int64(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    if keys.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    mask = (1 << _SHARED_GEOM_PACK_BITS) - 1
    rows = np.zeros((keys.shape[0], 3), dtype=np.int32)
    rows[:, 2] = (keys & mask).astype(np.int32) - int(_SHARED_GEOM_PACK_BIAS)
    rows[:, 1] = ((keys >> _SHARED_GEOM_PACK_BITS) & mask).astype(np.int32) - int(_SHARED_GEOM_PACK_BIAS)
    rows[:, 0] = ((keys >> (2 * _SHARED_GEOM_PACK_BITS)) & mask).astype(np.int32) - int(_SHARED_GEOM_PACK_BIAS)
    return rows


def build_shared_pointcloud_lookup_from_predictions(
    predictions,
    depth_scale=1.0,
    max_depth=None,
    include_colors: bool = False,
    voxel_size: float = 0.01,
):
    """
    Aggregate all per-frame backprojected points into one chunk-local point set and
    build per-frame pixel lookups into that aggregated array.

    The returned geometry lives in the same coordinate frame as predictions["extrinsic"].
    For the object segmentation pipeline this is the chunk-local frame; predictions["T_world_local"]
    maps it into world space.
    """
    depths = predictions["depth"]
    images = predictions["images"]
    intrinsics = predictions["intrinsic"]
    extrinsics = predictions["extrinsic"]

    frame_raw_point_counts = []
    total_pixels = 0
    total_valid_pixels = 0
    frame_unique_keys = []
    frame_point_sums = []
    frame_point_counts = []
    frame_color_sums = [] if include_colors else None

    frame_count = int(len(depths))
    image_height = int(depths.shape[1]) if depths.ndim >= 3 else 0
    image_width = int(depths.shape[2]) if depths.ndim >= 3 else 0
    voxel_size = float(voxel_size)

    for depth, rgb, intr, extr in zip(depths, images, intrinsics, extrinsics):
        precomp = precompute_backprojection(
            depth,
            rgb,
            intr,
            extr,
            depth_scale=depth_scale,
            max_depth=max_depth,
        )
        points = np.asarray(precomp["points"], dtype=np.float32)
        pixel_to_index = np.asarray(precomp["pixel_to_index"], dtype=np.int32)
        valid = pixel_to_index >= 0
        total_pixels += int(pixel_to_index.size)
        total_valid_pixels += int(valid.sum())
        frame_raw_point_counts.append(int(points.shape[0]))
        if points.shape[0] == 0:
            frame_unique_keys.append(np.zeros((0,), dtype=np.int64))
            frame_point_sums.append(np.zeros((0, 3), dtype=np.float32))
            frame_point_counts.append(np.zeros((0,), dtype=np.float32))
            if include_colors:
                frame_color_sums.append(np.zeros((0, 3), dtype=np.float32))
            continue

        if voxel_size > 0:
            voxel_rows = np.floor(points / voxel_size).astype(np.int32)
            packed_keys = _pack_voxel_rows_int64(voxel_rows)
            unique_keys, inverse = np.unique(packed_keys, return_inverse=True)
            counts = np.bincount(inverse, minlength=unique_keys.shape[0]).astype(np.float32)
            point_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float32)
            for axis in range(3):
                point_sums[:, axis] = np.bincount(
                    inverse,
                    weights=points[:, axis],
                    minlength=unique_keys.shape[0],
                ).astype(np.float32)
            frame_unique_keys.append(unique_keys.astype(np.int64, copy=False))
            frame_point_sums.append(point_sums)
            frame_point_counts.append(counts)
            if include_colors:
                colors = np.asarray(precomp["colors"], dtype=np.float32)
                color_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float32)
                for axis in range(3):
                    color_sums[:, axis] = np.bincount(
                        inverse,
                        weights=colors[:, axis],
                        minlength=unique_keys.shape[0],
                    ).astype(np.float32)
                frame_color_sums.append(color_sums)
        else:
            frame_unique_keys.append(np.zeros((0,), dtype=np.int64))
            frame_point_sums.append(points.astype(np.float32, copy=False))
            frame_point_counts.append(np.ones((points.shape[0],), dtype=np.float32))
            if include_colors:
                frame_color_sums.append(np.asarray(precomp["colors"], dtype=np.float32))

    raw_point_count = int(sum(frame_raw_point_counts))
    pixel_to_index_arr = -np.ones((frame_count, image_height, image_width), dtype=np.int32)

    if raw_point_count == 0:
        points_arr = np.zeros((0, 3), dtype=np.float32)
        colors_arr = np.zeros((0, 3), dtype=np.float32) if include_colors else None
        unique_voxels = np.zeros((0, 3), dtype=np.int32)
    elif voxel_size > 0:
        all_unique_keys = np.concatenate(frame_unique_keys, axis=0) if frame_unique_keys else np.zeros((0,), dtype=np.int64)
        if all_unique_keys.size == 0:
            points_arr = np.zeros((0, 3), dtype=np.float32)
            colors_arr = np.zeros((0, 3), dtype=np.float32) if include_colors else None
            global_keys = np.zeros((0,), dtype=np.int64)
            unique_voxels = np.zeros((0, 3), dtype=np.int32)
        else:
            global_keys, global_inverse = np.unique(all_unique_keys, return_inverse=True)
            all_point_sums = np.concatenate(frame_point_sums, axis=0)
            all_point_counts = np.concatenate(frame_point_counts, axis=0)
            global_counts = np.bincount(
                global_inverse,
                weights=all_point_counts,
                minlength=global_keys.shape[0],
            ).astype(np.float32)
            points_arr = np.zeros((global_keys.shape[0], 3), dtype=np.float32)
            denom = np.maximum(global_counts, 1.0)
            for axis in range(3):
                points_arr[:, axis] = np.bincount(
                    global_inverse,
                    weights=all_point_sums[:, axis],
                    minlength=global_keys.shape[0],
                ).astype(np.float32) / denom
            colors_arr = None
            if include_colors:
                all_color_sums = np.concatenate(frame_color_sums, axis=0)
                colors_arr = np.zeros((global_keys.shape[0], 3), dtype=np.float32)
                for axis in range(3):
                    colors_arr[:, axis] = np.bincount(
                        global_inverse,
                        weights=all_color_sums[:, axis],
                        minlength=global_keys.shape[0],
                    ).astype(np.float32) / denom
            unique_voxels = _unpack_voxel_keys_int64(global_keys)

        for frame_idx, (depth, rgb, intr, extr) in enumerate(zip(depths, images, intrinsics, extrinsics)):
            precomp = precompute_backprojection(
                depth,
                rgb,
                intr,
                extr,
                depth_scale=depth_scale,
                max_depth=max_depth,
            )
            points = np.asarray(precomp["points"], dtype=np.float32)
            pixel_to_index = np.asarray(precomp["pixel_to_index"], dtype=np.int32)
            if points.shape[0] == 0:
                pixel_to_index_arr[frame_idx] = pixel_to_index
                continue
            voxel_rows = np.floor(points / voxel_size).astype(np.int32)
            packed_keys = _pack_voxel_rows_int64(voxel_rows)
            global_point_indices = np.searchsorted(global_keys, packed_keys).astype(np.int32, copy=False)
            if np.any(global_keys[global_point_indices] != packed_keys):
                raise RuntimeError("Shared geometry voxel remapping failed due to missing global voxel keys.")
            frame_lookup = -np.ones_like(pixel_to_index, dtype=np.int32)
            valid = pixel_to_index >= 0
            if np.any(valid):
                frame_lookup[valid] = global_point_indices[pixel_to_index[valid]]
            pixel_to_index_arr[frame_idx] = frame_lookup
    else:
        points_arr = np.zeros((raw_point_count, 3), dtype=np.float32)
        colors_arr = np.zeros((raw_point_count, 3), dtype=np.float32) if include_colors else None
        unique_voxels = np.zeros((0, 3), dtype=np.int32)
        offset = 0
        for frame_idx, (depth, rgb, intr, extr) in enumerate(zip(depths, images, intrinsics, extrinsics)):
            precomp = precompute_backprojection(
                depth,
                rgb,
                intr,
                extr,
                depth_scale=depth_scale,
                max_depth=max_depth,
            )
            points = np.asarray(precomp["points"], dtype=np.float32)
            pixel_to_index = np.asarray(precomp["pixel_to_index"], dtype=np.int32)
            frame_lookup = -np.ones_like(pixel_to_index, dtype=np.int32)
            if points.shape[0] > 0:
                count = int(points.shape[0])
                points_arr[offset : offset + count] = points
                if include_colors:
                    colors_arr[offset : offset + count] = np.asarray(precomp["colors"], dtype=np.float32)
                valid = pixel_to_index >= 0
                if np.any(valid):
                    frame_lookup[valid] = pixel_to_index[valid] + int(offset)
                offset += count
            pixel_to_index_arr[frame_idx] = frame_lookup

    payload = {
        "points": points_arr,
        "pixel_to_index": pixel_to_index_arr,
        "frame_raw_point_counts": np.asarray(frame_raw_point_counts, dtype=np.int32),
        "coordinate_frame": "chunk_local",
        "point_format": "xyz",
        "frame_count": frame_count,
        "image_height": image_height,
        "image_width": image_width,
        "total_pixels": int(total_pixels),
        "valid_pixels": int(total_valid_pixels),
        "raw_point_count": raw_point_count,
        "voxel_downsample_size": voxel_size,
        "voxel_count": int(points_arr.shape[0]),
        "voxel_indices": unique_voxels,
    }
    if include_colors:
        payload["colors"] = colors_arr if colors_arr is not None else np.zeros((0, 3), dtype=np.float32)
        payload["point_format"] = "rgbxyz"
    return payload


def masked_point_indices_from_lookup(shared_geometry, fid, mask, erode_iters=0, erode_kernel=5):
    pixel_to_index = np.asarray(shared_geometry["pixel_to_index"][int(fid)], dtype=np.int64)

    if erode_iters > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        mask_proc = cv2.erode(mask.astype(np.uint8), kernel, iterations=erode_iters).astype(bool)
    else:
        mask_proc = mask.astype(bool)

    if not np.any(mask_proc):
        return np.zeros((0,), dtype=np.int64)

    idxs = pixel_to_index[mask_proc]
    idxs = idxs[idxs >= 0]
    if idxs.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(idxs.astype(np.int64, copy=False))


def backproject_mask_with_shared_geometry(
    shared_geometry,
    fid,
    mask,
    erode_iters=0,
    erode_kernel=5,
):
    idxs = masked_point_indices_from_lookup(
        shared_geometry,
        fid,
        mask,
        erode_iters=erode_iters,
        erode_kernel=erode_kernel,
    )
    pc = o3d.geometry.PointCloud()
    if idxs.size == 0:
        return pc

    points = np.asarray(shared_geometry["points"], dtype=np.float32)
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[idxs], dtype=np.float64))
    colors = np.asarray(shared_geometry.get("colors", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
    if colors.shape[0] == points.shape[0]:
        pc.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors[idxs], dtype=np.float64))
    return pc


def backproject_masks_to_pointclouds(
        precomp, 
        mask,
        erode_iters=0,
        erode_kernel=5
    ):
    """
    Build per-mask Open3D point clouds from a precomputed backprojection.

    Args:
        precomp (dict): Output of precompute_backprojection.
        masks (np.ndarray): Binary mask of shape (H,W) where True indicates pixels to include.
    Returns:
        o3d.geometry.PointCloud: Point cloud containing points and colors for the mask.
    """
    points = precomp['points']
    colors = precomp['colors']
    pixel_to_index = precomp['pixel_to_index']

    # Optionally expand mask to tolerate reprojection error
    if erode_iters > 0:
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        mask_proc = cv2.erode(mask.astype(np.uint8), kernel, iterations=erode_iters).astype(bool)
    else:
        mask_proc = mask.astype(bool)

    if not np.any(mask_proc):
        # If no valid pixels, return empty point cloud
        return o3d.geometry.PointCloud()

    idxs = pixel_to_index[mask_proc]
    idxs = idxs[idxs >= 0]
    pc = o3d.geometry.PointCloud()
    if idxs.size > 0:
        pc.points  = o3d.utility.Vector3dVector(np.ascontiguousarray(points[idxs],  dtype=np.float64))
        pc.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors[idxs], dtype=np.float64))
    return pc


def backproject_mask_from_predictions(
        predictions,
        fid,
        mask,
        depth_scale=1.0,
        max_depth=None,
        erode_iters=0,
        erode_kernel=5,
    ):
    """
    Backproject a single mask on demand from a predictions dict without precomputing
    all frame backprojections.
    """
    shared_geometry = predictions.get("shared_geometry", None)
    if shared_geometry is not None:
        return backproject_mask_with_shared_geometry(
            shared_geometry,
            fid,
            mask,
            erode_iters=erode_iters,
            erode_kernel=erode_kernel,
        )
    precomp = precompute_backprojection(
        predictions["depth"][fid],
        predictions["images"][fid],
        predictions["intrinsic"][fid],
        predictions["extrinsic"][fid],
        depth_scale=depth_scale,
        max_depth=max_depth,
    )
    return backproject_masks_to_pointclouds(
        precomp,
        mask,
        erode_iters=erode_iters,
        erode_kernel=erode_kernel,
    )
