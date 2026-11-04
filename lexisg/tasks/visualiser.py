import colorsys
import json
import os
import pathlib

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class Visualiser:
    def __init__(self, align_floor=True):
        self.align_floor = align_floor
        self.T_align = np.eye(4)

    def save(self, graph, output_path, save_segmented=True):
        os.makedirs(output_path, exist_ok=True)
        original_dir = os.path.join(output_path, "original")
        os.makedirs(original_dir, exist_ok=True)

        merged_pcd = o3d.geometry.PointCloud()
        room_pcds = []
        global_poses = []

        for idx, node in graph.room_nodes.items():
            pose = np.asarray(node.optimised_ref_pose, dtype=np.float64) if node.optimised_ref_pose is not None else np.eye(4)
            pcd_world = o3d.geometry.PointCloud(node.pcd).transform(pose)
            if len(pcd_world.points) == 0:
                continue
            merged_pcd += pcd_world
            room_pcds.append((f"{node.label}_{node.idx}", pcd_world))

            timestamps = [pathlib.Path(p).stem for p in node.img_paths]
            for frame_idx, local_pose in enumerate(node.local_poses):
                global_poses.append((timestamps[frame_idx], pose @ local_pose))

        if self.align_floor and len(merged_pcd.points) > 0:
            self.T_align = _floor_alignment(merged_pcd, [T for _, T in global_poses])
        merged_pcd.transform(self.T_align)

        for name, pcd_world in room_pcds:
            o3d.io.write_point_cloud(
                os.path.join(original_dir, f"{name}.pcd"),
                pcd_world.transform(self.T_align),
            )

        all_poses = []
        for timestamp, global_pose in global_poses:
            global_pose = self.T_align @ global_pose
            x, y, z = global_pose[0:3, 3]
            quat = R.from_matrix(global_pose[0:3, 0:3]).as_quat()
            all_poses.append((timestamp, [x, y, z, *quat]))

        if len(merged_pcd.points) > 0:
            o3d.io.write_point_cloud(os.path.join(output_path, "full_pcd.pcd"), merged_pcd)

        all_poses.sort(key=lambda x: x[0])
        with open(os.path.join(output_path, "pose.txt"), "w") as f:
            for timestamp, values in all_poses:
                f.write(f"{timestamp} " + " ".join(f"{v:.8f}" for v in values) + "\n")

        if save_segmented:
            self._save_segmented_pcd(output_path, graph)

    def _save_segmented_pcd(self, output_path, graph):
        merged = o3d.geometry.PointCloud()
        for idx, node in graph.room_nodes.items():
            path = os.path.join(
                output_path, "object_segmentation",
                f"room_{int(node.idx):04d}", "segmented_pointcloud", "merged.ply",
            )
            if not os.path.exists(path):
                continue
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) == 0:
                continue
            if node.optimised_ref_pose is not None and node.ref_pose is not None:
                correction = np.asarray(node.optimised_ref_pose, dtype=np.float64) @ np.linalg.inv(np.asarray(node.ref_pose, dtype=np.float64))
                pcd.transform(correction)
            merged += pcd
        merged.transform(self.T_align)
        if len(merged.points) > 0:
            o3d.io.write_point_cloud(os.path.join(output_path, "segmented_pcd.pcd"), merged)
            print(f"Segmented cloud saved: {len(merged.points)} points")

    def save_cross_room(self, output_path, result):
        """Per-room and cross-room object clouds, one distinct colour per object,
        plus the association in cross_room_objects.json."""
        keys = sorted(result["objects"])
        per_room, cross = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        for i, key in enumerate(keys):
            pcd = o3d.geometry.PointCloud(result["pcds"][key]).transform(self.T_align)
            per_room += o3d.geometry.PointCloud(pcd).paint_uniform_color(_distinct_colour(i))
            gid = result["objects"][key]["global_id"]
            cross += o3d.geometry.PointCloud(pcd).paint_uniform_color(_distinct_colour(gid))
        if len(per_room.points) > 0:
            o3d.io.write_point_cloud(os.path.join(output_path, "segmented_pcd_rooms.pcd"), per_room)
            o3d.io.write_point_cloud(os.path.join(output_path, "segmented_pcd_cross_room.pcd"), cross)
        summary = {
            "room_pairs": result["room_pairs"],
            "object_pairs_compared": result["compared"],
            "objects": len(keys),
            "global_objects": len(result["rooms"]),
            "matches": result["matches"],
            "global_objects_by_id": {
                gid: {"room": room, "members": [list(k) for k in keys if result["objects"][k]["global_id"] == gid]}
                for gid, room in result["rooms"].items()
            },
        }
        with open(os.path.join(output_path, "cross_room_objects.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Cross-room objects: {len(keys)} per-room objects -> {len(result['rooms'])} global objects, "
              f"{len(result['matches'])} matches")


def _distinct_colour(i):
    """Golden-ratio hue spacing: neighbouring ids get well-separated colours."""
    h = (i * 0.618033988749895) % 1.0
    return list(colorsys.hsv_to_rgb(h, 0.75, 0.95 - 0.25 * ((i // 7) % 2)))


def _floor_alignment(pcd, cam_poses, voxel_size=0.05, max_tilt_deg=35.0, min_frac=0.02, max_planes=6):
    """4x4 transform making the floor plane z = 0 with +Z up."""
    # OpenCV cameras: world up is roughly the mean of -Y over all frames.
    up = -np.mean([np.asarray(T)[:3, 1] for T in cam_poses], axis=0) if cam_poses else np.array([0.0, -1.0, 0.0])
    up /= np.linalg.norm(up)

    rest = pcd.voxel_down_sample(voxel_size)
    n_total = len(rest.points)
    floor = None
    for _ in range(max_planes):
        if len(rest.points) < 3:
            break
        (a, b, c, d), inliers = rest.segment_plane(voxel_size, 3, 1000)
        normal = np.array([a, b, c])
        if normal @ up < 0:
            normal, d = -normal, -d
        if len(inliers) >= min_frac * n_total and normal @ up >= np.cos(np.deg2rad(max_tilt_deg)):
            height = -d  # plane offset along the normal
            if floor is None or height < floor[1]:
                floor = (normal, height)
        rest = rest.select_by_index(inliers, invert=True)

    if floor is None:
        print("Floor alignment: no floor plane found; using camera up direction.")
        normal = up
        height = np.percentile(np.asarray(pcd.points) @ up, 2)
    else:
        normal, height = floor
        print(f"Floor alignment: normal {np.round(normal, 3)}, tilt from camera up {np.degrees(np.arccos(np.clip(normal @ up, -1, 1))):.1f} deg")

    T = np.eye(4)
    T[:3, :3] = R.align_vectors([[0.0, 0.0, 1.0]], [normal])[0].as_matrix()
    T[2, 3] = -height
    return T
