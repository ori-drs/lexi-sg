import numpy as np
import open3d as o3d

class Node:
    def __init__(
        self, 
        idx: int,
        pose: np.ndarray = None,
    ):
        self.idx = idx
        self.pose = pose

    def __repr__(self):
        return (
            f"Node(idx={self.idx}, "
            f"pose={'set' if self.pose is not None else 'None'}, "
        )

class RoomNode(Node): 
    def __init__(
        self, 
        idx: int, 
        label: str,
        predictions: dict,
        img_paths: np.ndarray, 
        ref_pose: int,
        local_poses: np.ndarray,
        feats: list,
        optimised_ref_pose: np.ndarray = None,
        visual_pose: np.ndarray = None,
        transition_frames: set = None,
    ):
        self.idx = idx
        self.label = label
        self.predictions = predictions
        self.img_paths = img_paths
        self.ref_pose = ref_pose
        self.local_poses = local_poses
        self.feats = feats
        self.optimised_ref_pose = optimised_ref_pose
        self.visual_pose = visual_pose
        self.pcd = None
        self.object_segmentation = None
        # Frames that came from a folded transition segment (e.g. a doorway).
        # They take part in reconstruction and object segmentation, but are
        # left out of the node's point cloud so the loop-closure overlap test
        # sees the room's own geometry only.
        self.transition_frames = set(transition_frames or ())

    def build_pcd(self, voxel_size=0.04):
        """Build point cloud in local frame (without optimised_ref_pose).
        Use copy.deepcopy(node.pcd).transform(node.optimised_ref_pose) to get world-space PCD."""
        pcd_total = o3d.geometry.PointCloud()
        for frame_idx, local_pose in enumerate(self.local_poses):
            if self.img_paths[frame_idx] in self.transition_frames:
                continue
            color = np.ascontiguousarray(self.predictions["images"][frame_idx])
            depth = self.predictions["depth"][frame_idx]
            intr = self.predictions["intrinsic"][frame_idx]
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color),
                o3d.geometry.Image(depth),
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
                depth_trunc=4.0
            )
            height, width = depth.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcd_total += pcd.transform(local_pose).voxel_down_sample(voxel_size)
        self.pcd = pcd_total
        return self.pcd

    def __repr__(self):
        return (
            f"Node(idx={self.idx}, "
        )
