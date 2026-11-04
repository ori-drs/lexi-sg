import hashlib
import json
import math
import os
from typing import Tuple

import numpy as np
from PIL import Image
from loguru import logger


class BaseDataset:
    def __init__(self, root_dir, scene_name=None):
        self.root_dir = root_dir
        self.scene_name = scene_name

    def get_rgb(self):
        raise NotImplementedError

    def get_poses(self):
        raise NotImplementedError


# MapAnything expects OpenCV cam-to-world poses (+X right, +Y down, +Z forward).
# Datasets rendered with OpenGL-style cameras (+X right, +Y up, -Z forward),
# e.g. Habitat, need the camera Y and Z axes flipped.
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


class CustomDataset(BaseDataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    POSE_DIRS = ("pose", "poses")

    def __init__(
        self,
        root_dir,
        intrinsics=None,
        depth_scale=1.0,
        pose_convention="opencv",
        poses_are_world_to_cam=False,
    ):
        super().__init__(root_dir)
        self.scene_root = root_dir
        self.intrinsics = intrinsics
        self.depth_scale = float(depth_scale)
        self.pose_convention = str(pose_convention).lower()
        self.poses_are_world_to_cam = bool(poses_are_world_to_cam)
        if self.pose_convention not in ("opencv", "opengl"):
            raise ValueError(
                f"pose_convention must be 'opencv' or 'opengl', got {pose_convention!r}"
            )

    @classmethod
    def _list_images(cls, directory):
        if not os.path.isdir(directory):
            return []
        names = sorted(os.listdir(directory))
        return [
            os.path.join(directory, name)
            for name in names
            if os.path.isfile(os.path.join(directory, name))
            and os.path.splitext(name)[1].lower() in cls.IMAGE_EXTENSIONS
        ]

    def get_rgb(self):
        rgb_dir = os.path.join(self.scene_root, "rgb")
        if os.path.isdir(rgb_dir):
            return self._list_images(rgb_dir)
        return self._list_images(self.scene_root)

    def get_depth(self):
        depth_dir = os.path.join(self.scene_root, "depth")
        return self._list_images(depth_dir)

    def _pose_dir(self):
        for name in self.POSE_DIRS:
            candidate = os.path.join(self.scene_root, name)
            if os.path.isdir(candidate):
                return candidate
        return None

    def get_poses(self):
        """Per-frame cam-to-world 4x4 poses (OpenCV convention), or None if absent.

        Reads one whitespace-separated .txt per frame from pose/ (or poses/),
        holding a 4x4 (16 values) or 3x4 (12 values) matrix, row-major. Sorted
        by filename, so frames must be named to sort in the same order as rgb/.
        """
        pose_dir = self._pose_dir()
        if pose_dir is None:
            return None
        files = sorted(
            os.path.join(pose_dir, name)
            for name in os.listdir(pose_dir)
            if name.lower().endswith(".txt")
        )
        poses = []
        for path in files:
            values = np.loadtxt(path).reshape(-1)
            if values.size == 16:
                T = values.reshape(4, 4)
            elif values.size == 12:
                T = np.eye(4)
                T[:3, :] = values.reshape(3, 4)
            else:
                raise ValueError(f"{path}: expected 12 or 16 values, got {values.size}")
            if self.poses_are_world_to_cam:
                T = np.linalg.inv(T)
            if self.pose_convention == "opengl":
                T = T @ _OPENGL_TO_OPENCV
            poses.append(T.astype(np.float64))
        return poses

    def get_intrinsics(self):
        """(K 3x3, depth_scale). Intrinsics come from config as [fx, fy, cx, cy],
        a 3x3 matrix, or a path to a text file holding either."""
        if self.intrinsics is None:
            raise NotImplementedError("No intrinsics configured for this dataset.")
        values = self.intrinsics
        if isinstance(values, (str, os.PathLike)):
            values = np.loadtxt(values)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 4:
            fx, fy, cx, cy = values
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        elif values.size == 9:
            K = values.reshape(3, 3)
        else:
            raise ValueError(f"intrinsics must be [fx, fy, cx, cy] or 3x3, got {values.size} values")
        return K, self.depth_scale


class ReplicaDataset(BaseDataset):
    def __init__(self, root_dir, scene_name):
        super().__init__(root_dir, scene_name)
        # self.root_dir = root_dir
    
    def get_rgb(self):
        rgb_path = os.path.join(self.root_dir, self.scene_name, "results")
        image_names = sorted(os.listdir(rgb_path))

        image_paths = []
        for image_name in image_names:
            if "frame" in image_name:
                image_paths.append(os.path.join(rgb_path, image_name))
                
        return image_paths
    
    def get_depth(self): 
        depth_path = os.path.join(self.root_dir, self.scene_name, "results")
        image_names = sorted(os.listdir(depth_path))

        image_paths = []
        for image_name in image_names:
            if "depth" in image_name:
                image_paths.append(os.path.join(depth_path, image_name))
                
        return image_paths
    
    def get_poses(self):
        pose_path = os.path.join(self.root_dir, self.scene_name, "traj.txt")
        poses = []
        with open(pose_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                values = [float(val) for val in line.split()]
                matrix = np.array(values).reshape((4, 4))
                poses.append(matrix)
        return poses


    def get_intrinsics(self):
        intrinsics_path = os.path.join(self.root_dir, "cam_params.json")
        with open(intrinsics_path, "r") as file:
            data = json.load(file)
            camera_params = data.get("camera")
            if camera_params:
                fx = camera_params.get("fx")
                fy = camera_params.get("fy")
                cx = camera_params.get("cx")
                cy = camera_params.get("cy")
                scale = camera_params.get("scale")
                # Creating the camera matrix K
                K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                K = np.array(K)
                return K, scale
            else:
                raise ValueError("Camera parameters not found in cam_params.json")


# {root_dir}/ID-{scene_name}/ # ID is a random number assigned by HM3D
# ├── depth
#       ├── {scene_name}_000000.png
#       ├── {scene_name}_000001.png
#       ├── ...
# ├── pose
#       ├── {scene_name}_000000.txt
#       ├── {scene_name}_000001.txt
#       ├── ...
# ├── rgb
#       ├── {scene_name}_000000.png
#       ├── {scene_name}_000001.png
#       ├── ...


class HM3DDataset(BaseDataset):
    def __init__(self, root_dir, scene_name, canonicalize_world_axes=False):
        super().__init__(root_dir, scene_name)
        self.canonicalize_world_axes = bool(canonicalize_world_axes)
        self.scene_folder = self._resolve_scene_folder()
        self.rgb_dir = os.path.join(self.scene_folder, "rgb")
        self.depth_dir = os.path.join(self.scene_folder, "depth")
        self.pose_dir = self._resolve_pose_dir()

        if not os.path.isdir(self.rgb_dir):
            raise ValueError(f"Missing rgb directory: {self.rgb_dir}")
        if not os.path.isdir(self.depth_dir):
            raise ValueError(f"Missing depth directory: {self.depth_dir}")
        if not os.path.isdir(self.pose_dir):
            raise ValueError(f"Missing pose directory: {self.pose_dir}")

        self._rgb_paths, self._depth_paths, self._poses = self._build_aligned_sequences()
        if len(self._rgb_paths) == 0:
            raise ValueError(
                f"No aligned rgb/depth/pose frames found in {self.scene_folder}"
            )

    def _is_hm3d_scene_dir(self, path):
        return (
            os.path.isdir(os.path.join(path, "rgb"))
            and os.path.isdir(os.path.join(path, "depth"))
            and (
                os.path.isdir(os.path.join(path, "pose"))
                or os.path.isdir(os.path.join(path, "poses"))
            )
        )

    def _scene_dir_matches_key(self, path, scene_key):
        folder = os.path.basename(path)
        token = folder.split("-", 1)[-1] if "-" in folder else folder
        if folder == scene_key or folder.endswith(f"-{scene_key}") or token == scene_key:
            return True

        rgb_dir = os.path.join(path, "rgb")
        if os.path.isdir(rgb_dir):
            try:
                for name in os.listdir(rgb_dir):
                    stem, _ = os.path.splitext(name)
                    if stem == scene_key or stem.startswith(f"{scene_key}_"):
                        return True
            except OSError:
                pass
        return False

    def _scene_dir_score(self, path, scene_key):
        folder = os.path.basename(path)
        token = folder.split("-", 1)[-1] if "-" in folder else folder
        if folder == scene_key:
            return 4
        if folder.endswith(f"-{scene_key}") or token == scene_key:
            return 3

        rgb_dir = os.path.join(path, "rgb")
        if os.path.isdir(rgb_dir):
            try:
                for name in os.listdir(rgb_dir):
                    stem, _ = os.path.splitext(name)
                    if stem == scene_key or stem.startswith(f"{scene_key}_"):
                        return 2
            except OSError:
                pass
        return 1

    def _resolve_scene_folder(self):
        candidates = []
        if self.scene_name:
            if os.path.isabs(self.scene_name):
                candidates.append(self.scene_name)
            else:
                candidates.append(os.path.join(self.root_dir, self.scene_name))

        # Allow passing root_dir directly as a scene folder.
        candidates.append(self.root_dir)
        for c in candidates:
            if self._is_hm3d_scene_dir(c):
                return c

        # Search recursively under root_dir (covers .../hm3d/val/... style layouts).
        scene_key = str(self.scene_name)
        matched = []
        for dirpath, dirnames, _ in os.walk(self.root_dir):
            if not self._is_hm3d_scene_dir(dirpath):
                continue
            if self._scene_dir_matches_key(dirpath, scene_key):
                matched.append(dirpath)
        if matched:
            matched.sort(key=lambda path: self._scene_dir_score(path, scene_key), reverse=True)
            return matched[0]

        raise ValueError(f"No HM3D scene found for scene_name={self.scene_name} in {self.root_dir}")

    def _resolve_pose_dir(self):
        pose_dir = os.path.join(self.scene_folder, "pose")
        if os.path.isdir(pose_dir):
            return pose_dir
        pose_dir = os.path.join(self.scene_folder, "poses")
        if os.path.isdir(pose_dir):
            return pose_dir
        raise ValueError(f"No pose/poses directory found in {self.scene_folder}")

    @staticmethod
    def _list_files(folder):
        return sorted(
            f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
        )

    def _build_aligned_sequences(self):
        rgb_map = {}
        for name in self._list_files(self.rgb_dir):
            stem, _ = os.path.splitext(name)
            rgb_map[stem] = os.path.join(self.rgb_dir, name)

        depth_map = {}
        for name in self._list_files(self.depth_dir):
            stem, _ = os.path.splitext(name)
            depth_map[stem] = os.path.join(self.depth_dir, name)

        pose_map = {}
        camera_basis_change = np.eye(4, dtype=np.float32)
        camera_basis_change[1, 1] = -1.0
        camera_basis_change[2, 2] = -1.0
        world_basis_change = np.eye(4, dtype=np.float32)
        if self.canonicalize_world_axes:
            # Optional visualization-oriented canonicalization to
            # x-right, y-forward, z-up.
            world_basis_change[:3, :3] = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        for name in self._list_files(self.pose_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() != ".txt":
                continue
            full_path = os.path.join(self.pose_dir, name)
            values = np.loadtxt(full_path, dtype=np.float64)
            if values.size != 16:
                continue
            pose = values.reshape((4, 4)).astype(np.float32)
            # First convert Habitat camera axes to the convention used by our
            # backprojection. Optionally rotate the world frame into a more
            # canonical x-right / y-forward / z-up basis for visualization.
            pose_map[stem] = world_basis_change @ pose @ camera_basis_change

        common_stems = sorted(set(rgb_map.keys()) & set(depth_map.keys()) & set(pose_map.keys()))
        rgb_paths = [rgb_map[s] for s in common_stems]
        depth_paths = [depth_map[s] for s in common_stems]
        poses = [pose_map[s] for s in common_stems]
        return rgb_paths, depth_paths, poses
    
    def get_rgb(self):
        return self._rgb_paths
    
    def get_depth(self): 
        return self._depth_paths
    
    def get_poses(self):
        return self._poses

    def get_intrinsics(self):
        # HM3D uses 90deg horizontal FOV; derive intrinsics from actual image size.
        w, h = Image.open(self._depth_paths[0]).size
        H, W = h, w
        hfov = 90 * np.pi / 180
        vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
        fx = W / (2.0 * np.tan(hfov / 2.0))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        cx = W / 2
        cy = H / 2
        depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        scale = 1000.0  # depth is saved in mm
        return depth_camera_matrix, scale


class ScanNetPPDataset(BaseDataset):
    def __init__(self, root_dir, scene_name):
        super().__init__(root_dir, scene_name)
        self.scene_dir = self._resolve_scene_dir()
        self.iphone_dir = os.path.join(self.scene_dir, "iphone")
        self.rgb_dir = os.path.join(self.iphone_dir, "rgb")
        self.colmap_dir = os.path.join(self.iphone_dir, "colmap")
        self.images_txt = os.path.join(self.colmap_dir, "images.txt")
        self.cameras_txt = os.path.join(self.colmap_dir, "cameras.txt")

        if not os.path.isdir(self.rgb_dir):
            raise ValueError(f"Missing RGB directory: {self.rgb_dir}")
        if not os.path.isfile(self.images_txt):
            raise ValueError(f"Missing COLMAP poses file: {self.images_txt}")
        if not os.path.isfile(self.cameras_txt):
            raise ValueError(f"Missing COLMAP camera file: {self.cameras_txt}")

        self.depth_dir = self._select_depth_dir()
        self._rgb_paths, self._depth_paths, self._poses = self._build_aligned_sequences()
        if len(self._rgb_paths) == 0:
            raise ValueError(
                f"No aligned RGB/Depth/Pose frames found under scene: {self.scene_dir}"
            )

        self._intrinsics = self._load_intrinsics()
        self._depth_scale = self._infer_depth_scale()

    def _resolve_scene_dir(self):
        # Support both: (root_dir=/path/to/scannetpp, scene_name=fd361ab85f)
        # and (root_dir=/path/to/scannetpp/fd361ab85f, scene_name=anything/empty).
        candidates = []

        if self.scene_name:
            if os.path.isabs(self.scene_name):
                candidates.append(self.scene_name)
            else:
                candidates.append(os.path.join(self.root_dir, self.scene_name))
        candidates.append(self.root_dir)

        for candidate in candidates:
            iphone_dir = os.path.join(candidate, "iphone")
            if os.path.isdir(iphone_dir):
                return candidate

        raise ValueError(
            f"Could not resolve ScanNet++ scene directory from root_dir={self.root_dir}, "
            f"scene_name={self.scene_name}. Expected an 'iphone' folder."
        )

    def _list_files(self, folder):
        return sorted(
            f
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        )

    def _get_image_size(self, image_path):
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height

    def _select_depth_dir(self):
        depth_candidates = [
            os.path.join(self.iphone_dir, "depth_resampled"),
            os.path.join(self.iphone_dir, "depth"),
            os.path.join(self.iphone_dir, "resized_depth"),
        ]
        existing = [d for d in depth_candidates if os.path.isdir(d)]
        if not existing:
            raise ValueError(
                f"No depth directory found. Tried: {depth_candidates}"
            )
        # Prefer RGB-aligned resampled depth when available.
        for preferred_name in ("depth_resampled", "depth", "resized_depth"):
            for depth_dir in existing:
                if os.path.basename(depth_dir) == preferred_name:
                    return depth_dir
        return existing[0]

    def _qvec_to_rotmat(self, qvec):
        qw, qx, qy, qz = qvec
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    def _load_pose_map(self):
        pose_map = {}
        with open(self.images_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # COLMAP images.txt alternates image metadata lines with points2D lines.
                if len(parts) < 10:
                    continue

                qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
                tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
                image_name = parts[9]

                rot = self._qvec_to_rotmat(qvec)
                T_w2c = np.eye(4, dtype=np.float64)
                T_w2c[:3, :3] = rot
                T_w2c[:3, 3] = tvec
                T_c2w = np.linalg.inv(T_w2c)
                pose_map[image_name] = T_c2w
        return pose_map

    def _build_aligned_sequences(self):
        pose_map = self._load_pose_map()

        rgb_map = {}
        for name in self._list_files(self.rgb_dir):
            stem, ext = os.path.splitext(name)
            rgb_map[stem] = os.path.join(self.rgb_dir, stem + ext)

        depth_map = {}
        for name in self._list_files(self.depth_dir):
            stem, ext = os.path.splitext(name)
            depth_map[stem] = os.path.join(self.depth_dir, stem + ext)

        rgb_paths = []
        depth_paths = []
        poses = []
        for image_name in sorted(pose_map.keys()):
            stem = os.path.splitext(image_name)[0]
            rgb_path = rgb_map.get(stem)
            depth_path = depth_map.get(stem)
            if rgb_path is None or depth_path is None:
                continue
            rgb_paths.append(rgb_path)
            depth_paths.append(depth_path)
            poses.append(pose_map[image_name].astype(np.float32))

        return rgb_paths, depth_paths, poses

    def _load_intrinsics(self):
        camera_line = None
        with open(self.cameras_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                camera_line = line
                break
        if camera_line is None:
            raise ValueError(f"No valid camera entry found in {self.cameras_txt}")

        parts = camera_line.split()
        if len(parts) < 8:
            raise ValueError(f"Invalid camera line in {self.cameras_txt}: {camera_line}")

        model = parts[1]
        width = float(parts[2])
        height = float(parts[3])
        params = [float(v) for v in parts[4:]]

        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = params[0]
            fy = params[0]
            cx = params[1]
            cy = params[2]
        elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx = params[0]
            fy = params[1]
            cx = params[2]
            cy = params[3]
        else:
            raise ValueError(f"Unsupported camera model in ScanNet++ loader: {model}")

        # Keep intrinsics in RGB resolution coordinates. Depth can be resized to RGB later.
        rgb_w, rgb_h = self._get_image_size(self._rgb_paths[0])
        sx = rgb_w / width
        sy = rgb_h / height
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

        intrinsics = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return intrinsics

    def _infer_depth_scale(self):
        depth_sample = np.array(Image.open(self._depth_paths[0]))
        if depth_sample.dtype == np.uint16 and float(depth_sample.max()) > 255.0:
            return 1000.0
        return 1.0

    def get_rgb(self):
        return self._rgb_paths

    def get_depth(self):
        return self._depth_paths

    def get_poses(self):
        return self._poses

    def get_intrinsics(self):
        return self._intrinsics, self._depth_scale


class ResizedDatasetProxy(BaseDataset):
    def __init__(self, dataset: BaseDataset, input_image_max_side: int = -1):
        super().__init__(dataset.root_dir, dataset.scene_name)
        self.dataset = dataset
        self.input_image_max_side = int(input_image_max_side)
        self._prepared = False
        self._rgb_paths = None
        self._depth_paths = None
        self._original_image_size = None
        self._resized_image_size = None

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    @staticmethod
    def _get_image_size(path: str) -> Tuple[int, int]:
        with Image.open(path) as img:
            return img.size

    def _compute_target_size(self, width: int, height: int) -> Tuple[int, int]:
        if self.input_image_max_side <= 0:
            return int(width), int(height)
        max_side = max(int(width), int(height))
        if max_side <= 0 or max_side == int(self.input_image_max_side):
            return int(width), int(height)
        scale = float(self.input_image_max_side) / float(max_side)
        new_width = max(1, int(round(float(width) * scale)))
        new_height = max(1, int(round(float(height) * scale)))
        return new_width, new_height

    def _cache_root(self) -> str:
        payload = (
            f"{self.dataset.__class__.__name__}|{self.root_dir}|{self.scene_name}|"
            f"{self.input_image_max_side}"
        ).encode("utf-8")
        token = hashlib.sha1(payload).hexdigest()[:12]
        return os.path.join(
            "/tmp",
            "lexisg_input_resize_cache",
            self.dataset.__class__.__name__,
            token,
            f"maxside_{int(self.input_image_max_side)}",
        )

    def _resize_image_paths(self, paths, kind: str, target_size: Tuple[int, int], is_depth: bool):
        save_dir = os.path.join(self._cache_root(), kind)
        os.makedirs(save_dir, exist_ok=True)
        resized_paths = []
        for src in paths:
            filename = os.path.basename(src)
            dst = os.path.join(save_dir, filename)
            resized_paths.append(dst)
            if os.path.exists(dst):
                continue
            with Image.open(src) as img:
                if is_depth:
                    arr = np.array(img)
                    out = Image.fromarray(arr)
                    if out.size != target_size:
                        out = out.resize(target_size, resample=Image.Resampling.NEAREST)
                    out.save(dst)
                else:
                    out = img.convert("RGB")
                    if out.size != target_size:
                        out = out.resize(target_size, resample=Image.Resampling.LANCZOS)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in {".jpg", ".jpeg"}:
                        out.save(dst, quality=95)
                    else:
                        out.save(dst)
        return resized_paths

    def _ensure_prepared(self):
        if self._prepared:
            return
        self._rgb_paths = list(self.dataset.get_rgb())
        self._depth_paths = list(self.dataset.get_depth()) if hasattr(self.dataset, "get_depth") else None
        if not self._rgb_paths:
            self._prepared = True
            return

        orig_width, orig_height = self._get_image_size(self._rgb_paths[0])
        self._original_image_size = (int(orig_width), int(orig_height))
        target_size = self._compute_target_size(orig_width, orig_height)
        self._resized_image_size = (int(target_size[0]), int(target_size[1]))

        if self.input_image_max_side > 0 and self._resized_image_size != self._original_image_size:
            logger.info(
                f"Dataset resize enabled: dataset={self.dataset.__class__.__name__} "
                f"path={self.root_dir} rgb_depth={self._original_image_size[0]}x{self._original_image_size[1]} "
                f"-> {self._resized_image_size[0]}x{self._resized_image_size[1]}"
            )
            self._rgb_paths = self._resize_image_paths(
                self._rgb_paths, kind="rgb", target_size=self._resized_image_size, is_depth=False,
            )
            if self._depth_paths is not None:
                self._depth_paths = self._resize_image_paths(
                    self._depth_paths, kind="depth", target_size=self._resized_image_size, is_depth=True,
                )
        self._prepared = True

    def get_rgb(self):
        self._ensure_prepared()
        return self._rgb_paths

    def get_depth(self):
        self._ensure_prepared()
        return self._depth_paths

    def get_poses(self):
        return self.dataset.get_poses()

    def get_intrinsics(self):
        K, depth_scale = self.dataset.get_intrinsics()
        self._ensure_prepared()
        if self._original_image_size is None or self._resized_image_size is None:
            return K, depth_scale
        orig_width, orig_height = self._original_image_size
        new_width, new_height = self._resized_image_size
        if (orig_width, orig_height) == (new_width, new_height):
            return K, depth_scale
        sx = float(new_width) / float(orig_width)
        sy = float(new_height) / float(orig_height)
        K_scaled = np.asarray(K, dtype=np.float32).copy()
        K_scaled[0, 0] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx
        K_scaled[1, 2] *= sy
        return K_scaled, depth_scale


BENCHMARK_DATASETS = {
    "replica": ReplicaDataset,
    "hm3d": HM3DDataset,
    "scannetpp": ScanNetPPDataset,
}


def get_dataset(root_dir, input_image_max_side=-1, name=None, scene_name=None, **dataset_kwargs):
    if name is not None and str(name).lower() in BENCHMARK_DATASETS:
        dataset = BENCHMARK_DATASETS[str(name).lower()](root_dir, scene_name, **dataset_kwargs)
    elif name is not None and str(name).lower() != "custom":
        raise ValueError(f"Unknown dataset name: {name}")
    else:
        dataset = CustomDataset(root_dir, **dataset_kwargs)
    if int(input_image_max_side) > 0:
        return ResizedDatasetProxy(dataset, input_image_max_side=int(input_image_max_side))
    return dataset


def lookup_priors(rgb_batch, depth_by_rgb=None, pose_by_rgb=None):
    """Map a batch of RGB paths to (depth_batch, poses) for MapAny.run, using the
    per-frame dicts built in main.py. Returns None for either that isn't available."""
    depth_batch = [depth_by_rgb[p] for p in rgb_batch] if depth_by_rgb else None
    poses = [pose_by_rgb[p] for p in rgb_batch] if pose_by_rgb else None
    return depth_batch, poses
