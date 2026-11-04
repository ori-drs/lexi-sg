import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

from contextlib import nullcontext

import torch 
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs

import numpy as np
from PIL import Image

class MapAny:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
   
    def _device_context(self):
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            return torch.cuda.device(self.device)
        return nullcontext()
    
    def run(self, batch, depth_batch=None, intrinsics=None, poses=None, conf_threshold=0.0, depth_scale=None):
        """Run MapAnything on a batch of image paths.

        depth_batch: optional per-image depth paths (same order as batch); needs
            intrinsics and depth_scale (divisor to metres).
        poses: optional per-image 4x4 cam-to-world poses, OpenCV convention.
        Any inputs given are passed to the model as priors; the rest is predicted.
        """
        with self._device_context():
            views = self._preprocess_images(batch, depth_batch, intrinsics, depth_scale, poses)
            predictions = self._forward(views, conf_threshold)
        return predictions

    def _load_model(self):
        print("Loading MapAnything")
        with self._device_context():
            return MapAnything.from_pretrained("facebook/map-anything", revision="a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa").to(self.device)

    def _preprocess_images(self, image_batch, depth_batch=None, intrinsics=None, depth_scale=None, poses=None):
        has_intrinsics = intrinsics is not None and np.asarray(intrinsics).size > 0
        has_depth = bool(depth_batch)
        has_poses = poses is not None and len(poses) > 0
        if has_depth and not has_intrinsics:
            raise ValueError("Depth input requires intrinsics (see the dataset config's `inputs` block).")
        if has_depth and len(depth_batch) != len(image_batch):
            raise ValueError(f"{len(depth_batch)} depth maps for {len(image_batch)} images")
        if has_poses and len(poses) != len(image_batch):
            raise ValueError(f"{len(poses)} poses for {len(image_batch)} images")
        if has_depth and not depth_scale:
            raise ValueError("depth_scale must be set when depth input is used")

        K = torch.as_tensor(np.asarray(intrinsics), dtype=torch.float32) if has_intrinsics else None
        views = []
        for i, img_path in enumerate(image_batch):
            view = {"img": Image.open(img_path)}
            if has_intrinsics:
                view["intrinsics"] = K
            if has_depth:
                depth = np.asarray(Image.open(depth_batch[i]), dtype=np.float32) / float(depth_scale)
                view["depth_z"] = torch.as_tensor(depth, dtype=torch.float32)
            if has_poses:
                view["camera_poses"] = torch.as_tensor(np.asarray(poses[i]), dtype=torch.float32)
            if has_depth or has_poses:
                view["is_metric_scale"] = torch.tensor([True], device=self.device)
            views.append(view)

        # preprocess_inputs yields depth_z as (1, H, W); inference adds the channel dim itself.
        return preprocess_inputs(views)

    def _forward(self, views, conf_threshold):
        with self._device_context():
            output = self.model.infer(
                views,                            # Input views
                memory_efficient_inference=True, # Trades off speed for more views (up to 2000 views on 140 GB)
                use_amp=False,                    # Disabled for reproducibility (bf16 is non-deterministic)
                amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
                apply_mask=True,                  # Apply masking to dense geometry outputs
                mask_edges=True,                  # Remove edge artifacts by using normals and depth
                apply_confidence_mask=False,      # Filter low-confidence regions
                confidence_percentile=15,         # Remove bottom 10 percentile confidence pixels
                ignore_calibration_inputs=False,
                ignore_depth_inputs=False,
                ignore_pose_inputs=False,
                ignore_depth_scale_inputs=False,
                ignore_pose_scale_inputs=False,
            )

        predictions = {}
        intrinsics = []
        camera_poses = []
        depth_images = []
        depth_conf = []
        images = []
        for pred in output:
            depth_images.append(pred["depth_z"])

            intrinsic = pred["intrinsics"]
            intrinsics.append(intrinsic)

            camera_pose = pred["camera_poses"]
            camera_poses.append(camera_pose[0].detach().cpu().numpy())

            confidence = pred["conf"]
            depth_conf.append(confidence)

            images.append(pred["img_no_norm"])
        
        predictions["intrinsic"] = torch.vstack(intrinsics)
        predictions["extrinsic"] = np.array(camera_poses)
        predictions["depth"] = torch.vstack(depth_images)
        predictions["depth_conf"] = torch.vstack(depth_conf)
        predictions["images"] = torch.vstack(images)

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().detach().numpy() # remove batch dimension and convert to numpy

        predictions["images"] = (predictions["images"] * 255).clip(0, 255).astype(np.uint8) # B, H, W, C
        predictions["depth"] = np.where(predictions["depth_conf"] < conf_threshold, 0.0, predictions["depth"].squeeze(3)) # B, H, W

        for key in predictions.keys():
            if isinstance(predictions[key], np.ndarray):
                predictions[key] = np.ascontiguousarray(predictions[key])
        
        return predictions
