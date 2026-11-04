import os

import hydra
import numpy as np
import yaml
from loguru import logger

from lexisg.utils.config import get_path, get_object_segmentation_value, load_object_segmentation_service_config
from lexisg.utils.data_loader import get_dataset
from lexisg.tasks.object_segmentation import ObjectSegmentationService


class ObjectSegmentationPipeline:
    STAGE_ORDER = ("detect", "track", "merge", "clip", "eval")
    FRAME_SAMPLING_MODES = ("count", "stride")

    def __init__(self, config):
        self.config = load_object_segmentation_service_config(config)
        self.dataset = str(self.config.get("dataset") or "custom")
        self.data_path = str(self.config["data_path"])
        self.scene_name = str(self.config.get("scene_name") or os.path.basename(self.data_path.rstrip("/\\")))
        self.subsample = int(self.config.get("subsample", 1))
        self.input_image_max_side = int(self.config.get("input_image_max_side", -1))
        self.hm3d_canonicalize_world_axes = bool(self.config.get("hm3d_canonicalize_world_axes", False))
        self.num_uniform_frames = int(self.config["num_uniform_frames"])
        self.frame_sampling_mode = str(self.config.get("frame_sampling_mode", "count")).strip().lower()
        self.uniform_frame_stride = int(self.config.get("uniform_frame_stride", 0) or 0)
        self.log_dir = str(self.config["log_dir"])
        self.color_seed = int(get_object_segmentation_value(self.config, "color_seed", 0))

        if self.frame_sampling_mode not in self.FRAME_SAMPLING_MODES:
            raise ValueError(
                f"Unsupported frame_sampling_mode={self.frame_sampling_mode!r}. "
                f"Expected one of {self.FRAME_SAMPLING_MODES}."
            )

        os.makedirs(self.log_dir, exist_ok=True)
        logger.add(os.path.join(self.log_dir, "pipeline.log"), rotation="50 MB")
        with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)
        self.object_segmentation = ObjectSegmentationService(
            config=self.config,
            dataset=self.dataset,
            scene_name=self.scene_name,
            data_path=self.data_path,
            artifact_root=self.log_dir,
            color_seed=self.color_seed,
        )

    def _load_dataset_context(self):
        dataset_kwargs = {}
        if self.dataset.lower() == "hm3d":
            dataset_kwargs["canonicalize_world_axes"] = self.hm3d_canonicalize_world_axes
        dataset = get_dataset(
            self.data_path,
            input_image_max_side=self.input_image_max_side,
            name=self.dataset,
            scene_name=self.scene_name if self.dataset.lower() != "custom" else None,
            **dataset_kwargs,
        )
        rgb_paths = dataset.get_rgb()[:: self.subsample]
        depth_paths = dataset.get_depth()[:: self.subsample]
        poses = dataset.get_poses()[:: self.subsample]
        intrinsics, depth_scale = dataset.get_intrinsics()

        assert len(rgb_paths) == len(depth_paths), "RGB and depth length mismatch."
        assert len(rgb_paths) == len(poses), "RGB and poses length mismatch."
        assert len(rgb_paths) > 0, "No frames found."
        return {
            "dataset_obj": dataset,
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths,
            "poses": poses,
            "intrinsics": intrinsics,
            "depth_scale": depth_scale,
        }

    @staticmethod
    def _uniform_sample_ids(total: int, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("num_uniform_frames must be > 0.")
        if n >= total:
            return np.arange(total, dtype=np.int32)
        return np.linspace(0, total - 1, num=n, dtype=np.int32)

    @staticmethod
    def _stride_sample_ids(total: int, stride: int) -> np.ndarray:
        if stride <= 0:
            raise ValueError("uniform_frame_stride must be > 0 when frame_sampling_mode='stride'.")
        return np.arange(0, total, stride, dtype=np.int32)

    def _sampled_frame_ids(self, total: int) -> np.ndarray:
        if self.frame_sampling_mode == "count":
            return self._uniform_sample_ids(total, self.num_uniform_frames)
        return self._stride_sample_ids(total, self.uniform_frame_stride)

    def _build_sampling_plan(self, ctx: dict, frame_ids=None):
        if frame_ids is None:
            frame_ids = np.arange(len(ctx["rgb_paths"]), dtype=np.int32)
        else:
            frame_ids = np.asarray(frame_ids, dtype=np.int32)
        sampled_local_ids = self._sampled_frame_ids(len(frame_ids))
        sampled_frame_ids = frame_ids[sampled_local_ids]
        return {
            "frame_ids": frame_ids,
            "frame_sampling_mode": self.frame_sampling_mode,
            "uniform_frame_stride": int(self.uniform_frame_stride),
            "num_uniform_frames": int(self.num_uniform_frames),
            "sampled_frame_ids": sampled_frame_ids,
            "sampled_frame_ids_local": sampled_local_ids,
        }

    def run_batch(self):
        ctx = self._load_dataset_context()
        plan = self._build_sampling_plan(ctx)
        return self.object_segmentation.run_from_context(
            ctx=ctx,
            plan=plan,
            include_clip=True,
            include_eval=True,
        )


@hydra.main(
    version_base=None,
    config_path=os.path.join(get_path(), "object_segmentation"),
    config_name="minimal",
)
def main(config):
    p = ObjectSegmentationPipeline(config)
    p.run_batch()


if __name__ == "__main__":
    main()
