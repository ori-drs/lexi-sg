from lexisg.models.dino_handler import Dino
from lexisg.structures.graph import Graph
from lexisg.structures.node import RoomNode
from lexisg.structures.edge import Edge
from lexisg.tasks.room_segmentation import RoomSegmentation
from lexisg.tasks.loop_closure import LoopClosure
from lexisg.tasks.object_segmentation import ObjectSegmentationService
from lexisg.tasks.object_segmentation import cross_room
from lexisg.tasks.optimiser import Optimiser
from lexisg.tasks.visualiser import Visualiser
from lexisg.utils.data_loader import get_dataset
from lexisg.utils.config import get_path, get_object_segmentation_value, load_object_segmentation_service_config

import numpy as np
import open3d as o3d

import logging
import os
import yaml
import torch

import argparse
import shutil
from tqdm import tqdm
from loguru import logger


def _parse_override_value(raw_value):
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _apply_config_override(config, override):
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")

    key_path, raw_value = override.split("=", 1)
    key_path = key_path.strip()
    if not key_path:
        raise ValueError(f"Invalid override '{override}'. Key cannot be empty.")

    value = _parse_override_value(raw_value)
    current = config
    keys = [part.strip() for part in key_path.split(".")]

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


class LEXISG:
    def __init__(self, config):
        self.config = config
        self.model_devices = dict(self.config.get("model_devices", {}))
        self.reconstruction_device = self._resolve_device(self.model_devices.get("reconstruction"))
        self.semantic_device = self._resolve_device(self.model_devices.get("semantic"))

        from lexisg.models.mapa_handler import MapAny
        self.model = MapAny(device=self.reconstruction_device)
        
        self.data_path = self.config["data_path"]

        self.sem_model = Dino(device=self.semantic_device)

        self.batch_size = self.config["reconstruction"]["batch_size"]
        # Voxel size of each room's stored/saved point cloud. Independent of
        # loop_closure.voxel_size, which only controls the overlap comparison
        # (_compute_pcd_overlap re-downsamples to that on the fly).
        self.pcd_voxel_size = float(
            self.config["reconstruction"].get("pcd_voxel_size", self.config["loop_closure"]["voxel_size"])
        )
        self.conf_threshold = float(self.config["reconstruction"].get("conf_threshold", 0.0))
        self.boundary_pairs = self.config["room_segmentation"]["boundary_pairs"]
        self.room_prompt_list = self.config["room_segmentation"]["room_prompt_list"]
        self.room_txt_feat = self.sem_model.get_txt_feat(self.room_prompt_list)


        self.graph = Graph()
        self.room_segmentation = RoomSegmentation(self.model, self.config)
        self.loop_handler = LoopClosure(self.model, self.config)
        self.optimiser = Optimiser()
        self.visualiser = Visualiser()

        self.prev_node = None
        self.rgb_paths = None
        self.intrinsics = None
        self.depth_scale = None
        self.depth_paths = None
        self.poses = None
        # Optional depth / pose priors for MapAnything (see config `inputs` block).
        self.inputs_cfg = dict(self.config.get("inputs", {}))
        self.accumulated_feats = []
        self.run_object_segmentation = self.config.get("object_segmentation", {}).get("enabled", True)
        self.cross_room_cfg = dict(self.config.get("object_segmentation", {}).get("cross_room", {}) or {})
        self.object_feature_encoder = str(
            self.config.get("object_segmentation", {}).get("feature_encoder", "dino")
        ).lower()
        if self.object_feature_encoder not in ("dino", "clip"):
            raise ValueError(
                f"object_segmentation.feature_encoder must be 'dino' or 'clip', got {self.object_feature_encoder!r}"
            )
        self.object_segmentation_config = load_object_segmentation_service_config(self.config)
        self.object_segmentation_root = os.path.join(
            self.config["output_folder"],
            "object_segmentation",
        )
        os.makedirs(self.object_segmentation_root, exist_ok=True)

    def run_batch(self): 

        dataset = get_dataset(
            self.data_path,
            intrinsics=self.inputs_cfg.get("intrinsics"),
            depth_scale=self.inputs_cfg.get("depth_scale", 1.0),
            pose_convention=self.inputs_cfg.get("pose_convention", "opencv"),
            poses_are_world_to_cam=self.inputs_cfg.get("poses_are_world_to_cam", False),
        )
        self.intrinsics, self.depth_scale = self._load_dataset_intrinsics(dataset)
        self.room_segmentation.intrinsics = self.intrinsics
        self.room_segmentation.depth_scale = self.depth_scale
        self.room_segmentation.conf_threshold = self.conf_threshold
        self.loop_handler.intrinsics = self.intrinsics
        self.loop_handler.depth_scale = self.depth_scale
        self.loop_handler.conf_threshold = self.conf_threshold
        self.rgb_paths = sorted(dataset.get_rgb())
        self.depth_paths, self.poses = self._load_optional_inputs(dataset)
        depth_by_rgb = dict(zip(self.rgb_paths, self.depth_paths)) if self.depth_paths else None
        pose_by_rgb = dict(zip(self.rgb_paths, self.poses)) if self.poses else None
        for task in (self.room_segmentation, self.loop_handler):
            task.depth_by_rgb = depth_by_rgb
            task.pose_by_rgb = pose_by_rgb
        self.loop_handler.transition_frames = self.room_segmentation.transition_frames

        # Extract room batches
        for img_idx, img_path in tqdm(enumerate(self.rgb_paths), total=len(self.rgb_paths), desc="Frames"):
            feat = self.sem_model.get_img_txt_feat([img_path])
            self.accumulated_feats.append(feat)

            logits = self.sem_model.get_similarity(self.room_txt_feat, feat)
            room_label = self.room_prompt_list[np.argmax(logits)]

            threshold_detected, batch, current_key = self.room_segmentation.estimate_room_batch(
                                            img_path, 
                                            logits, 
                                            room_label
                                        )


            if threshold_detected:
                # Process batch and check for loop closures
                active_idx = self._process_room_batch(batch, current_key, self.accumulated_feats, self.rgb_paths)
                merged_ids = self.loop_handler.run_pcd_overlap_check(
                    self.graph, self.accumulated_feats, self.rgb_paths, self.optimiser
                )
                updated_nodes = list(set([active_idx] + merged_ids))
                if self.run_object_segmentation:
                    self._run_object_segmentation_for_updated_nodes(updated_nodes)
                
        
        #  Finalise last unfinished batch 
        threshold_detected, batch, current_key = self.room_segmentation.estimate_room_batch(
            None, None, None, end_of_sequence=True
        )

        if threshold_detected and batch is not None:
            print("Final segment detected — processing remaining frames.")
            active_idx = self._process_room_batch(batch[:-1], current_key, self.accumulated_feats, self.rgb_paths)
            if self.run_object_segmentation:
                self._run_object_segmentation_for_updated_nodes([active_idx])


        output_folder = self.config["output_folder"]
        self.visualiser.save(self.graph, output_folder, save_segmented=self.run_object_segmentation)
        if self.run_object_segmentation and self.cross_room_cfg.get("enabled", False):
            self.visualiser.save_cross_room(output_folder, cross_room.associate(self.graph, self.cross_room_cfg))

        # Print all nodes
        print("Nodes:")
        for idx, node in self.graph.room_nodes.items():
            print(f"Node idx: {node.idx}, label: {node.label}")

        # Print all unique edges
        print("\nEdges:")
        seen_edges = set()
        for (src, tgt) in self.graph.room_to_room_edge.keys():
            # Ensure each edge is printed once (ignore direction)
            edge_key = tuple(sorted((src, tgt)))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                print(f"Edge: {edge_key[0]} -> {edge_key[1]}")
                                
                

    def _even_subsample(self, lst, n=32, edge_count=5):
        if len(lst) <= n:
            return lst  

        first = list(range(edge_count))
        last = list(range(len(lst) - edge_count, len(lst)))
        
        remaining = n - 2 * edge_count
        if remaining <= 0:
            idx = np.linspace(0, len(lst) - 1, n, dtype=int)
            return [lst[i] for i in idx]
        
        middle_start = edge_count
        middle_end = len(lst) - edge_count
        
        middle_idx = np.linspace(middle_start, middle_end - 1, remaining, dtype=int)
        idx = np.concatenate([first, middle_idx, last])
        
        return [lst[i] for i in idx]
    
    def _process_room_batch(self, batch, current_key, accumulated_feats, rgb_paths):
        subsampled = self._even_subsample(batch, n=self.batch_size)
        subsampled_idx = [rgb_paths.index(p) for p in subsampled]
        subsampled_feats = [accumulated_feats[i] for i in subsampled_idx]
        depth_batch = [self.depth_paths[i] for i in subsampled_idx] if self.depth_paths else None
        poses = [self.poses[i] for i in subsampled_idx] if self.poses else None

        predictions = self.model.run(
            batch=subsampled,
            depth_batch=depth_batch,
            intrinsics=self.intrinsics,
            poses=poses,
            conf_threshold=self.conf_threshold,
            depth_scale=self.depth_scale,
        )

        ref_pose = predictions["extrinsic"][0]
        local_poses = [np.linalg.inv(ref_pose) @ p for p in predictions["extrinsic"]]

        # Create new room node
        new_node = RoomNode(
            idx = current_key[1],
            label = current_key[0],
            predictions = predictions,
            img_paths = subsampled, 
            ref_pose = ref_pose,
            local_poses = local_poses,
            feats = subsampled_feats,
            transition_frames = self.room_segmentation.transition_frames,
        )

        self.graph.add_node(new_node)
        new_node.build_pcd(voxel_size=self.pcd_voxel_size)

        # Estimate Boundary Transforms
        if new_node.idx > 0:
            # Add sequential boundary edges
            rel_transforms, pair_data = self.room_segmentation.estimate_boundary_pairs(
                self.prev_node,
                new_node,
                n_pairs=self.boundary_pairs
            )

            for transform, pair in zip(rel_transforms, pair_data):
                edge = Edge(
                    source=self.prev_node.idx,
                    target=new_node.idx,
                    edge_type="odom",
                    relative_transform=transform,
                    boundary_pairs = pair
                )
                self.graph.add_edge(edge)


            # # Search for loop closures
            success, self.prev_node = self.loop_handler.run(new_node, self.graph, accumulated_feats, self.rgb_paths)
        
        else:
            self.prev_node = new_node
            self.loop_handler.add_to_database(new_node)

        # Optimise
        self.optimiser.optimise_sim3(self.graph)
        return self.prev_node.idx

    def _make_object_segmentation_service(self, node: RoomNode):
        artifact_root = os.path.join(self.object_segmentation_root, f"room_{int(node.idx):04d}")
        os.makedirs(artifact_root, exist_ok=True)
        return ObjectSegmentationService(
            config=self.object_segmentation_config,
            scene_name=f"room_{int(node.idx):04d}",
            data_path=self.data_path,
            artifact_root=artifact_root,
            color_seed=int(get_object_segmentation_value(self.object_segmentation_config, "color_seed", 0)),
            feature_model=self.sem_model if self.object_feature_encoder == "dino" else None,
            feature_encoder=self.object_feature_encoder,
        )

    def _save_room_pointcloud_artifact(self, node: RoomNode, artifact_root: str):
        if node.pcd is None:
            return
        room_pcd = o3d.geometry.PointCloud(node.pcd)
        if node.optimised_ref_pose is not None:
            room_pcd.transform(np.asarray(node.optimised_ref_pose, dtype=np.float64))
        o3d.io.write_point_cloud(
            os.path.join(artifact_root, "room_pointcloud_optimized.pcd"),
            room_pcd,
        )

    def _run_object_segmentation_for_updated_nodes(self, updated_nodes):
        for node_id in sorted(set(int(idx) for idx in updated_nodes)):
            node = self.graph.room_nodes.get(int(node_id))
            if node is None:
                continue
            service = self._make_object_segmentation_service(node)
            self._save_room_pointcloud_artifact(node, service.artifact_root)
            node.object_segmentation = service.run_from_predictions(
                predictions=node.predictions,
            )

    def _load_optional_inputs(self, dataset):
        """Load per-frame depth paths and poses if enabled in config and present on
        disk, aligned with self.rgb_paths. Either is skipped (with a warning) if
        missing or if its count doesn't match the RGB frames."""
        n = len(self.rgb_paths)
        depth_paths = None
        poses = None

        if self.inputs_cfg.get("depth", False):
            candidates = sorted(dataset.get_depth() or [])
            if not candidates:
                logger.warning("inputs.depth is enabled but no depth/ directory was found; ignoring.")
            elif len(candidates) != n:
                logger.warning(f"inputs.depth: {len(candidates)} depth maps for {n} RGB frames; ignoring depth.")
            elif self.intrinsics is None:
                raise ValueError("inputs.depth requires inputs.intrinsics ([fx, fy, cx, cy]) to be set.")
            else:
                depth_paths = candidates
                logger.info(f"Using {n} input depth maps (depth_scale={self.depth_scale}).")

        if self.inputs_cfg.get("poses", False):
            candidates = dataset.get_poses()
            if not candidates:
                logger.warning("inputs.poses is enabled but no pose/ directory was found; ignoring.")
            elif len(candidates) != n:
                logger.warning(f"inputs.poses: {len(candidates)} poses for {n} RGB frames; ignoring poses.")
            else:
                poses = candidates
                logger.info(f"Using {n} input poses (convention={self.inputs_cfg.get('pose_convention', 'opencv')}).")

        return depth_paths, poses

    def _load_dataset_intrinsics(self, dataset):
        if not hasattr(dataset, "get_intrinsics"):
            return None, None
        try:
            return dataset.get_intrinsics()
        except NotImplementedError:
            logging.info(
                "Dataset %s does not define intrinsics; continuing without calibration inputs.",
                dataset.__class__.__name__,
            )
        except AttributeError:
            logging.info(
                "Dataset %s does not expose calibration inputs; continuing without them.",
                dataset.__class__.__name__,
            )
        return None, None

    def _resolve_device(self, configured_device):
        if configured_device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(configured_device, int):
            return f"cuda:{configured_device}"
        configured = str(configured_device).strip()
        if configured.isdigit():
            return f"cuda:{configured}"
        return configured
    
def main():
    default_config = os.path.join(get_path(), "datasets", "aod.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config, help="Path to YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted keys, e.g. reconstruction.method=mapanything",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="After the run, write <output_folder>/scene_graph.rrd for Rerun (see visualise.py)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for override in args.overrides:
        _apply_config_override(config, override)

    log_level = str(config.get("log_level", "WARNING")).upper()
    log_level_int = getattr(logging, log_level, logging.WARNING)
    logging.basicConfig(level=log_level_int, format="%(levelname)s: %(message)s")
    logging.getLogger().setLevel(log_level_int)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level, colorize=True)

    output_folder = config.get("output_folder", "output")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    p = LEXISG(config)
    p.run_batch()

    if args.visualise:
        from visualise import build_rrd
        build_rrd(output_folder)

if __name__ == "__main__":
    main()
