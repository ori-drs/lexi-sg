import logging
import numpy as np
import open3d as o3d

from lexisg.structures.node import RoomNode
from lexisg.structures.edge import Edge
from lexisg.utils.data_loader import lookup_priors

logger = logging.getLogger(__name__)


class LoopClosure:
    def __init__(self, model, config):
        self.database = {}
        self.model = model

        self.room_threshold = config["loop_closure"]["room_threshold"]
        self.image_threshold = config["loop_closure"]["image_threshold"]
        self.iou_threshold = config["loop_closure"]["iou_threshold"]
        self.num_matches = config["loop_closure"]["num_matches"]
        self.voxel_size = config["loop_closure"]["voxel_size"]
        self.pcd_voxel_size = float(config["reconstruction"].get("pcd_voxel_size", self.voxel_size))
        self.depth_by_rgb = None
        self.pose_by_rgb = None
        self.transition_frames = set()
        self.dist = config["loop_closure"]["dist"]
        self.batch_size = config["reconstruction"]["batch_size"]
        self.intrinsics = None
        self.depth_scale = None
        self.conf_threshold = 0.0

    def run(self, new_node, graph, accumulated_feats, rgb_paths):

        match = self.check_for_room_match(new_node, graph, self.room_threshold)

        if match is not None:
            matched_node = graph.room_nodes[match]
            merged, candidate_node = self._merge_nodes(new_node, matched_node, graph, accumulated_feats, rgb_paths)
            if merged:
                return True, candidate_node

        self.add_to_database(new_node)
        return False, new_node

    def run_pair(self, new_node, matched_node, graph, accumulated_feats, rgb_paths):

        merged, candidate_node = self._merge_nodes(new_node, matched_node, graph, accumulated_feats, rgb_paths)
        if merged:
            return True, candidate_node

        return False, new_node

    def run_pcd_overlap_check(self, graph, accumulated_feats, rgb_paths, optimiser):
        node_ids = list(graph.room_nodes.keys())
        merged_node_ids = []
        for i, idx_a in enumerate(node_ids):
            for j in range(i + 1, len(node_ids)):
                idx_b = node_ids[j]
                node_a = graph.room_nodes.get(idx_a)
                node_b = graph.room_nodes.get(idx_b)
                if node_a is None or node_b is None:
                    continue

                if node_a.label != node_b.label:
                    continue

                ratio = self._overlap_of_downsampled(
                    self._world_downsampled(node_a), self._world_downsampled(node_b), self.dist
                )
                if ratio <= self.iou_threshold:
                    continue

                success, candidate = self.run_pair(node_a, node_b, graph, accumulated_feats, rgb_paths)
                if success:
                    optimiser.optimise_sim3(graph)
                    merged_node_ids.append(candidate.idx)
                    print(f"IOU loop closure accepted: nodes {idx_a} <-> {idx_b}, iou={ratio:.4f}")
                    logger.info("Graph re-optimized after loop closure between %s and %s", idx_a, idx_b)
        return merged_node_ids
    
    def add_to_database(self, node):
        flatten_feats = np.array([f for batch in node.feats for f in batch])
        self.database[node.idx] = flatten_feats

    def _merge_nodes(self, new_node, matched_node, graph, accumulated_feats, rgb_paths):
        candidate_node = self._build_candidate_node(new_node, matched_node, graph, accumulated_feats, rgb_paths)
        affected_keys = [
            key for key in graph.room_to_room_edge.keys()
            if new_node.idx in key or matched_node.idx in key
        ]
        new_edges = self._build_candidate_edges(candidate_node, new_node, matched_node, graph, affected_keys)
        if new_edges is None:
            return False, new_node
        self._commit_merge(candidate_node, new_node, matched_node, graph, affected_keys, new_edges)
        return True, candidate_node

    def _build_candidate_node(self, new_node, matched_node, graph, accumulated_feats, rgb_paths):
        new_subsampling = self._merged_subsample(
            matched_node.img_paths,
            new_node.img_paths,
            n=self.batch_size,
            graph=graph,
            node_a_idx=matched_node.idx,
            node_b_idx=new_node.idx
        )
        merged_idx = [rgb_paths.index(p) for p in new_subsampling]

        depth_batch, poses = lookup_priors(new_subsampling, self.depth_by_rgb, self.pose_by_rgb)
        predictions = self.model.run(batch=new_subsampling, depth_batch=depth_batch, intrinsics=self.intrinsics, poses=poses, conf_threshold=self.conf_threshold, depth_scale=self.depth_scale)

        ref_pose = predictions["extrinsic"][0]
        local_poses = [np.linalg.inv(ref_pose) @ p for p in predictions["extrinsic"]]

        return RoomNode(
            idx=matched_node.idx,
            label=matched_node.label,
            img_paths=new_subsampling,
            ref_pose=ref_pose,
            local_poses=local_poses,
            predictions=predictions,
            feats=[accumulated_feats[i] for i in merged_idx],
            transition_frames=self.transition_frames,
        )

    def _build_candidate_edges(self, candidate_node, new_node, matched_node, graph, affected_keys):
        logger.debug("Affected keys: %s", affected_keys)
        edges = []

        for src, tgt in affected_keys:
            neighbor_idx = src if src not in [new_node.idx, matched_node.idx] else tgt

            if neighbor_idx in [new_node.idx, matched_node.idx]:
                continue

            logger.debug("%s %s", candidate_node.idx, neighbor_idx)

            neighbor_node = graph.room_nodes[neighbor_idx]
            found_transforms, pair_data = self.check_for_image_match(candidate_node, neighbor_node, self.image_threshold)
            if not found_transforms:
                logger.warning("No valid transforms for edge (%s, %s) — rejecting match", candidate_node.idx, neighbor_idx)
                return None

            for transform, pair in zip(found_transforms, pair_data):
                edges.append(Edge(
                    source=candidate_node.idx,
                    target=neighbor_idx,
                    edge_type="loop",
                    relative_transform=transform,
                    boundary_pairs=pair
                ))

        return edges if edges else None

    def _commit_merge(self, candidate_node, new_node, matched_node, graph, affected_keys, new_edges):
        logger.info("Successful match found %s %s", candidate_node.idx, new_node.idx)

        del graph.room_nodes[matched_node.idx]
        del graph.room_nodes[new_node.idx]
        graph.add_node(candidate_node)
        candidate_node.build_pcd(self.pcd_voxel_size)
        logger.debug("Deleted %s and %s, new node is %s", matched_node.idx, new_node.idx, candidate_node.idx)

        for src, tgt in affected_keys:
            logger.debug("Deleted edge %s %s", src, tgt)
            graph.room_to_room_edge.pop((src, tgt), None)

        for edge in new_edges:
            logger.debug("New edge: %s %s", edge.source, edge.target)
            graph.add_edge(edge)

        self.database.pop(matched_node.idx, None)
        self.database.pop(new_node.idx, None)
        self.add_to_database(candidate_node)

    def _world_downsampled(self, node):
        """node.pcd in the world frame, voxel-downsampled for the overlap check.

        Memoised per (node, cloud object, pose): a node is compared against every
        other node each step, and transforming + downsampling its full cloud for
        each pair was the dominant cost. A merge or re-optimisation changes the
        cloud object or pose, so those nodes are simply recomputed.
        """
        pose = np.asarray(node.optimised_ref_pose, dtype=np.float64)
        key = (node.idx, id(node.pcd), pose.tobytes())
        cache = self.__dict__.setdefault("_overlap_cache", {})
        hit = cache.get(node.idx)
        if hit is not None and hit[0] == key:
            return hit[1]
        down = o3d.geometry.PointCloud(node.pcd).transform(pose).voxel_down_sample(self.voxel_size)
        cache[node.idx] = (key, down)
        return down

    @staticmethod
    def _overlap_of_downsampled(pcd1_down, pcd2_down, distance_thresh=0.1):
        """Fraction of pcd1_down within distance_thresh of pcd2_down (both already downsampled)."""
        if len(pcd1_down.points) == 0 or len(pcd2_down.points) == 0:
            return 0.0
        # If the bounding boxes, grown by the distance threshold, don't
        # intersect, no point of pcd1 can be within range of pcd2: overlap is 0.
        lo1, hi1 = pcd1_down.get_min_bound(), pcd1_down.get_max_bound()
        lo2, hi2 = pcd2_down.get_min_bound(), pcd2_down.get_max_bound()
        if np.any(hi1 + distance_thresh < lo2) or np.any(hi2 + distance_thresh < lo1):
            return 0.0
        dists = np.asarray(pcd1_down.compute_point_cloud_distance(pcd2_down))
        return float(np.count_nonzero(dists <= distance_thresh)) / len(dists)

    @staticmethod
    def _compute_pcd_overlap(pcd1, pcd2, voxel_size=0.04, distance_thresh=0.1):
        """Returns the fraction of points in pcd1 within distance_thresh of pcd2.

        Same quantity as the original per-point KD-tree loop, computed with
        Open3D's vectorised nearest-neighbour distance instead.
        """
        if len(pcd1.points) == 0 or len(pcd2.points) == 0:
            return 0.0

        return LoopClosure._overlap_of_downsampled(
            pcd1.voxel_down_sample(voxel_size), pcd2.voxel_down_sample(voxel_size), distance_thresh
        )
    
    def _merged_subsample(self, list_a, list_b, n=60, graph=None, node_a_idx=None, node_b_idx=None):
        """
        Merge two sets of image paths (list_a and list_b), ensuring all images
        previously used in boundary pairs from any edge connected to node_a or node_b
        are included in the merged subsample.
        """

        combined = list_a + list_b
        total_len = len(combined)
        if total_len <= n:
            return combined

        must_include = set()

        # Gather all boundary-pair images from edges touching node_a or node_b
        if graph is not None and (node_a_idx is not None and node_b_idx is not None):
            for (src, tgt), edges in graph.room_to_room_edge.items():
                if src in [node_a_idx, node_b_idx] or tgt in [node_a_idx, node_b_idx]:
                    for edge in edges:
                        bp = edge.boundary_pairs
                        # Collect images used for node_a
                        if src == node_a_idx and bp["img_a"] in list_a:
                            must_include.add(bp["img_a"])
                        elif tgt == node_a_idx and bp["img_b"] in list_a:
                            must_include.add(bp["img_b"])
                        # Collect images used for node_b
                        if src == node_b_idx and bp["img_a"] in list_b:
                            must_include.add(bp["img_a"])
                        elif tgt == node_b_idx and bp["img_b"] in list_b:
                            must_include.add(bp["img_b"])

        # Convert must-include images to indices in the combined list
        must_include_indices = [combined.index(p) for p in must_include if p in combined]
        must_include_indices = sorted(set(must_include_indices))

        # Fill remaining slots evenly to reach n
        remaining_slots = n - len(must_include_indices)
        if remaining_slots > 0:
            all_indices = np.arange(total_len)
            mask = np.ones(total_len, dtype=bool)
            mask[must_include_indices] = False
            available_indices = all_indices[mask]

            fill_idx = np.linspace(0, len(available_indices) - 1, remaining_slots, dtype=int)
            additional_indices = available_indices[fill_idx]
            final_indices = sorted(np.concatenate([must_include_indices, additional_indices]).astype(int))
        else:
            final_indices = must_include_indices[:n]

        return [combined[i] for i in final_indices]

    def check_for_room_match(self, new_node, graph, threshold=0.85):
        new_feats = np.array([f for batch in new_node.feats for f in batch])
        query_norm = new_feats / (np.linalg.norm(new_feats, axis=1, keepdims=True) + 1e-9)

        best_match_idx = None
        best_num_matches = 0

        for idx, db_feats in self.database.items():
            # Avoid matching the node against itself or the previous batch
            if idx in [new_node.idx, new_node.idx - 1]:
                continue

            db_node = graph.room_nodes[idx]

            # Only compare nodes with the same room label
            if new_node.label != db_node.label:
                continue

            db_norm = db_feats / (np.linalg.norm(db_feats, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(query_norm, db_norm.T)
            matched_indices = np.where(sims >= threshold)
            num_matches = len(matched_indices[0])

            if num_matches > best_num_matches:
                best_num_matches = num_matches
                best_match_idx = db_node.idx

        if best_match_idx is not None and best_num_matches > self.num_matches:
            logger.info(
                "Found best match between %s %s and %s %s (%s matches)",
                new_node.label, new_node.idx,
                graph.room_nodes[best_match_idx].label, best_match_idx,
                best_num_matches
            )
            return best_match_idx
        return None

    def check_for_image_match(self, node1, node2, threshold=0.95):
        node1_feats = np.array([f for batch in node1.feats for f in batch])
        node1_norm = node1_feats / (np.linalg.norm(node1_feats, axis=1, keepdims=True) + 1e-9)
        node1_paths = node1.img_paths

        node2_feats = np.array([f for batch in node2.feats for f in batch])
        node2_norm = node2_feats / (np.linalg.norm(node2_feats, axis=1, keepdims=True) + 1e-9)
        node2_paths = node2.img_paths

        sims = np.dot(node1_norm, node2_norm.T)
        rows, cols = np.where(sims >= threshold)

        matches = []
        pair_data = []
        for row, col in zip(rows[:1], cols[:1]):
            img_a = node1_paths[row]
            img_b = node2_paths[col]

            depth_batch, poses = lookup_priors([img_a, img_b], self.depth_by_rgb, self.pose_by_rgb)
            predictions = self.model.run(batch=[img_a, img_b], depth_batch=depth_batch, intrinsics=self.intrinsics, poses=poses, conf_threshold=self.conf_threshold, depth_scale=self.depth_scale)

            T_ab = predictions["extrinsic"][1]

            T_refa_a = node1.local_poses[row]
            T_refb_b = node2.local_poses[col]
            T_refa_refb = T_refa_a @ T_ab @ np.linalg.inv(T_refb_b)
            matches.append(np.array(T_refa_refb, dtype=np.float32))

            pair_data.append({
                "img_a": img_a,
                "img_b": img_b,
                "idx_a": row,
                "idx_b": col,
            })

        return matches, pair_data
