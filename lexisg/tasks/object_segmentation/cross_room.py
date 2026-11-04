"""Cross-room object association.

Object segmentation runs per room, so an object seen from two rooms -- most often
one physical room split into consecutive nodes -- is reported twice. This pass
links objects of rooms within `max_hops` of each other in the room graph, using
the same rule as the within-room CLIP-aware merge (voxel IoU, then CLIP cosine
similarity), applied in the world frame.

Tractability: only neighbouring rooms are compared, over which drift is small,
and only objects whose bounding box reaches into the other room's bounding box.
No model is run; the pass reuses each room's object clouds and CLIP features.
A room revisited much later in the sequence is not linked.
"""

import logging
from collections import defaultdict, deque

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def room_to_world(node):
    """Object clouds are in the room's reconstruction frame; the exported
    segmented cloud maps them to the world frame with this same transform."""
    if node.optimised_ref_pose is None or node.ref_pose is None:
        return np.eye(4)
    return np.asarray(node.optimised_ref_pose, dtype=np.float64) @ np.linalg.inv(
        np.asarray(node.ref_pose, dtype=np.float64))


def _neighbour_pairs(graph, max_hops):
    pairs = set()
    for start in graph.room_nodes:
        seen, queue = {start}, deque([(start, 0)])
        while queue:
            cur, hops = queue.popleft()
            if hops == max_hops:
                continue
            for nxt in graph.adj.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, hops + 1))
                if nxt in graph.room_nodes:
                    pairs.add((min(start, nxt), max(start, nxt)))
    return sorted(pairs)


def _l2(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def associate(graph, cfg):
    """Link objects across neighbouring rooms.

    Returns a dict with:
      objects: {(room, obj): {"global_id", "points"}}
      rooms:   {global_id: room it is assigned to (most points)}
      matches: list of accepted cross-room pairs with their IoU and similarity
    and the world-frame object clouds under "pcds" for export.
    """
    max_hops = int(cfg.get("max_hops", 2))
    voxel_size = float(cfg.get("voxel_size", 0.05))
    overlap_thr = float(cfg.get("overlap_threshold", 0.2))
    clip_thr = float(cfg.get("clip_similarity_threshold", 0.7))
    min_points = int(cfg.get("min_points", 50))
    margin = float(cfg.get("room_margin", 0.3))

    pcds, info, room_bounds = {}, {}, {}
    for ridx, node in graph.room_nodes.items():
        seg = node.object_segmentation
        if seg is None:
            continue
        T = room_to_world(node)
        for oid, pcd in seg.obj_pcds_world.items():
            if pcd is None or len(pcd.points) == 0:
                continue
            key = (int(ridx), int(oid))
            world = o3d.geometry.PointCloud(pcd).transform(T)
            pcds[key] = world
            pts = np.asarray(world.points)
            feat = seg.obj_clip_feats.get(oid)
            info[key] = {
                "n": len(pts),
                "lo": pts.min(axis=0), "hi": pts.max(axis=0),
                "vox": set(map(tuple, np.floor(pts / voxel_size).astype(np.int32))),
                "feat": None if feat is None else _l2(feat),
            }
        if node.pcd is not None and len(node.pcd.points) > 0:
            room = o3d.geometry.PointCloud(node.pcd).transform(
                np.asarray(node.optimised_ref_pose, dtype=np.float64))
            room_bounds[int(ridx)] = (room.get_min_bound() - margin, room.get_max_bound() + margin)

    by_room = defaultdict(list)
    for key in info:
        by_room[key[0]].append(key)

    def reaches(key, room):
        if room not in room_bounds:
            return False
        lo, hi = room_bounds[room]
        return not (np.any(info[key]["hi"] < lo) or np.any(info[key]["lo"] > hi))

    def usable(key):
        return info[key]["n"] >= min_points and info[key]["feat"] is not None

    parent = {k: k for k in info}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    matches, compared = [], 0
    pairs = _neighbour_pairs(graph, max_hops)
    for ra, rb in pairs:
        cand_a = [k for k in by_room.get(ra, []) if usable(k) and reaches(k, rb)]
        cand_b = [k for k in by_room.get(rb, []) if usable(k) and reaches(k, ra)]
        for a in cand_a:
            ia = info[a]
            for b in cand_b:
                ib = info[b]
                if np.any(ia["hi"] < ib["lo"]) or np.any(ib["hi"] < ia["lo"]):
                    continue
                compared += 1
                inter = len(ia["vox"] & ib["vox"])
                if inter == 0:
                    continue
                iou = inter / len(ia["vox"] | ib["vox"])
                if iou < overlap_thr:
                    continue
                sim = float(ia["feat"] @ ib["feat"])
                if sim < clip_thr:
                    continue
                parent[find(a)] = find(b)
                matches.append({"a": list(a), "b": list(b), "iou": round(iou, 4), "clip_sim": round(sim, 4)})

    groups = defaultdict(list)
    for key in sorted(info):
        groups[find(key)].append(key)

    objects, rooms = {}, {}
    for gid, members in enumerate(sorted(groups.values())):
        points = defaultdict(int)
        for key in members:
            objects[key] = {"global_id": gid, "points": info[key]["n"]}
            points[key[0]] += info[key]["n"]
        rooms[gid] = max(points, key=points.get)

    logger.info("Cross-room association: %d room pairs, %d object pairs compared, %d matches, "
                "%d objects -> %d global objects", len(pairs), compared, len(matches), len(info), len(groups))
    return {"objects": objects, "rooms": rooms, "matches": matches, "pcds": pcds,
            "room_pairs": len(pairs), "compared": compared}
