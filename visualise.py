#!/usr/bin/env python3
"""
Scene graph visualisation using Rerun – single 3D view, three stacked layers.

The pipeline saves floor-aligned clouds (+Z up, floor at z = 0); they are
mapped to Rerun's Y-down frame so the floor sits at Y = 0. For outputs saved
without floor alignment, --fit-floor estimates the floor by RANSAC instead.

Vertical layout (top → bottom in world space, Y-down coordinate system):
  Layer 1 (top)    – graph: room nodes at centroids + adjacency edges
  Layer 2 (middle) – segmented_pcd.pcd   (floor-aligned)
  Layer 3 (bottom) – full_pcd.pcd        (floor-aligned, shifted down by LAYER_GAP)

Edges:
  • Within graph layer  : room-to-room adjacency (k-NN on centroids)
  • Graph → segmented   : each node drops a vertical connector to its
                          room centroid in the segmented PCD layer
  • Segmented → full    : each room centroid drops a connector to the
                          corresponding position in the full PCD layer

Rooms and centroids are read from 'original/*.pcd'.

Usage:
    python visualise.py [--output-dir /workspace/lexi-sg/output]
                        [--output scene_graph.rrd] [--k 2]
                        [--layer-gap-scale 2.5]
                        [--subsample 500000] [--spawn]
"""

import argparse
import glob
import os
import subprocess

import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb


PALETTE = [
    [220,  80,  80],
    [ 80, 200,  80],
    [ 80, 120, 220],
    [220, 200,  60],
    [200,  80, 200],
    [ 60, 200, 200],
    [240, 140,  40],
    [140,  60, 200],
    [ 80, 120, 220],
    [200, 120,  80],
]


def rotation_vec_to_vec(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src /= np.linalg.norm(src)
    dst /= np.linalg.norm(dst)

    v = np.cross(src, dst)
    s = np.linalg.norm(v)
    c = float(np.dot(src, dst))

    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.array([1, 0, 0]) if abs(src[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([[ 0,    -v[2],  v[1]],
                   [ v[2],  0,    -v[0]],
                   [-v[1],  v[0],  0   ]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def compute_floor_transform(pcd_path, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    print("  Fitting floor plane via RANSAC …")
    pcd = o3d.io.read_point_cloud(pcd_path)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal)
    print(f"  Floor normal (raw): {normal.round(4)}  d={d:.4f}  inliers={len(inliers):,}")

    target = np.array([0.0, -1.0, 0.0])
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    mask = np.zeros(len(pts_all), dtype=bool)
    mask[np.asarray(inliers)] = True

    def floor_is_below(n):
        R_mat = rotation_vec_to_vec(n, target)
        pts_r = (R_mat @ pts_all.T).T
        return pts_r[mask, 1].mean() > pts_r[~mask, 1].mean()

    if not floor_is_below(normal):
        normal = -normal
        print(f"  Flipped normal: {normal.round(4)}")

    R_mat = rotation_vec_to_vec(normal, target)
    pts_rot = (R_mat @ pts_all.T).T
    floor_y = float(pts_rot[mask, 1].mean())
    t = np.array([0.0, -floor_y, 0.0], dtype=np.float32)

    print(f"  Rotation R =\n{R_mat.round(4)}")
    print(f"  Translation t = {t.round(4)}")
    return R_mat.astype(np.float32), t


def apply_transform(pts, R_mat, t):
    return (R_mat @ pts.T).T + t


def load_pcd(path, subsample=None):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8) if pcd.has_colors() else None
    if subsample and len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts    = pts[idx]
        colors = colors[idx] if colors is not None else None
    return pts, colors


def shift_y(pts, delta):
    out = pts.copy()
    out[:, 1] += delta
    return out


def load_poses(path):
    positions = np.loadtxt(path, usecols=(1, 2, 3), dtype=np.float32, ndmin=2)
    print(f"  Loaded {len(positions)} poses from {path}")
    return positions


def compute_adjacency(centroids, k=2):
    n = len(centroids)
    edges = set()
    for i in range(n):
        dists = sorted(
            [(np.linalg.norm(centroids[i] - centroids[j]), j)
             for j in range(n) if j != i]
        )
        for _, j in dists[:k]:
            edges.add(tuple(sorted((i, j))))
    return list(edges)


def build_rrd(output_dir, output="scene_graph.rrd", k=2, layer_gap_scale=2.5,
              subsample=900000, point_radius=0.015, fit_floor=False):
    pcd_dir      = output_dir
    original_dir = os.path.join(pcd_dir, "original")
    full_path    = os.path.join(pcd_dir, "full_pcd.pcd")
    seg_path     = os.path.join(pcd_dir, "segmented_pcd.pcd")
    pose_path    = os.path.join(pcd_dir, "pose.txt")
    subsample    = subsample if subsample > 0 else None

    # ------------------------------------------------------------------ #
    # 0. Floor-alignment transform
    # ------------------------------------------------------------------ #
    if fit_floor:
        print("Computing floor-alignment transform …")
        R_floor, t_floor = compute_floor_transform(full_path)
    else:
        # Z-up (saved outputs) to Y-down: x -> x, z -> -y, y -> z.
        R_floor = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        t_floor = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 1. Per-room PCDs → centroids & colours
    # ------------------------------------------------------------------ #
    room_files = sorted(glob.glob(os.path.join(original_dir, "*.pcd")))
    if not room_files:
        raise FileNotFoundError(f"No .pcd files found in: {original_dir}")

    print(f"Found {len(room_files)} room(s) in {original_dir}:")
    rooms = {}
    for idx, path in enumerate(room_files):
        name = os.path.splitext(os.path.basename(path))[0]
        pts, colors = load_pcd(path, subsample=subsample)
        if len(pts) == 0:
            print(f"  [skip] {name} – empty")
            continue
        palette_color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.uint8)
        if colors is None:
            colors = np.tile(palette_color, (len(pts), 1))
        pts      = apply_transform(pts, R_floor, t_floor)
        centroid = pts.mean(axis=0)
        rooms[name] = {"pts": pts, "colors": colors, "centroid": centroid, "color": palette_color}
        print(f"  {name:30s}: {len(pts):>8,} pts  centroid={centroid.round(2)}")

    if not rooms:
        raise RuntimeError("No rooms loaded.")

    room_names = list(rooms.keys())
    centroids  = np.array([rooms[n]["centroid"] for n in room_names], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 2. Layer Y offsets
    # ------------------------------------------------------------------ #
    all_room_pts = np.concatenate([r["pts"] for r in rooms.values()])
    y_extent = float(all_room_pts[:, 1].max()) - float(all_room_pts[:, 1].min())
    gap = y_extent * layer_gap_scale

    graph_dy = -gap
    traj_dy  = -gap * 0.5
    seg_dy   =  0.0
    full_dy  = +gap

    print(f"\nLayer gap: {gap:.2f} m  (scale={layer_gap_scale})")

    # ------------------------------------------------------------------ #
    # 3. Adjacency edges
    # ------------------------------------------------------------------ #
    edges = compute_adjacency(centroids, k=k)
    print(f"\nAdjacency edges ({len(edges)} total, k={k}):")
    for i, j in edges:
        d = np.linalg.norm(centroids[i] - centroids[j])
        print(f"  {room_names[i]:25s} ↔  {room_names[j]:25s}  (d={d:.2f} m)")

    # ------------------------------------------------------------------ #
    # 4. Full & segmented PCDs
    # ------------------------------------------------------------------ #
    print(f"\nLoading full PCD …")
    full_pts, full_colors = load_pcd(full_path, subsample)
    full_pts = apply_transform(full_pts, R_floor, t_floor)
    print(f"  {len(full_pts):,} points")

    has_seg = os.path.exists(seg_path)
    if has_seg:
        print(f"Loading segmented PCD …")
        seg_pts, seg_colors = load_pcd(seg_path, subsample)
        seg_pts = apply_transform(seg_pts, R_floor, t_floor)
        print(f"  {len(seg_pts):,} points")
        if seg_colors is None:
            seg_colors  = np.full((len(seg_pts),  3), 200, dtype=np.uint8)
    else:
        print(f"No {seg_path}; segmented layer left out.")

    if full_colors is None:
        full_colors = np.full((len(full_pts), 3), 200, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # 4b. Trajectory
    # ------------------------------------------------------------------ #
    print(f"\nLoading trajectory …")
    traj_pts = load_poses(pose_path)
    traj_pts = apply_transform(traj_pts, R_floor, t_floor)

    # ------------------------------------------------------------------ #
    # 5. Rerun
    # ------------------------------------------------------------------ #
    output_path = os.path.join(pcd_dir, output)
    blueprint = rrb.Blueprint(rrb.Spatial3DView(name="Scene Graph"), collapse_panels=True)
    rr.init("scene_graph", default_blueprint=blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.save(output_path)

    if has_seg:
        rr.log("segmented/cloud",
               rr.Points3D(shift_y(seg_pts, seg_dy), colors=seg_colors, radii=point_radius),
               static=True)

    rr.log("full/cloud",
           rr.Points3D(shift_y(full_pts, full_dy), colors=full_colors, radii=point_radius),
           static=True)

    traj_shifted = shift_y(traj_pts, traj_dy)
    rr.log("trajectory/path",
           rr.LineStrips3D([traj_shifted], colors=[[80, 120, 220]], radii=0.04),
           static=True)
    rr.log("trajectory/poses",
           rr.Points3D(traj_shifted, colors=[[80, 120, 220]], radii=0.08),
           static=True)

    for traj_pt in [traj_shifted[0], traj_shifted[-1]]:
        full_pt = traj_pt.copy()
        full_pt[1] += full_dy - traj_dy
        rr.log("graph/edges/traj_to_full",
               rr.LineStrips3D([np.array([traj_pt, full_pt], dtype=np.float32)],
                               colors=[[80, 120, 220, 120]], radii=0.03),
               static=True)

    graph_centroids = centroids.copy()
    graph_centroids[:, 1] += graph_dy
    node_colors = np.array([rooms[n]["color"] for n in room_names], dtype=np.uint8)

    rr.log("graph/nodes",
           rr.Points3D(graph_centroids, colors=node_colors, radii=0.35, labels=room_names),
           static=True)

    if edges:
        rr.log("graph/edges/adjacency",
               rr.LineStrips3D(
                   [np.array([graph_centroids[i], graph_centroids[j]], dtype=np.float32) for i, j in edges],
                   colors=[[220, 220, 220]] * len(edges),
                   radii=0.05,
               ),
               static=True)

    rr.log("graph/edges/node_to_segmented",
           rr.LineStrips3D(
               [np.array([graph_centroids[i], centroids[i]], dtype=np.float32) for i in range(len(room_names))],
               colors=[list(rooms[n]["color"]) for n in room_names],
               radii=0.04,
           ),
           static=True)

    seg_to_full = []
    for i in range(len(room_names)):
        bot = centroids[i].copy()
        bot[1] += full_dy
        seg_to_full.append(np.array([centroids[i], bot], dtype=np.float32))
    rr.log("graph/edges/segmented_to_full",
           rr.LineStrips3D(seg_to_full, colors=[list(rooms[n]["color"]) for n in room_names], radii=0.03),
           static=True)

    print(f"\nSaved to {output_path}")
    print(f"Open with:  rerun {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",      default="/workspace/lexi-sg/output",
                        help="Path to the pipeline output folder")
    parser.add_argument("--output",          default="scene_graph.rrd")
    parser.add_argument("--k",               type=int,   default=2,
                        help="k-NN adjacency edges per node")
    parser.add_argument("--layer-gap-scale", type=float, default=2.5)
    parser.add_argument("--subsample",       type=int,   default=900000,
                        help="Max points per cloud (0 = no limit)")
    parser.add_argument("--point-radius",    type=float, default=0.015)
    parser.add_argument("--spawn",           action="store_true")
    parser.add_argument("--fit-floor",       action="store_true",
                        help="Estimate the floor by RANSAC (outputs saved without floor alignment)")
    args = parser.parse_args()

    output_path = build_rrd(args.output_dir, args.output, args.k, args.layer_gap_scale,
                            args.subsample, args.point_radius, args.fit_floor)
    if args.spawn:
        subprocess.Popen(["rerun", output_path])


if __name__ == "__main__":
    main()
