from __future__ import annotations

import os
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from supervision import Detections

from lexisg.utils.config import get_object_segmentation_stage_config
from lexisg.models.track_handler import SAMTracker
from lexisg.tasks.object_segmentation import artifact_io
from lexisg.utils.sv_compat import mask_to_xyxy


class BatchSAMTracker(SAMTracker):
    def __init__(self, config, clear_cuda_cache_on_add: bool = True):
        super().__init__(config)
        self.clear_cuda_cache_on_add = bool(clear_cuda_cache_on_add)

    def _maybe_clear_cuda_cache(self):
        if not self.clear_cuda_cache_on_add:
            return
        if self.device != "cuda" or not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _build_detections(mask: np.ndarray, obj_id: int, confidence: float) -> Detections:
        mask_arr = np.asarray(mask, dtype=bool)[None, ...]
        return Detections(
            xyxy=mask_to_xyxy(mask_arr),
            mask=mask_arr,
            class_id=np.array([int(obj_id)], dtype=np.int32),
            confidence=np.array([float(confidence)], dtype=np.float32),
            tracker_id=np.array([int(obj_id)], dtype=np.int32),
        )

    def _safe_get_ious(self, obj_id: int, frame_idx: int):
        try:
            return float(self.get_ious(int(obj_id), int(frame_idx)))
        except KeyError:
            return None

    def propagate_masks_together(
        self,
        source_fid: int,
        seeds: List[Dict],
        max_adjacent_frames: int,
        conf_thres: float,
        min_detections: int,
    ) -> Tuple[Dict[int, Dict[int, Detections]], Dict[int, Dict]]:
        if self.tracking_state is None:
            raise ValueError("Tracking state is not initialized.")

        self.video_predictor.reset_state(self.tracking_state)
        accepted: Dict[int, Dict[int, Detections]] = defaultdict(dict)
        metadata: Dict[int, Dict] = {}
        for seed in seeds:
            tid = int(seed["tracklet_id"])
            metadata[tid] = {
                "tracklet_id": tid,
                "source_fid_local": int(source_fid),
                "seed_det_idx": int(seed["det_idx"]),
                "seed_confidence": float(seed["seed_confidence"]),
                "accepted_frames_local": [],
                "rejected_frames_local": [],
                "survived_min_detections": False,
            }

        with torch.inference_mode(), self._autocast_context():
            for seed in seeds:
                self.video_predictor.add_new_mask(
                    self.tracking_state,
                    int(source_fid),
                    int(seed["tracklet_id"]),
                    torch.as_tensor(np.asarray(seed["mask"], dtype=bool), dtype=torch.bool),
                )
                self._maybe_clear_cuda_cache()

            for seed in seeds:
                tid = int(seed["tracklet_id"])
                conf = float(seed["seed_confidence"])
                accepted[tid][int(source_fid)] = self._build_detections(seed["mask"], tid, conf)
                metadata[tid]["accepted_frames_local"].append({"fid_local": int(source_fid), "confidence": conf})

            for reverse in (False, True):
                for fid, obj_ids, m_t in self.video_predictor.propagate_in_video(
                    self.tracking_state,
                    start_frame_idx=int(source_fid),
                    reverse=reverse,
                ):
                    fid = int(fid)
                    if fid == int(source_fid):
                        continue
                    if abs(fid - int(source_fid)) > int(max_adjacent_frames):
                        break

                    masks = (m_t > 0.0).squeeze(1).cpu().numpy()
                    for idx, obj_id in enumerate(obj_ids):
                        tid = int(obj_id)
                        mask = np.asarray(masks[idx], dtype=bool)
                        if mask.sum() == 0:
                            metadata[tid]["rejected_frames_local"].append(
                                {"fid_local": fid, "reason": "empty_mask", "confidence": None}
                            )
                            continue
                        conf = self._safe_get_ious(tid, fid)
                        if conf is None:
                            metadata[tid]["rejected_frames_local"].append(
                                {"fid_local": fid, "reason": "missing_tracking_iou", "confidence": None}
                            )
                            continue
                        if conf < float(conf_thres):
                            metadata[tid]["rejected_frames_local"].append(
                                {"fid_local": fid, "reason": "tracking_confidence", "confidence": conf}
                            )
                            continue
                        accepted[tid][fid] = self._build_detections(mask, tid, conf)
                        metadata[tid]["accepted_frames_local"].append({"fid_local": fid, "confidence": conf})

        accepted = {int(tid): dict(sorted(per_fid.items())) for tid, per_fid in accepted.items()}
        for tid, per_fid in accepted.items():
            survived = len(per_fid) >= int(min_detections)
            metadata[tid]["survived_min_detections"] = bool(survived)
            if not survived:
                metadata[tid]["rejected_reason"] = "short_track"
        accepted = {tid: per_fid for tid, per_fid in accepted.items() if len(per_fid) >= int(min_detections)}
        return accepted, metadata


class BidirectionalTrackHandler:
    def __init__(self, config):
        self.config = config
        tracking_cfg = get_object_segmentation_stage_config(self.config, "track")
        backend_cfg = tracking_cfg.get("bidirectional", {})
        self.max_adjacent_frames = int(backend_cfg.get("adjacent_frames", 3))
        self.iou_threshold = float(backend_cfg.get("iou_threshold", 0.8))
        self.ios_threshold = float(backend_cfg.get("ios_threshold", -1.0))
        self.covering_mask_threshold = float(backend_cfg.get("covering_mask_threshold", -1.0))
        self.covering_mask_ios_threshold = float(backend_cfg.get("covering_mask_ios_threshold", -1.0))
        self.ios_prune_removes_smaller = bool(backend_cfg.get("ios_prune_removes_smaller", False))
        self.overlap_resolution = str(backend_cfg.get("overlap_resolution", "merge")).strip().lower()
        self.min_overlap_frames = int(backend_cfg.get("min_overlap_frames", 2))
        self.clear_cuda_cache_on_add = bool(backend_cfg.get("clear_cuda_cache_on_add", True))
        self.tracker_model = None
        if self.overlap_resolution not in {"prune", "merge"}:
            raise ValueError(
                f"Unsupported object_segmentation.stages.track.bidirectional.overlap_resolution={self.overlap_resolution!r}. "
                "Expected 'prune' or 'merge'."
            )

    @staticmethod
    def _work_dir(tmp_img_dir: str) -> str:
        return os.path.join(os.path.dirname(str(tmp_img_dir)), "bidirectional_runtime")

    @staticmethod
    def _tracklet_root(work_dir: str) -> str:
        return os.path.join(work_dir, "tracklets")

    @staticmethod
    def _merge_frame_shard_root(work_dir: str) -> str:
        return os.path.join(work_dir, "cache", "merge_frame_shards")

    @classmethod
    def _merge_frame_shard_ready_path(cls, work_dir: str) -> str:
        return os.path.join(cls._merge_frame_shard_root(work_dir), "ready.pkl.gz")

    @classmethod
    def _merge_frame_shard_frame_dir(cls, work_dir: str, fid_global: int) -> str:
        return os.path.join(cls._merge_frame_shard_root(work_dir), f"frame_{int(fid_global):06d}")

    def _new_tracker_model(self):
        return BatchSAMTracker(self.config, clear_cuda_cache_on_add=self.clear_cuda_cache_on_add)

    @staticmethod
    def _should_log_progress(idx: int, total: int, every: int = 25) -> bool:
        return total > 0 and (idx == 1 or idx == total or idx % max(1, int(every)) == 0)

    @staticmethod
    def _pairwise_ios(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        mask_a = np.asarray(mask_a, dtype=bool)
        mask_b = np.asarray(mask_b, dtype=bool)
        intersection = int(np.logical_and(mask_a, mask_b).sum())
        if intersection <= 0:
            return 0.0
        smaller = min(int(mask_a.sum()), int(mask_b.sum()))
        return float(intersection / smaller) if smaller > 0 else 0.0

    @staticmethod
    def _pairwise_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        mask_a = np.asarray(mask_a, dtype=bool)
        mask_b = np.asarray(mask_b, dtype=bool)
        intersection = int(np.logical_and(mask_a, mask_b).sum())
        if intersection <= 0:
            return 0.0
        union = int(np.logical_or(mask_a, mask_b).sum())
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def _mask_area(mask: np.ndarray) -> int:
        return int(np.asarray(mask, dtype=bool).sum())

    @staticmethod
    def _mask_bbox(mask: np.ndarray):
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        if ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _bbox_intersects(a, b) -> bool:
        if a is None or b is None:
            return False
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    @staticmethod
    def _filter_frame_detections(detections: Detections | None, removed_ids: set[int]) -> Detections | None:
        if detections is None or len(detections) == 0:
            return None
        keep = ~np.isin(
            np.asarray(detections.class_id, dtype=np.int32),
            np.asarray(sorted(removed_ids), dtype=np.int32),
        )
        return detections[keep] if np.any(keep) else None

    @staticmethod
    def _find_parent(parent: Dict[int, int], x: int) -> int:
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @classmethod
    def _union_prefer_confidence(cls, parent: Dict[int, int], a: int, b: int, anchor_confidence: Dict[int, float]) -> int:
        ra = cls._find_parent(parent, int(a))
        rb = cls._find_parent(parent, int(b))
        if ra == rb:
            return int(ra)
        conf_a = float(anchor_confidence.get(int(ra), 0.0))
        conf_b = float(anchor_confidence.get(int(rb), 0.0))
        if conf_a > conf_b:
            root, other = int(ra), int(rb)
        elif conf_b > conf_a:
            root, other = int(rb), int(ra)
        else:
            root, other = (int(ra), int(rb)) if int(ra) < int(rb) else (int(rb), int(ra))
        parent[other] = root
        return root

    def _build_source_seed_map(self, candidates, load_candidate_mask):
        seeds_per_fid = defaultdict(list)
        next_tracklet_id = 0
        for candidate in candidates:
            fid_local = int(candidate["fid"])
            det_idx = int(candidate["det_idx"])
            mask = load_candidate_mask(fid_local, det_idx)
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            seeds_per_fid[fid_local].append(
                {
                    "tracklet_id": int(next_tracklet_id),
                    "det_idx": det_idx,
                    "source_fid_local": fid_local,
                    "mask": mask,
                    "seed_confidence": float(candidate.get("confidence", 0.0)),
                }
            )
            next_tracklet_id += 1
        return dict(seeds_per_fid)

    def _save_tracklet_records(self, work_dir: str, sampled_frame_ids: np.ndarray, accepted, metadata):
        tracklet_root = self._tracklet_root(work_dir)
        os.makedirs(tracklet_root, exist_ok=True)
        for tracklet_id, per_fid in accepted.items():
            payload = {
                "tracklet_id": int(tracklet_id),
                "source_fid_global": int(sampled_frame_ids[int(metadata[tracklet_id]["source_fid_local"])]),
                "anchor_confidence": float(metadata[tracklet_id]["seed_confidence"]),
                "frames": {},
            }
            for fid_local, dets in sorted(per_fid.items()):
                fid_global = int(sampled_frame_ids[int(fid_local)])
                payload["frames"][fid_global] = {
                    "fid_local": int(fid_local),
                    "confidence": float(dets.confidence[0]) if dets.confidence is not None else 0.0,
                    "mask": np.asarray(dets.mask[0], dtype=bool),
                }
            artifact_io.save_pickle_gz(
                os.path.join(tracklet_root, f"tracklet_{int(tracklet_id):08d}.pkl.gz"),
                payload,
            )

    def _collect_tracklets_to_disk(self, work_dir: str, tmp_img_dir: str, tracking_inputs, candidates, load_candidate_mask):
        tracking_cfg = get_object_segmentation_stage_config(self.config, "track")
        sampled_frame_ids = np.asarray(tracking_inputs["frame_ids_global"], dtype=np.int32)
        tracker_model = self._new_tracker_model()
        tracker_model.init_tracking(tmp_img_dir)
        self.tracker_model = tracker_model
        seeds_per_fid = self._build_source_seed_map(candidates, load_candidate_mask)
        active_sources = [(int(fid_local), seeds) for fid_local, seeds in sorted(seeds_per_fid.items()) if seeds]

        logger.info(
            "Collecting bidirectional tracklets to disk "
            f"sources={len(active_sources)} work_dir={work_dir}"
        )
        for source_idx, (source_fid_local, seeds) in enumerate(active_sources, start=1):
            if self._should_log_progress(source_idx, len(active_sources), every=10):
                logger.info(
                    "Tracklet collection progress "
                    f"source={source_idx}/{len(active_sources)} "
                    f"source_fid_local={int(source_fid_local)} seeds={len(seeds)}"
                )
            accepted, metadata = tracker_model.propagate_masks_together(
                source_fid=int(source_fid_local),
                seeds=seeds,
                max_adjacent_frames=self.max_adjacent_frames,
                conf_thres=float(tracking_cfg["conf_threshold"]),
                min_detections=int(tracking_cfg["min_detections"]),
            )
            self._save_tracklet_records(work_dir, sampled_frame_ids, accepted, metadata)

    def _build_frame_detection_shards_from_tracklet_files(self, work_dir: str) -> Tuple[List[int], Dict[int, float]]:
        root = self._merge_frame_shard_root(work_dir)
        ready_path = self._merge_frame_shard_ready_path(work_dir)
        if os.path.exists(ready_path):
            payload = artifact_io.load_pickle_gz(ready_path)
            return [int(fid) for fid in payload.get("frame_ids", [])], {
                int(k): float(v) for k, v in payload.get("anchor_confidence", {}).items()
            }

        os.makedirs(root, exist_ok=True)
        anchor_conf = {}
        frame_ids = set()
        tracklet_root = self._tracklet_root(work_dir)
        tracklet_files = sorted(
            name for name in os.listdir(tracklet_root) if name.startswith("tracklet_") and name.endswith(".pkl.gz")
        )
        logger.info(f"Building frame detection shards from tracklets: files={len(tracklet_files)}")
        for idx, name in enumerate(tracklet_files, start=1):
            payload = artifact_io.load_pickle_gz(os.path.join(tracklet_root, name))
            tid = int(payload["tracklet_id"])
            anchor_conf[tid] = float(payload.get("anchor_confidence", 0.0))
            for fid_global, frame_payload in sorted(payload.get("frames", {}).items()):
                fid_global = int(fid_global)
                frame_ids.add(fid_global)
                frame_dir = self._merge_frame_shard_frame_dir(work_dir, fid_global)
                os.makedirs(frame_dir, exist_ok=True)
                artifact_io.save_pickle_gz(
                    os.path.join(frame_dir, f"tracklet_{tid:08d}.pkl.gz"),
                    {
                        "tracklet_id": tid,
                        "fid_global": fid_global,
                        "fid_local": int(frame_payload.get("fid_local", -1)),
                        "confidence": float(frame_payload.get("confidence", 0.0)),
                        "mask": np.asarray(frame_payload["mask"], dtype=bool),
                    },
                )
            if idx % 200 == 0 or idx == len(tracklet_files):
                logger.info(f"Frame shard build progress: {idx}/{len(tracklet_files)}")

        artifact_io.save_pickle_gz(
            ready_path,
            {
                "frame_ids": sorted(int(fid) for fid in frame_ids),
                "tracklet_count": int(len(anchor_conf)),
                "anchor_confidence": anchor_conf,
            },
        )
        return sorted(int(fid) for fid in frame_ids), anchor_conf

    def _load_frame_detection_from_shards(self, work_dir: str, fid_global: int) -> Detections | None:
        frame_dir = self._merge_frame_shard_frame_dir(work_dir, int(fid_global))
        if not os.path.isdir(frame_dir):
            return None
        det_list = []
        for name in sorted(os.listdir(frame_dir)):
            if not name.endswith(".pkl.gz"):
                continue
            payload = artifact_io.load_pickle_gz(os.path.join(frame_dir, name))
            mask = np.asarray(payload["mask"], dtype=bool)[None, ...]
            tid = int(payload["tracklet_id"])
            conf = float(payload.get("confidence", 0.0))
            det_list.append(
                Detections(
                    xyxy=mask_to_xyxy(mask),
                    mask=mask,
                    class_id=np.array([tid], dtype=np.int32),
                    confidence=np.array([conf], dtype=np.float32),
                    tracker_id=np.array([tid], dtype=np.int32),
                )
            )
        return Detections.merge(det_list) if det_list else None

    def _resolve_frame_detections_by_overlap(self, work_dir: str, frame_ids: List[int], anchor_confidence: Dict[int, float]):
        removed_ids: set[int] = set()
        resolution_log: List[Dict] = []
        parent = {int(tid): int(tid) for tid in anchor_confidence.keys()}
        iou_count: Dict[Tuple[int, int], int] = defaultdict(int)
        covering_count: Dict[int, int] = defaultdict(int)
        ios_count: Dict[Tuple[int, int], int] = defaultdict(int)
        iou_details: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        covering_details: Dict[int, List[Dict]] = defaultdict(list)
        ios_details: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        logger.info(
            f"Starting frame-wise overlap resolution: frames={len(frame_ids)} "
            f"overlap_resolution={self.overlap_resolution} "
            f"iou_threshold={self.iou_threshold:.3f} covering_mask_threshold={self.covering_mask_threshold:.3f} "
            f"covering_mask_ios_threshold={self.covering_mask_ios_threshold:.3f} "
            f"ios_threshold={self.ios_threshold:.3f} min_overlap_frames={self.min_overlap_frames}"
        )

        for frame_idx, fid_global in enumerate(sorted(frame_ids), start=1):
            dets = self._load_frame_detection_from_shards(work_dir, int(fid_global))
            dets = self._filter_frame_detections(dets, removed_ids) if dets is not None else None
            if dets is None or len(dets) < 2:
                continue
            logger.info(
                f"Resolving overlaps frame_progress={frame_idx}/{len(frame_ids)} fid={int(fid_global)} "
                f"active_masks={len(dets)} removed_tracklets={len(removed_ids)}"
            )
            masks = np.asarray(dets.mask, dtype=bool)
            class_ids = np.asarray(dets.class_id, dtype=np.int32)
            boxes = [self._mask_bbox(mask) for mask in masks]

            if self.iou_threshold >= 0:
                for i in range(len(dets)):
                    tid_a = int(class_ids[i])
                    if tid_a in removed_ids:
                        continue
                    for j in range(i + 1, len(dets)):
                        tid_b = int(class_ids[j])
                        if tid_b in removed_ids:
                            continue
                        if self._find_parent(parent, tid_a) == self._find_parent(parent, tid_b):
                            continue
                        if not self._bbox_intersects(boxes[i], boxes[j]):
                            continue
                        key = (tid_a, tid_b) if tid_a < tid_b else (tid_b, tid_a)
                        iou = self._pairwise_iou(masks[i], masks[j])
                        if iou <= self.iou_threshold:
                            continue
                        iou_count[key] += 1
                        iou_details[key].append({"fid_global": int(fid_global), "iou": float(iou)})
                        if iou_count[key] < self.min_overlap_frames:
                            continue
                        if self.overlap_resolution == "merge":
                            root_tid = self._union_prefer_confidence(parent, tid_a, tid_b, anchor_confidence)
                            resolution_log.append(
                                {
                                    "mode": "merge",
                                    "overlap_reason": "iou",
                                    "merged_root_tracklet_id": int(root_tid),
                                    "tracklet_id_a": int(tid_a),
                                    "tracklet_id_b": int(tid_b),
                                    "anchor_confidence_a": float(anchor_confidence.get(tid_a, 0.0)),
                                    "anchor_confidence_b": float(anchor_confidence.get(tid_b, 0.0)),
                                    "qualifying_frame_count": int(iou_count[key]),
                                    "qualifying_frames": list(iou_details[key]),
                                    "iou_threshold": float(self.iou_threshold),
                                    "ios_threshold": None,
                                    "min_overlap_frames": int(self.min_overlap_frames),
                                }
                            )
                            continue
                        conf_a = float(anchor_confidence.get(tid_a, 0.0))
                        conf_b = float(anchor_confidence.get(tid_b, 0.0))
                        if conf_a < conf_b:
                            remove_tid, keep_tid = tid_a, tid_b
                        elif conf_b < conf_a:
                            remove_tid, keep_tid = tid_b, tid_a
                        else:
                            remove_tid, keep_tid = max(tid_a, tid_b), min(tid_a, tid_b)
                        if remove_tid in removed_ids:
                            continue
                        removed_ids.add(int(remove_tid))
                        resolution_log.append(
                            {
                                "mode": "prune",
                                "overlap_reason": "iou",
                                "removed_tracklet_id": int(remove_tid),
                                "kept_tracklet_id": int(keep_tid),
                                "removed_anchor_confidence": float(anchor_confidence.get(remove_tid, 0.0)),
                                "kept_anchor_confidence": float(anchor_confidence.get(keep_tid, 0.0)),
                                "iou_frame_count": int(iou_count[key]),
                                "qualifying_frame_count": int(iou_count[key]),
                                "qualifying_frames": list(iou_details[key]),
                                "iou_threshold": float(self.iou_threshold),
                                "covering_mask_threshold": None,
                                "ios_threshold": None,
                                "min_overlap_frames": int(self.min_overlap_frames),
                                "prune_mode": "iou",
                            }
                        )

            remaining_indices = [idx for idx, tid in enumerate(class_ids.tolist()) if int(tid) not in removed_ids]
            if self.covering_mask_threshold >= 0 and len(remaining_indices) >= 3:
                for idx_big in remaining_indices:
                    tid_big = int(class_ids[idx_big])
                    if tid_big in removed_ids:
                        continue
                    big_mask = masks[idx_big]
                    big_area = self._mask_area(big_mask)
                    if big_area <= 0:
                        continue
                    union_small = None
                    small_tracklet_ids = []
                    for idx_small in remaining_indices:
                        if idx_small == idx_big:
                            continue
                        tid_small = int(class_ids[idx_small])
                        if tid_small in removed_ids:
                            continue
                        if self._mask_area(masks[idx_small]) >= big_area:
                            continue
                        if not self._bbox_intersects(boxes[idx_big], boxes[idx_small]):
                            continue
                        if self.covering_mask_ios_threshold >= 0:
                            ios_small_big = self._pairwise_ios(masks[idx_small], big_mask)
                            if ios_small_big <= self.covering_mask_ios_threshold:
                                continue
                        union_small = (
                            np.asarray(masks[idx_small], dtype=bool)
                            if union_small is None
                            else np.logical_or(union_small, np.asarray(masks[idx_small], dtype=bool))
                        )
                        small_tracklet_ids.append(tid_small)
                    if union_small is None or len(small_tracklet_ids) < 2:
                        continue
                    covering_iou = self._pairwise_iou(big_mask, union_small)
                    if covering_iou <= self.covering_mask_threshold:
                        continue
                    covering_count[tid_big] += 1
                    covering_details[tid_big].append(
                        {
                            "fid_global": int(fid_global),
                            "covering_iou": float(covering_iou),
                            "small_tracklet_ids": [int(tid) for tid in sorted(small_tracklet_ids)],
                        }
                    )
                    if covering_count[tid_big] < self.min_overlap_frames or tid_big in removed_ids:
                        continue
                    removed_ids.add(tid_big)
                    resolution_log.append(
                        {
                            "mode": "prune",
                            "overlap_reason": "covering_mask",
                            "removed_tracklet_id": int(tid_big),
                            "kept_tracklet_id": None,
                            "supporting_tracklet_ids": [int(tid) for tid in sorted(set(small_tracklet_ids))],
                            "removed_anchor_confidence": float(anchor_confidence.get(tid_big, 0.0)),
                            "kept_anchor_confidence": None,
                            "iou_frame_count": 0,
                            "qualifying_frame_count": int(covering_count[tid_big]),
                            "qualifying_frames": list(covering_details[tid_big]),
                            "iou_threshold": None,
                            "covering_mask_threshold": float(self.covering_mask_threshold),
                            "ios_threshold": None,
                            "min_overlap_frames": int(self.min_overlap_frames),
                            "prune_mode": "covering_mask",
                        }
                    )

            remaining_indices = [idx for idx, tid in enumerate(class_ids.tolist()) if int(tid) not in removed_ids]
            if self.ios_threshold < 0 or len(remaining_indices) < 2:
                continue
            remaining_masks = masks[remaining_indices]
            remaining_class_ids = class_ids[remaining_indices]
            remaining_boxes = [boxes[idx] for idx in remaining_indices]
            for i in range(len(remaining_indices)):
                tid_a = int(remaining_class_ids[i])
                if tid_a in removed_ids:
                    continue
                for j in range(i + 1, len(remaining_indices)):
                    tid_b = int(remaining_class_ids[j])
                    if tid_b in removed_ids:
                        continue
                    if self._find_parent(parent, tid_a) == self._find_parent(parent, tid_b):
                        continue
                    if not self._bbox_intersects(remaining_boxes[i], remaining_boxes[j]):
                        continue
                    key = (tid_a, tid_b) if tid_a < tid_b else (tid_b, tid_a)
                    ios = self._pairwise_ios(remaining_masks[i], remaining_masks[j])
                    if ios <= self.ios_threshold:
                        continue
                    ios_count[key] += 1
                    ios_details[key].append({"fid_global": int(fid_global), "ios": float(ios)})
                    if ios_count[key] < self.min_overlap_frames:
                        continue
                    area_a = self._mask_area(remaining_masks[i])
                    area_b = self._mask_area(remaining_masks[j])
                    if area_a < area_b:
                        smaller_tid, larger_tid = tid_a, tid_b
                    elif area_b < area_a:
                        smaller_tid, larger_tid = tid_b, tid_a
                    else:
                        smaller_tid, larger_tid = (tid_a, tid_b) if tid_a < tid_b else (tid_b, tid_a)
                    remove_tid, keep_tid = (
                        (smaller_tid, larger_tid) if self.ios_prune_removes_smaller else (larger_tid, smaller_tid)
                    )
                    if remove_tid in removed_ids:
                        continue
                    removed_ids.add(int(remove_tid))
                    resolution_log.append(
                        {
                            "mode": "prune",
                            "overlap_reason": "ios",
                            "removed_tracklet_id": int(remove_tid),
                            "kept_tracklet_id": int(keep_tid),
                            "removed_anchor_confidence": float(anchor_confidence.get(remove_tid, 0.0)),
                            "kept_anchor_confidence": float(anchor_confidence.get(keep_tid, 0.0)),
                            "iou_frame_count": int(iou_count.get(key, 0)),
                            "qualifying_frame_count": int(ios_count[key]),
                            "qualifying_frames": list(ios_details[key]),
                            "iou_threshold": float(self.iou_threshold),
                            "covering_mask_threshold": None,
                            "ios_threshold": float(self.ios_threshold),
                            "min_overlap_frames": int(self.min_overlap_frames),
                            "prune_mode": "ios",
                        }
                    )

        logger.info(
            f"Frame-wise overlap resolution finished: initial={len(anchor_confidence)} "
            f"removed={len(removed_ids)} "
            f"merged_roots={len({self._find_parent(parent, tid) for tid in anchor_confidence.keys()})} "
            f"remaining={len(anchor_confidence) - len(removed_ids)}"
        )
        merge_map = {
            int(tid): int(self._find_parent(parent, tid))
            for tid in anchor_confidence.keys()
            if int(self._find_parent(parent, tid)) != int(tid)
        }
        return removed_ids, merge_map, resolution_log

    @staticmethod
    def _merge_frame_detections_by_root(filtered: Detections, merge_map: Dict[int, int]) -> Detections:
        per_root = {}
        masks_arr = np.asarray(filtered.mask, dtype=bool)
        class_ids = np.asarray(filtered.class_id, dtype=np.int32)
        confidences = (
            np.asarray(filtered.confidence, dtype=np.float32)
            if filtered.confidence is not None
            else np.ones((len(filtered),), dtype=np.float32)
        )
        for idx in range(len(filtered)):
            tid = int(class_ids[idx])
            root = int(merge_map.get(tid, tid))
            entry = per_root.setdefault(root, {"mask": np.zeros_like(masks_arr[idx], dtype=bool), "confidence": 0.0})
            entry["mask"] = np.logical_or(entry["mask"], masks_arr[idx])
            entry["confidence"] = max(float(entry["confidence"]), float(confidences[idx]))
        root_ids = sorted(int(root) for root in per_root.keys())
        mask_stack = np.stack([np.asarray(per_root[root]["mask"], dtype=bool) for root in root_ids], axis=0)
        return Detections(
            xyxy=mask_to_xyxy(mask_stack),
            mask=mask_stack,
            class_id=np.asarray(root_ids, dtype=np.int32),
            tracker_id=np.asarray(root_ids, dtype=np.int32),
            confidence=np.asarray([float(per_root[root]["confidence"]) for root in root_ids], dtype=np.float32),
        )

    def _materialize_final_masks(self, work_dir: str, frame_ids: List[int], removed_ids: set[int], merge_map: Dict[int, int]):
        final_masks = {}
        survived_tracklet_ids = set()
        sorted_frame_ids = sorted(int(fid) for fid in frame_ids)
        logger.info(f"Saving resolved detection summaries frames={len(sorted_frame_ids)}")
        for frame_idx, fid_global in enumerate(sorted_frame_ids, start=1):
            if self._should_log_progress(frame_idx, len(sorted_frame_ids), every=25):
                logger.info(
                    "Resolved detection summary progress "
                    f"frame={frame_idx}/{len(sorted_frame_ids)} fid={int(fid_global)} "
                    f"removed_tracklets={len(removed_ids)}"
                )
            dets = self._load_frame_detection_from_shards(work_dir, int(fid_global))
            filtered = self._filter_frame_detections(dets, removed_ids) if dets is not None else None
            if filtered is not None and len(filtered) > 0 and merge_map:
                filtered = self._merge_frame_detections_by_root(filtered, merge_map)
            if filtered is None or len(filtered) == 0:
                continue
            per_frame = {}
            for i in range(len(filtered)):
                tid = int(filtered.class_id[i])
                survived_tracklet_ids.add(tid)
                per_frame[tid] = np.asarray(filtered.mask[i], dtype=bool)
            if per_frame:
                final_masks[int(fid_global)] = per_frame
        logger.info(
            "Resolved detection summary finished "
            f"frames={len(sorted_frame_ids)} surviving_tracklets={len(survived_tracklet_ids)}"
        )
        return final_masks, sorted(int(tid) for tid in survived_tracklet_ids)

    def track_from_candidates(self, tracking_inputs, tmp_img_dir, candidates, load_candidate_mask):
        work_dir = self._work_dir(tmp_img_dir)
        os.makedirs(work_dir, exist_ok=True)
        try:
            logger.info(
                "Starting bidirectional tracked mask propagation "
                f"candidates={len(candidates)} adjacent_frames={self.max_adjacent_frames}"
            )
            self._collect_tracklets_to_disk(work_dir, tmp_img_dir, tracking_inputs, candidates, load_candidate_mask)
            logger.info("Bidirectional propagation finished; building per-frame shard cache")
            frame_ids, anchor_confidence = self._build_frame_detection_shards_from_tracklet_files(work_dir)
            logger.info(f"Per-frame shard cache ready frames={len(frame_ids)} tracklets={len(anchor_confidence)}")
            removed_ids, merge_map, _ = self._resolve_frame_detections_by_overlap(
                work_dir=work_dir,
                frame_ids=frame_ids,
                anchor_confidence=anchor_confidence,
            )
            final_masks, final_tracklet_ids = self._materialize_final_masks(work_dir, frame_ids, removed_ids, merge_map)
            sampled_frame_ids = np.asarray(tracking_inputs["frame_ids_global"], dtype=np.int32).tolist()
            masks_local = {
                int(fid_local): {
                    int(obj_id): np.asarray(mask, dtype=bool)
                    for obj_id, mask in final_masks.get(int(fid_global), {}).items()
                }
                for fid_local, fid_global in enumerate(sampled_frame_ids)
                if final_masks.get(int(fid_global), {})
            }
            logger.info(
                "Bidirectional tracking complete "
                f"tracklets_before={len(anchor_confidence)} removed={len(removed_ids)} "
                f"objects_after={len(final_tracklet_ids)} overlap_resolution={self.overlap_resolution}"
            )
            return masks_local, {}
        finally:
            if self.tracker_model is not None:
                self.tracker_model.cleanup_after_run()
                self.tracker_model = None
            shutil.rmtree(work_dir, ignore_errors=True)
