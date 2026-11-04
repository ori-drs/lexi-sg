# Object Segmentation Pipeline Reference for AI Agents

This file explains the logic of `lexisg/tasks/object_segmentation/pipeline.py` in a self-contained way so another agent can reason about the pipeline without opening the implementation or related libraries.

The goal is not to mirror every line. The goal is to explain:

- what the pipeline is trying to compute
- what each stage expects and produces
- how intermediate artifacts are used
- what rules control merging, ranking, and filtering
- what each config parameter means in practice

## 1. Pipeline Purpose

The pipeline builds object-level 3D scene outputs from an RGB-D video sequence with poses.

Conceptually, it does this:

1. Select a subset of frames uniformly from a scene.
2. Generate object mask candidates for those frames.
3. Track the chosen masks across the selected frames.
4. Save tracked per-frame detections to disk.
5. Merge duplicate tracklets in 2D, then build and merge 3D objects.
6. Compute CLIP features from the best object views.
7. Export object point clouds and embeddings for OpenLex3D-style semantic evaluation.

The pipeline assumes:

- RGB images exist.
- depth maps exist.
- camera poses exist.
- camera intrinsics exist.
- all of them are aligned frame-by-frame.

If depth or poses are missing, this pipeline is not valid.

## 2. High-Level Mental Model

Think of the pipeline as turning a scene into this final object table:

| Field | Meaning |
| --- | --- |
| `object_id` | Stable integer ID after merges. |
| `masks` | Binary masks for frames where the object appears. |
| `point_cloud` | 3D points backprojected from the object masks. |
| `best_frames` | Frames ranked as best object views. |
| `clip_feature` | Averaged L2-normalized visual embedding. |
| `clip_crops` | Cropped RGB images used to compute the embedding. |

Every stage either creates or refines parts of that table.

Point-cloud construction note:

- object geometry is assembled from per-frame mask backprojections
- an optional per-frame point-cloud filter can run before those frame clouds are merged into an object cloud
- the current filter supports voxel downsampling and optional DBSCAN cluster pruning

## 3. Global Execution Structure

The stage order is fixed:

1. `detect`
2. `track`
3. `merge`
4. `clip`
5. `eval`

The user cannot request a subset. `run_batch()` always executes the full pipeline in that order.

Important rule:

- each stage is executed at most once per `run_batch()` call, in stage order

This means stage outputs are treated as run-local artifacts, not as resumable stage checkpoints.

Service integration note:

- the reusable `ObjectSegmentationService` now defaults to running detect, track, merge, and CLIP
- callers must still explicitly request eval if they want that final stage
- the standalone `lexisg/tasks/object_segmentation/pipeline.py` scene wrapper still opts into eval explicitly

Implementation note:

- `lexisg/tasks/object_segmentation/pipeline.py` is now only the scene-level orchestrator
- each stage lives in its own file under `lexisg/tasks/object_segmentation/`
- each stage instance receives the same Hydra config plus its input/output artifact paths
- stages do not call each other directly; they communicate only through saved artifacts
- merge/postprocess behavior is delegated to modular helper components so the merge path is easier to swap and extend
- all stage artifacts now live directly under one run directory rather than stage-named subdirectories

## 4. Core Data Structures

This section is the most important part for another agent.

### 4.1 Dataset context

Once loaded, the pipeline operates on this conceptual context:

| Key | Meaning |
| --- | --- |
| `rgb_paths` | Ordered list of RGB file paths after `subsample`. |
| `depth_paths` | Ordered list of depth file paths after `subsample`. |
| `poses` | Ordered list of camera-to-world or world-aligned pose matrices after `subsample`. |
| `intrinsics` | Camera intrinsics matrix. |
| `depth_scale` | Divisor used to convert stored depth to metric depth. |

### 4.2 Uniform plan

The uniform plan decides which frames will be processed.

| Key | Meaning |
| --- | --- |
| `frame_ids` | All candidate frame indices after subsampling. |
| `frame_sampling_mode` | Sampling mode used to build `sampled_frame_ids`; either `count` or `stride`. |
| `num_uniform_frames` | Requested sample count when `frame_sampling_mode=count`. |
| `uniform_frame_stride` | Requested step size when `frame_sampling_mode=stride`. |
| `sampled_frame_ids` | Sampled frame IDs that will be processed in the subsampled scene sequence. |

### 4.3 Predictions object

Before tracking, the pipeline prepares a `predictions` bundle for the sampled frame set.

It contains:

| Key | Meaning |
| --- | --- |
| `images` | RGB images for sampled frames in this run. |
| `depth` | Depth maps for sampled frames in this run. |
| `extrinsic` | Per-frame poses. |
| `intrinsic` | Per-frame copy of intrinsics. |
| `T_world_local` | Rigid transform that maps local geometry into world coordinates. |
| `aligned_pose` | Final aligned pose for the sampled frame set. |
| `frame_ids_global` | Global frame IDs for the sampled frame set. |
| `candidate_frame_ids_global` | Global candidate frame IDs for the full plan. |
| `all_selected_frame_ids_global` | All globally sampled frame IDs in this run. |
| `rgb_paths_all` | Full RGB path list for the full scene. |
| `shared_geometry` | Local voxel-downsampled XYZ point cloud plus per-frame pixel-to-point lookup maps into that downsampled cloud. |

Important integration note for the room-level pipeline in `main.py`:

- when object segmentation is run from precomputed reconstruction predictions, the service should use RGB images on the same pixel grid as `predictions["images"]`
- this avoids mask/depth shape mismatches during 3D backprojection in the merge stage
- in the current room-level integration, those model-space RGB images may be materialized to files and used in place of the original raw `rgb_paths`

Interpretation of `T_world_local`:

- tracking/backprojection initially builds object geometry in a local coordinate system
- `T_world_local` converts that local geometry into a common world frame
- after this transform, tracked objects can be merged in the same coordinate system

Interpretation of `shared_geometry`:

- it is built once after the RGB-D forward pass and before tracking/geometry reconstruction when needed
- its points are in the same local coordinate system as `extrinsic`
- in the current implementation it stores XYZ only, not per-point RGB, because tracker-side reuse only needs geometry
- it starts from all valid RGB-D backprojected points in the sampled frame set, then voxel-downsamples that shared cloud
- its per-frame lookup maps translate image pixels directly into indices of the voxel-downsampled point array
- later tracker modules can reuse those indices instead of recomputing backprojection for each mask

### 4.4 Object result structure

Most later logic revolves around a result object with these fields:

| Key | Meaning |
| --- | --- |
| `sampled_frame_ids` | Sampled frame IDs used for this result. |
| `masks` | `frame_id -> object_id -> binary_mask`. |
| `obj_pcds_world` | `object_id -> world-space point cloud`. |
| `best_fid_per_obj` | `object_id -> best global frame ID`. |
| `best_fids_per_obj` | `object_id -> ranked global frame IDs`. |
| `obj_clip_feats` | `object_id -> CLIP feature vector`. |
| `obj_clip_crops` | `object_id -> list of (frame_id, rank, crop_image)`. |
| `merge_map` | mapping from old object IDs to merged object IDs. |

## 5. Stage Contracts

This section describes each stage as a contract: inputs, logic, outputs, and important invariants.

## 5.1 Detect Stage

### Purpose

The detect stage prepares reusable object mask candidates. It does not produce final tracked objects yet.

It exists mainly to separate expensive mask generation from tracking.

### Inputs

- dataset context
- uniform plan
- RGB images for the sampled frames
- detection config

### Internal logic

For the sampled frame set:

1. Load sampled RGB images only.
2. Run the configured detector independently on each sampled frame.
3. Apply a per-frame post-detection mask-overlap filter. If two masks have IoS above the configured threshold, remove the larger mask when it mostly covers the smaller one.
4. Save one detection artifact per sampled frame.
5. Build a candidate manifest by sorting those per-frame detections.

The detect stage is now tracker-free.

- no adjacent-window propagation
- no dense tracker-assisted aggregation
- no cross-frame detection merging inside detect

### Outputs

- per-frame detection artifacts
- one candidate manifest
- frozen frame selection description
- raw annotated detection overlays under `visualizations/detect_raw/`
- stage metadata

### Important invariant

The detect stage defines the sampled-frame plan consumed by later stages. If the sampling-related config changes, later stages must not reuse old detect artifacts.

Visualization control:

- `object_segmentation.stages.detect.save_visualizations` controls whether detect overlays are written
- default is `true`

## 5.2 Track Stage

### Purpose

The track stage converts mask candidates into tracked per-frame detections.

It no longer finalizes object geometry. Instead it writes a backend-independent
tracked-mask artifact that the merge stage can reload later.

### Inputs

- detect-stage artifacts
- full RGB-D-pose data for the sampled frames
- tracking config
- reconstruction config

### Internal logic

For the sampled frame set:

1. Load RGB, depth, poses, and intrinsics for sampled frames.
2. Build the full `predictions` object.
3. Build the full `predictions` object without shared geometry because the track stage is bidirectional-only.
4. Save temporary RGB images because the tracker expects an image directory.
5. Load candidate masks saved by the detect stage.
6. Select promising masks and track them across the sampled frames.
7. Optionally prune duplicate bidirectional tracklets using the tracker-side overlap rules.
8. Save one tracked-detection artifact per sampled frame, keyed by global frame ID.
9. Save a lightweight track manifest that records tracklet IDs, keypoint associations, and the tracked-detection directory.

### What “tracking” means here

Tracking is not just temporal correspondence. In this pipeline it effectively means:

- choose a set of initial candidate segments
- propagate or associate them across the selected frames
- optionally prune obviously duplicate bidirectional tracklets before stage output
- build an object-centric mask set
- emit those tracked masks in the common per-frame stage format

### Outputs

For the sampled frame set:

- tracked detections keyed by global frame ID
- tracklet IDs present in the tracked result
- optional keypoint associations
- a track manifest pointing to the saved tracked detections
- annotated tracked/pruned overlays under `visualizations/track_tracked/`

### Important invariant

- tracked masks are serialized per frame
- the saved class/tracker ID in each detection is the tracklet ID
- merge never needs backend-specific decoding logic

- merge stage still owns the scene-level consolidation that combines surviving tracklets into final objects

Visualization control:

- `object_segmentation.stages.track.save_visualizations` controls whether tracked/pruned overlays are written
- default is `true`

## 5.3 Merge Stage

### Purpose

The merge stage now owns object consolidation after tracking.

It reloads tracked detections from disk, performs incremental 2D tracklet merging,
then reconstructs object geometry from the merged masks, and finally runs 3D overlap
merging to produce one scene-level object set.

### Inputs

- the track result plus tracked detections
- full RGB-D-pose data needed to rebuild object geometry
- merging config

### Internal logic

1. Load the tracked result.
2. Iterate tracked detections in global frame order and count pairwise 2D overlaps incrementally.
3. Accept a 2D merge once a pair has satisfied the configured IoU and/or IoS rule for at least `min_overlap_frames`.
4. Re-materialize per-frame merged masks from the accepted 2D unions.
5. Rebuild object point clouds from those merged masks using RGB-D backprojection.
6. Run the configured merge-stage strategy list, which now typically means only the 3D overlap merge because the 2D pass has already run explicitly.

Default strategy behavior:

- the default `object_segmentation.stages.merge.merge_stage_strategies` value is effectively `["overlap"]`
- in the current staged path, merge stage handles 2D overlap explicitly first and then uses the overlap strategy for the 3D pass

#### Incremental 2D merge

Two tracklets are merged during the first merge-stage pass if, while iterating frames in order:

- both appear in the same frame
- their boxes intersect
- their masks satisfy either:
  - IoU = intersection / union
  - IoS = intersection / smaller-mask-area
- the chosen criterion is met on at least `min_overlap_frames`

The union is accepted as soon as the counter reaches that threshold.

#### Geometry rebuild

After 2D unions are fixed:

- masks that map to the same 2D root are OR-ed together per frame
- the merge stage reruns RGB-D preparation for the sampled frame set
- merged masks are backprojected into per-object point clouds
- optional per-frame point-cloud filtering is applied during that reconstruction
- resulting local clouds are transformed into world coordinates

#### 3D merge

After geometry exists, the overlap merge strategy can still merge objects by 3D voxel IoU:

- voxelize each object cloud
- compute voxel intersection and union
- compute voxel IoU
- merge if it exceeds the threshold

### Merge implementation style

Merging uses union-find style grouping:

- each object starts as its own parent
- pairwise conditions create unions
- final groups are formed from connected components
- all objects in one group collapse to the smallest root ID

### Outputs

- merged masks
- merged object point clouds
- merged keypoint sets
- a merge map from old IDs to new IDs
- annotated merged-mask overlays under `visualizations/merge_merged/`

### Important invariant

Merge stage outputs are scene-level, and the saved `merge_map`
maps original track-stage tracklet IDs to final merged object IDs.

Visualization control:

- `object_segmentation.stages.merge.save_visualizations` controls whether merged-mask overlays are written
- default is `true`

## 5.4 Clip Stage

### Purpose

The clip stage creates the final semantic representation of each merged object.

### Inputs

- merged object masks
- merged object point clouds
- original RGB images
- CLIP config
- merge config for optional CLIP-aware merge

### Internal logic

For each object:

1. Rank frames where the object appears.
2. Keep the top `N` frames.
3. For each selected frame:
  - take the object mask
  - find the bounding rectangle of the mask
  - expand the crop using either fixed-pixel padding or ratio padding
  - apply the configured background mode:
    - original image background
    - white background outside the mask
    - black background outside the mask
  - crop the masked image region
  - compute the CLIP image embedding
4. Average embeddings from the selected frames.
5. L2-normalize the result.

Then:

- remove objects that still have no CLIP embedding
- optionally run the configured clip-stage merge strategy list

Default strategy behavior:

- the default `object_segmentation.stages.merge.clip_stage_strategies` value is effectively `["clip_aware_world"]`
- that strategy implements the CLIP-aware 3D merge described below

### CLIP-aware 3D merge

This is stricter than ordinary 3D merge.

Two objects merge only if:

- their voxelized 3D clouds overlap enough
- their CLIP embeddings are sufficiently similar

This reduces the chance of incorrectly merging spatially close but semantically different objects.

### Final export logic

The clip stage writes:

- one `.ply` per object
- one merged scene `.ply`
- one `.npy` feature vector per object
- saved crop images used for CLIP
- OpenLex3D-format exports

For OpenLex3D exports:

- only objects that have both geometry and CLIP features are kept
- points with invalid colors are removed
- pure white points are removed
- near-zero color points are removed
- an embedding table is created
- each exported point stores the row index of its object embedding

### Outputs

- final object point clouds
- final CLIP embeddings
- crop images
- OpenLex3D export bundle

### Important invariant

By the end of clip stage, each surviving object has both:

- geometry
- semantic embedding

Objects missing one of these are filtered out.

## 5.5 Eval Stage

### Purpose

The eval stage runs external semantic evaluation on the clip-stage exports.

### Inputs

- clip-stage output root
- eval config

### Internal logic

1. Validate clip-stage metadata.
2. Release heavy GPU resources.
3. Launch an external evaluation script.
4. Pass evaluator settings through environment variables.

### Outputs

- evaluation directory with OpenLex3D results

### Important invariant

This stage does not compute new objects. It only consumes clip-stage exports.

## 6. Frame Ranking Logic

The same ranking idea is used to decide which object views are best for CLIP.

For each object appearance in each frame:

1. compute mask area
2. compute how close the mask center is to the image center
3. compute fill ratio = mask area / bounding-box area

Each metric is normalized across that object’s frames.
Then a weighted sum is computed:

`score = w_center * center + w_area * area + w_fill * fill_ratio`

Frames are ranked by descending score.

Interpretation:

- center term prefers centered views
- area term prefers large visible views
- fill-ratio term prefers compact, less fragmented masks

## 7. Object Filtering Logic

Some objects are intentionally discarded.

### Objects without CLIP features

If no valid CLIP crop can be generated for an object, the object is removed from:

- masks
- point clouds
- keypoint lists
- best-frame tables
- crop outputs

This means the final object set is biased toward objects with usable RGB evidence.

### Small objects in 3D merge

Objects with too few points can be skipped by 3D overlap merging. This prevents unstable merge decisions from tiny or noisy clouds.

## 8. Artifact Layout and Meaning

This section explains what the saved files mean semantically.

### Detect artifacts

- `detect_candidates.pkl.gz`
  - candidate list for tracking
- `detections/`
  - per-frame detection payloads for the sampled frames

### Track artifacts

- `track_result.pkl.gz`
  - tracked-result metadata pointing to tracked detections plus tracklet/keypoint metadata
- `tracked_detections/frame_XXXXXX.pkl.gz`
  - tracked per-frame detections keyed by global frame ID

Tracked detections are the stable stage contract. Geometry is rebuilt later in merge,
so track artifacts no longer include per-object point-cloud or voxel files.

Track-stage scratch files used internally during propagation and overlap resolution are cleaned up before the stage returns.

### Merge artifacts

- `merge_result.pkl.gz`
  - merged masks, merged geometry, keypoints, merge map

### Clip artifacts

- `clip_result.pkl.gz`
  - final masks, point clouds, CLIP features, crops, and merge map for the clip stage
- `segmented_pointcloud/object_XXXX.ply`
  - final object geometry
- `segmented_pointcloud/merged.ply`
  - all final object clouds combined
- `clip_features/object_XXXX.npy`
  - final embedding per object
- `clip_crops/object_XXXX/*.jpg`
  - images used to derive the embedding
- `openlex3d_output/<scene>/`
  - export bundle for semantic retrieval/evaluation

### Eval artifacts

- `openlex3d_eval/`
  - external evaluator output

## 9. Execution Model

The pipeline now assumes one end-to-end run per output directory.

- the user cannot run a subset of stages
- the code no longer uses per-stage metadata validation
- later stages consume the artifacts produced earlier in the same run directory

## 10. Legacy vs Current Staged Path

The file contains both:

- a staged path used by `run_batch()`
- an older single-pass helper path that does detect+track+merge+clip more directly

For understanding current behavior, prioritize the staged path:

- `run_batch()`
- detect stage
- track stage
- merge stage
- clip stage
- eval stage

Some flags exist mainly for the older path. When documenting behavior, prefer the staged interpretation unless the user explicitly asks about legacy helpers.

## 11. Common Failure Modes Another Agent Should Expect

### No frames found

Cause:

- dataset path or scene name is wrong
- subsampling removed everything

### No objects survive

Cause:

- detector generated poor candidates
- tracking selection was too strict
- merges collapsed too aggressively
- no valid CLIP crops were produced

### Eval silently skipped

Cause:

- dataset name has no OpenLex3D mapping
- external eval script is missing

## 12. Minimal “How To Reason About This Pipeline” Summary

If another agent only remembers one thing, it should remember this:

- `detect` prepares reusable mask candidates
- `track` turns candidates into tracked per-frame detections
- `merge` rebuilds geometry from tracked detections and runs scene-level 3D merging
- `clip` assigns semantic embeddings and writes final exports
- `eval` runs an external evaluator

The most important internal object is:

- `frame_id -> object_id -> mask`
- `object_id -> point_cloud`
- `object_id -> clip_feature`

Everything else exists to build, merge, validate, or export that object table.

## 13. Parameter Reference for `object_segmentation_minimal.yaml` and `object_segmentation_service.yaml`

`object_segmentation_minimal.yaml` now carries scene/run inputs and points at `object_segmentation_service.yaml` through `object_segmentation.config_path`. The shared service YAML keeps object-segmentation settings under the `object_segmentation` namespace, with stage-specific parameters under `object_segmentation.stages.*` and shared geometry settings under `object_segmentation.geometry.*`. Both the standalone pipeline and `LEXISG` load the same service config.


Descriptions below are concise and practical. They are intended for an agent deciding what changing a parameter will do.

### Top-Level Parameters

| Parameter | Description |
| --- | --- |
| `dataset` | Chooses the dataset loader and affects how scene files are discovered. |
| `scene_name` | Chooses which scene inside the dataset root is processed. |
| `data_path` | Root directory containing the dataset. |
| `subsample` | Uses every `n`-th frame before uniform sampling; larger values reduce sequence density. |
| `frame_sampling_mode` | Chooses whether sampled frames are controlled by `num_uniform_frames` (`count`) or `uniform_frame_stride` (`stride`). |
| `num_uniform_frames` | Number of frames sampled from the subsampled sequence when `frame_sampling_mode=count`. |
| `uniform_frame_stride` | Take every `k`-th subsampled frame when `frame_sampling_mode=stride`; must be positive in that mode. |
| `object_segmentation.config_path` | Relative or absolute path to the shared object-segmentation service config merged into this run config before the pipeline constructs its stages. |
| `object_segmentation.geometry.shared_geometry_voxel_size` | Voxel size used when compressing shared geometry before building pixel lookup indices. |
| `object_segmentation.stages.merge.max_backproj_depth` | Ignores depths beyond this value during merge-time mask backprojection; negative disables the cap. |
| `object_segmentation.stages.merge.object_pcd_per_frame_filter.*` | Controls per-frame point-cloud cleanup before object clouds are accumulated in the merge path. |
| `object_segmentation.stages.merge.object_pcd_voxel_size` | Voxel size used when merge-stage geometry accumulation downsamples rebuilt object point clouds; set `<= 0` to disable. |
| `object_segmentation.color_seed` | Seeds deterministic object coloring for saved geometry and overlays. |
| `object_segmentation.stages.clip.save_outputs` | Enables writing final geometry, CLIP features, crops, and OpenLex3D export artifacts. |
| `defaults` | Hydra defaults controlling config composition and logging overrides. |
| `log_dir` | Root output directory for all stage artifacts and copied config snapshots. |

### `object_segmentation.stages.detect`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.detect.method` | Detection backend family used to generate mask candidates; supported values are `amg` and `grounded_sam`. |
| `object_segmentation.stages.detect.model` | Specific model name inside the chosen detection backend. |

### `object_segmentation.stages.detect.post_filter`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.detect.post_filter.mask_ios_threshold` | Per-frame post-detection IoS threshold. When two masks overlap above this value, the larger covering mask is removed; negative disables the filter. |

### `object_segmentation.stages.detect.amg`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.detect.amg.apply_postprocessing` | Applies backend-specific postprocessing to SAM2-generated masks used by the AMG path. |

### `object_segmentation.stages.detect.grounded_sam`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.detect.grounded_sam.enable_ram` | Enables RAM-generated class proposals for grounded detection. |
| `object_segmentation.stages.detect.grounded_sam.gsa_path` | Optional path override for the Grounded-SAM workspace. |
| `object_segmentation.stages.detect.grounded_sam.ram_repo_root` | Optional path override for the RAM repository root. |
| `object_segmentation.stages.detect.grounded_sam.ram_checkpoint_path` | Optional path override for the RAM checkpoint. |
| `object_segmentation.stages.detect.grounded_sam.grounding_dino_config_path` | Optional path override for GroundingDINO config. |
| `object_segmentation.stages.detect.grounded_sam.grounding_dino_checkpoint_path` | Optional path override for GroundingDINO weights. |
| `object_segmentation.stages.detect.grounded_sam.sam2_model_id` | SAM2 model identifier used inside the grounded pipeline. |
| `object_segmentation.stages.detect.grounded_sam.sam2_config_path` | Optional local SAM2 config override. |
| `object_segmentation.stages.detect.grounded_sam.sam2_checkpoint_path` | Optional local SAM2 checkpoint override. |
| `object_segmentation.stages.detect.grounded_sam.sam2_apply_postprocessing` | Applies postprocessing to SAM2 masks in the grounded path. |
| `object_segmentation.stages.detect.grounded_sam.confidence_threshold` | Minimum confidence required to keep grounded detections. |
| `object_segmentation.stages.detect.grounded_sam.sam2_iou_threshold` | IoU threshold used when filtering/refining grounded SAM2 masks. |
| `object_segmentation.stages.detect.grounded_sam.box_threshold` | Confidence threshold for grounded boxes. |
| `object_segmentation.stages.detect.grounded_sam.text_threshold` | Text-match threshold for grounded text prompts. |
| `object_segmentation.stages.detect.grounded_sam.nms_threshold` | NMS threshold for grounded box suppression. |
| `object_segmentation.stages.detect.grounded_sam.max_box_area_ratio` | Rejects boxes that cover too much of the image. |
| `object_segmentation.stages.detect.grounded_sam.cpu_fallback_on_oom` | Falls back to CPU execution on CUDA out-of-memory. |
| `object_segmentation.stages.detect.grounded_sam.cpu_fallback_max_side` | Resize cap used during CPU fallback. |
| `object_segmentation.stages.detect.grounded_sam.add_background_classes` | Adds background-like classes to the text prompt set. |
| `object_segmentation.stages.detect.grounded_sam.add_classes` | Appends extra user-provided classes to the prompt set. |
| `object_segmentation.stages.detect.grounded_sam.remove_substrings` | Removes unwanted substrings from generated class names before prompting. |

### `object_segmentation.stages.detect.amg`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.detect.amg.max_side` | Resize cap for automatic mask generation; `0` keeps full resolution. |
| `object_segmentation.stages.detect.amg.oom_retry_max_side` | Smaller resize cap used when retrying after OOM. |
| `object_segmentation.stages.detect.amg.points_per_side` | Prompt grid density for automatic mask generation. |
| `object_segmentation.stages.detect.amg.points_per_batch` | Number of prompts processed per generation batch. |
| `object_segmentation.stages.detect.amg.pred_iou_thresh` | Minimum predicted IoU for keeping a generated mask. |
| `object_segmentation.stages.detect.amg.stability_score_thresh` | Minimum stability score for keeping a mask. |
| `object_segmentation.stages.detect.amg.stability_score_offset` | Offset used when computing stability scores. |
| `object_segmentation.stages.detect.amg.mask_threshold` | Threshold used to binarize masks. |
| `object_segmentation.stages.detect.amg.box_nms_thresh` | NMS threshold for generated mask boxes. |
| `object_segmentation.stages.detect.amg.crop_n_layers` | Number of crop pyramid layers for multi-scale mask generation. |
| `object_segmentation.stages.detect.amg.crop_nms_thresh` | NMS threshold across crop layers. |
| `object_segmentation.stages.detect.amg.crop_overlap_ratio` | Overlap ratio between neighboring crops. |
| `object_segmentation.stages.detect.amg.crop_n_points_downscale_factor` | Downscale factor for prompt density in deeper crop layers. |
| `object_segmentation.stages.detect.amg.point_grids` | Explicit prompt grids, if provided, replacing the default grid. |
| `object_segmentation.stages.detect.amg.min_mask_region_area` | Minimum connected region area retained in a generated mask. |
| `object_segmentation.stages.detect.amg.output_mode` | Mask output encoding mode. |
| `object_segmentation.stages.detect.amg.use_m2m` | Enables mask-to-mask refinement if supported by the backend. |
| `object_segmentation.stages.detect.amg.multimask_output` | Allows multiple mask hypotheses per prompt. |

### `object_segmentation.stages.track`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.track.method` | Tracking backend family used to propagate candidate masks. |
| `object_segmentation.stages.track.model` | Specific tracking model name. |
| `object_segmentation.stages.track.conf_threshold` | Confidence cutoff used by the tracking stage. |
| `object_segmentation.stages.track.min_detections` | Minimum evidence required before keeping a tracked object hypothesis. |

### `object_segmentation.stages.track.best_frame`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.track.best_frame.weights.center` | Weight for image-center proximity when ranking object views. |
| `object_segmentation.stages.track.best_frame.weights.area` | Weight for visible mask area when ranking object views. |
| `object_segmentation.stages.track.best_frame.weights.fill_ratio` | Weight for mask compactness inside its bounding box when ranking object views. |

### `object_segmentation.stages.track.sam2`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.track.sam2.apply_postprocessing` | Applies backend-specific SAM2 postprocessing during tracking. |
| `object_segmentation.stages.track.sam2.vos_optimized` | Enables VOS-oriented tracker behavior if supported. |
| `object_segmentation.stages.track.sam2.offload_state_to_cpu` | Controls whether SAM2 keeps tracker inference state tensors on CPU after initialization and updates to reduce GPU memory usage. |
| `object_segmentation.stages.track.sam2.offload_video_to_cpu` | Controls whether decoded video frames stay on CPU instead of GPU inside the SAM2 video predictor state. |

### `object_segmentation.stages.track.bidirectional`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.track.bidirectional.adjacent_frames` | Maximum temporal distance the bidirectional backend propagates from each seed frame. |
| `object_segmentation.stages.track.bidirectional.overlap_resolution` | Chooses how qualified IoU overlaps are handled in track stage: `merge` unions overlapping tracklets into one object ID, while `prune` removes one side of the overlap instead. IoS and covering-mask rules always remain prune-only. |
| `object_segmentation.stages.track.bidirectional.iou_threshold` | Track-stage pairwise IoU overlap-resolution threshold. Negative disables the IoU pass. |
| `object_segmentation.stages.track.bidirectional.covering_mask_threshold` | Track-stage covering-mask prune threshold. Negative disables the pass that removes a larger tracklet when multiple smaller masks explain it well. |
| `object_segmentation.stages.track.bidirectional.covering_mask_ios_threshold` | Optional IoS prefilter for the small masks used by covering-mask pruning. Negative disables the prefilter. |
| `object_segmentation.stages.track.bidirectional.ios_threshold` | Track-stage pairwise IoS prune threshold. Negative disables the IoS pass. |
| `object_segmentation.stages.track.bidirectional.ios_prune_removes_smaller` | Controls IoS prune direction. When `False`, IoS pruning removes the larger tracklet; when `True`, it removes the smaller one instead. |
| `object_segmentation.stages.track.bidirectional.min_overlap_frames` | Number of overlapping frames required before a tracker-side merge or prune decision is applied. |
| `object_segmentation.stages.track.bidirectional.clear_cuda_cache_on_add` | Clears CUDA cache after adding each seed mask to reduce fragmentation during batched propagation. |

### `object_segmentation.stages.merge`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.merge.merge_stage_strategies` | Ordered merge-strategy list for the merge stage. Default behavior uses `overlap`. |
| `object_segmentation.stages.merge.clip_stage_strategies` | Ordered merge-strategy list for the post-CLIP merge step. Default behavior uses `clip_aware_world`. |

### `object_segmentation.stages.merge.overlap_merge`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.merge.overlap_merge.voxel_size` | Voxel size used for overlap tests in the merge stage. |
| `object_segmentation.stages.merge.overlap_merge.overlap_threshold` | Minimum voxel IoU needed to merge two objects in the merge stage. Set negative to disable this merge strategy. |
| `object_segmentation.stages.merge.overlap_merge.min_points` | Minimum point count required before an object is considered for merge-stage 3D overlap merge. |

### `object_segmentation.stages.merge.clip_aware_overlap_merge`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.merge.clip_aware_overlap_merge.voxel_size` | Voxel size used for overlap tests in the clip-aware merge step. |
| `object_segmentation.stages.merge.clip_aware_overlap_merge.overlap_threshold` | Minimum voxel IoU needed before the clip-aware merge step considers two objects. Set negative to disable this merge strategy. |
| `object_segmentation.stages.merge.clip_aware_overlap_merge.min_points` | Minimum point count required before an object is considered for clip-aware overlap merge. |
| `object_segmentation.stages.merge.clip_aware_overlap_merge.clip_similarity_threshold` | In the clip stage, minimum cosine similarity between object CLIP features required after the 3D overlap gate passes. Set negative to disable the CLIP-similarity gate while keeping the overlap gate enabled. |

### `object_segmentation.stages.clip`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.clip.top_n_frames` | Number of best object views averaged to create each CLIP embedding. |
| `object_segmentation.stages.clip.mask_pad_ratio` | Relative crop expansion around the object box; negative disables ratio-based padding. |
| `object_segmentation.stages.clip.mask_pad_pixel` | Fixed-pixel crop padding; negative disables pixel-based padding. |
| `object_segmentation.stages.clip.mask_mode` | Background style outside the mask before CLIP encoding: keep original, white, or black. |

### `object_segmentation.stages.eval`

| Parameter | Description |
| --- | --- |
| `object_segmentation.stages.eval.device` | Device string forwarded to the external evaluator. |
| `object_segmentation.stages.eval.save_object_labels` | Asks the external evaluator to save object-level predicted labels when supported. |

## 14. Practical Notes for Another Agent

- If you need to change object sampling behavior, check `subsample`, `frame_sampling_mode`, `num_uniform_frames`, and `uniform_frame_stride` first.
- If you need to reduce duplicate objects, inspect `object_segmentation.stages.track.bidirectional.*`, `object_segmentation.stages.merge.overlap_merge.*`, and `object_segmentation.stages.merge.clip_aware_overlap_merge.*`.
- If objects disappear unexpectedly, check the CLIP filtering step, because objects without usable CLIP features are removed.
- If later stages refuse to run, inspect stage metadata and reduced-config mismatch before debugging anything else.
- If semantic outputs look poor, check frame ranking and CLIP crop settings before touching 3D merging.

## 15. Most Important Implementation Facts in One Page

- The pipeline is stage-based and runs the full fixed stage order.
- `run_batch()` executes detect, track, merge, clip, and eval once in order.
- `detect` produces reusable mask candidates, not final objects.
- `track` writes tracked per-frame detections in a backend-independent format.
- `track` resolves 2D overlap conflicts by either pruning or merging tracklets before saving tracked detections.
- `merge` rebuilds object geometry from tracked detections and then runs scene-level 3D merging.
- object geometry is transformed into world coordinates during merge before final downstream use.
- `clip` creates semantic embeddings from top-ranked masked crops.
- the CLIP-stage post-merge is also strategy-driven.
- objects without CLIP features are removed.
- final exports are object point clouds plus embedding tables.
- `eval` is external and only consumes clip-stage outputs.
