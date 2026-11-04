import copy
import os
import re

import yaml
from omegaconf import OmegaConf

__BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "config")
_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")
_OBJECT_SEG_STAGE_ALIASES = {
    "detect": "detection",
    "track": "tracking",
    "merge": "merging",
    "clip": "clip",
    "eval": "eval",
}


def get_path():
    return __BASE_DIR


def _deep_merge_dicts(base, override):
    if not isinstance(base, dict):
        return copy.deepcopy(override)
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _lookup_interpolation_key(root, key, cwd):
    if key == "hydra:runtime.cwd":
        return cwd
    current = root
    for part in key.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _resolve_known_interpolations(value, root, cwd):
    if isinstance(value, dict):
        return {k: _resolve_known_interpolations(v, root, cwd) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_known_interpolations(v, root, cwd) for v in value]
    if not isinstance(value, str):
        return value

    def _replace(match):
        resolved = _lookup_interpolation_key(root, match.group(1), cwd)
        if resolved is None:
            return match.group(0)
        return str(resolved)

    return _INTERPOLATION_RE.sub(_replace, value)


def _as_plain_dict(config):
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config if isinstance(config, dict) else {}


def get_object_segmentation_namespace(config):
    plain = _as_plain_dict(config)
    namespace = plain.get("object_segmentation", {})
    return namespace if isinstance(namespace, dict) else {}


def get_object_segmentation_value(config, key, default=None):
    plain = _as_plain_dict(config)
    namespace = get_object_segmentation_namespace(plain)
    if key in namespace:
        return namespace.get(key)
    return plain.get(key, default)


def get_object_segmentation_geometry_config(config):
    plain = _as_plain_dict(config)
    namespace = get_object_segmentation_namespace(plain)
    geometry_cfg = namespace.get("geometry", {})
    if not isinstance(geometry_cfg, dict):
        geometry_cfg = {}
    legacy_cfg = {
        key: plain[key]
        for key in ("shared_geometry_voxel_size",)
        if key in plain
    }
    return _deep_merge_dicts(legacy_cfg, geometry_cfg)


def get_object_segmentation_stage_config(config, stage_name):
    plain = _as_plain_dict(config)
    namespace = get_object_segmentation_namespace(plain)
    stages_cfg = namespace.get("stages", {})
    if not isinstance(stages_cfg, dict):
        stages_cfg = {}
    legacy_tasks = plain.get("tasks", {}) if isinstance(plain.get("tasks", {}), dict) else {}
    legacy_key = _OBJECT_SEG_STAGE_ALIASES.get(str(stage_name), str(stage_name))
    legacy_cfg = legacy_tasks.get(legacy_key, {}) if isinstance(legacy_tasks, dict) else {}
    stage_cfg = stages_cfg.get(stage_name, {}) if isinstance(stages_cfg.get(stage_name, {}), dict) else {}
    if str(stage_name) == "merge":
        legacy_merge_cfg = {
            key: plain[key]
            for key in ("max_backproj_depth", "object_pcd_per_frame_filter", "object_pcd_voxel_size")
            if key in plain
        }
        legacy_cfg = _deep_merge_dicts(legacy_merge_cfg, legacy_cfg or {})
    return _deep_merge_dicts(legacy_cfg or {}, stage_cfg or {})


def load_object_segmentation_service_config(config, config_dir=None):
    base_config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else copy.deepcopy(config)
    object_segmentation_cfg = base_config.get("object_segmentation", {})
    if not isinstance(object_segmentation_cfg, dict):
        return base_config

    config_path = object_segmentation_cfg.get("config_path")
    if not config_path:
        return base_config

    if os.path.isabs(config_path):
        resolved_path = config_path
    else:
        base_dir = config_dir or os.path.join(get_path(), "object_segmentation")
        resolved_path = os.path.join(base_dir, str(config_path))

    with open(resolved_path, "r") as f:
        shared_config = yaml.safe_load(f) or {}

    merged = _deep_merge_dicts(shared_config, base_config)
    return _resolve_known_interpolations(merged, merged, os.getcwd())
