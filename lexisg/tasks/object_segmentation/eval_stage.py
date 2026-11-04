from __future__ import annotations

import gc
import os
import re
import subprocess

import torch
from loguru import logger

from lexisg.utils.config import get_object_segmentation_stage_config

class EvalStage:
    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    _TQDM_PROGRESS_RE = re.compile(r"\b\d{1,3}%\|")

    def __init__(
        self,
        config,
        dataset: str,
        scene_name: str,
        data_path: str,
        clip_dir: str,
        eval_dir: str,
    ):
        self.config = config
        self.dataset = str(dataset)
        self.scene_name = str(scene_name)
        self.data_path = str(data_path)
        self.clip_dir = str(clip_dir)
        self.eval_dir = str(eval_dir)
        self.eval_cfg = get_object_segmentation_stage_config(self.config, "eval")

    @classmethod
    def _is_tqdm_progress_line(cls, raw_line: str) -> bool:
        if not raw_line:
            return False
        if "\r" in raw_line:
            return True
        message = cls._ANSI_ESCAPE_RE.sub("", raw_line).strip()
        if not message:
            return False
        return bool(cls._TQDM_PROGRESS_RE.search(message) and "|" in message and ("it/s" in message or "s/it" in message))

    def _openlex3d_dataset_name(self):
        dataset_name = self.dataset.lower()
        dataset_map = {
            "replica": "replica",
            "scannetpp": "scannetpp",
            "hm3d": "hm3d",
        }
        return dataset_map.get(dataset_name)

    def run(self):
        output_dir = os.path.join(self.eval_dir, "openlex3d_eval")

        openlex_dataset = self._openlex3d_dataset_name()
        if openlex_dataset is None:
            logger.info(f"Skipping OpenLex3D eval for dataset={self.dataset} (no mapping configured).")
            return {"eval_dir": output_dir}

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        script_path = os.path.join(repo_root, "scripts", "eval_openlex3d_top5.sh")
        if not os.path.isfile(script_path):
            logger.warning(f"OpenLex3D eval script not found: {script_path}")
            return {"eval_dir": output_dir}

        cmd = [script_path, "--dataset", openlex_dataset, "--output-dir", output_dir, self.clip_dir, self.scene_name]
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["OPENLEX_DEVICE"] = str(self.eval_cfg.get("device", "cpu"))
        env["OPENLEX_SAVE_OBJECT_LABELS"] = "1" if bool(self.eval_cfg.get("save_object_labels", True)) else "0"
        replica_path = self.eval_cfg.get("replica_path")
        scannetpp_path = self.eval_cfg.get("scannetpp_path")
        hm3d_path = self.eval_cfg.get("hm3d_path")
        if replica_path:
            env["OPENLEX_REPLICA_PATH"] = str(replica_path)
        if scannetpp_path:
            env["OPENLEX_SCANNETPP_PATH"] = str(scannetpp_path)
        if hm3d_path:
            env["OPENLEX_HM3D_PATH"] = str(hm3d_path)
        logger.info(f"Running OpenLex3D top-5 eval: {' '.join(cmd)}")
        try:
            with subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if self._is_tqdm_progress_line(line):
                        continue
                    message = line.rstrip()
                    if message:
                        logger.info(f"[openlex3d] {message}")
                return_code = proc.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
        except Exception as exc:
            logger.warning(f"OpenLex3D eval failed: {exc}")
        return {"eval_dir": output_dir}
