"""Configuration loader. All paths are resolved relative to the project root,
never to the user's home directory, so the repo is portable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

_REQUIRED_SECTIONS = ("project", "paths", "dataset", "labels", "retrieval", "generation", "logging")


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate configs/config.yaml."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {config_path} did not parse to a mapping")
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"Config is missing required sections: {missing}")
    return cfg


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved, project-relative filesystem locations."""

    root: Path
    raw: Path
    interim: Path
    processed: Path
    train: Path
    validation: Path
    test: Path
    charts: Path
    reports: Path
    predictions: Path

    def ensure_dirs(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.train,
                  self.validation, self.test, self.charts, self.reports, self.predictions):
            p.mkdir(parents=True, exist_ok=True)


def get_paths(cfg: dict[str, Any]) -> ProjectPaths:
    """Build ProjectPaths from the 'paths' section of the config."""
    paths_cfg = cfg["paths"]
    required_keys = ("data_raw", "data_interim", "data_processed", "data_train",
                     "data_validation", "data_test", "outputs_charts",
                     "outputs_reports", "outputs_predictions")
    missing = [k for k in required_keys if k not in paths_cfg]
    if missing:
        raise KeyError(f"Config paths section missing keys: {missing}")
    return ProjectPaths(
        root=PROJECT_ROOT,
        raw=PROJECT_ROOT / paths_cfg["data_raw"],
        interim=PROJECT_ROOT / paths_cfg["data_interim"],
        processed=PROJECT_ROOT / paths_cfg["data_processed"],
        train=PROJECT_ROOT / paths_cfg["data_train"],
        validation=PROJECT_ROOT / paths_cfg["data_validation"],
        test=PROJECT_ROOT / paths_cfg["data_test"],
        charts=PROJECT_ROOT / paths_cfg["outputs_charts"],
        reports=PROJECT_ROOT / paths_cfg["outputs_reports"],
        predictions=PROJECT_ROOT / paths_cfg["outputs_predictions"],
    )