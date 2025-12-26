"""Logging utilities for MLflow."""

import subprocess
from typing import Any, Dict

import mlflow
from omegaconf import DictConfig


def setup_mlflow(cfg: DictConfig) -> None:
    """Setup MLflow tracking.

    Args:
        cfg: Hydra configuration
    """
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.project_name)


def log_params(params: Dict[str, Any]) -> None:
    """Log hyperparameters to MLflow.

    Args:
        params: Dictionary of parameters
    """
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: int = None) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Dictionary of metric names and values
        step: Step/epoch number
    """
    mlflow.log_metrics(metrics, step=step)


def log_git_commit() -> None:
    """Log git commit hash to MLflow."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        mlflow.set_tag("git_commit", commit)
    except Exception as e:
        print(f"Could not log git commit: {e}")


def log_config(cfg: DictConfig) -> None:
    """Log entire configuration to MLflow.

    Args:
        cfg: Hydra configuration
    """
    from video_captioning.utils.config import get_config_dict

    params = get_config_dict(cfg)
    # Flatten nested dict for MLflow
    flat_params = {}

    def flatten(d, parent_key=""):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_params[new_key] = str(v)

    flatten(params)
    log_params(flat_params)
    log_git_commit()
