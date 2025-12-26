"""Experiment tracking helpers.

Keeps MLflow runs self-contained by logging:
- resolved hyperparameters (flattened)
- git commit id (+ dirty flag)

Also provides optional export of plots/logs into the repository `plots/` folder.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def _is_primitive(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def flatten_for_mlflow(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings/lists into a dot-key dict suitable for MLflow params."""
    out: dict[str, Any] = {}

    if isinstance(data, Mapping):
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_for_mlflow(v, key))
        return out

    if isinstance(data, (list, tuple)):
        # MLflow params are scalar-ish; keep lists as JSON strings.
        key = prefix or "value"
        try:
            out[key] = json.dumps(list(data))
        except Exception:
            out[key] = str(data)
        return out

    key = prefix or "value"
    out[key] = data if _is_primitive(data) else str(data)
    return out


def get_git_commit_info(repo_root: Path) -> dict[str, str]:
    """Best-effort git commit info."""
    repo_root = Path(repo_root)

    def _run(args: list[str]) -> str:
        try:
            res = subprocess.run(
                args,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            )
            return (res.stdout or "").strip()
        except Exception:
            return ""

    full = _run(["git", "rev-parse", "HEAD"])
    short = _run(["git", "rev-parse", "--short", "HEAD"])

    dirty = "unknown"
    try:
        res = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=str(repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dirty = "true" if res.returncode != 0 else "false"
    except Exception:
        dirty = "unknown"

    if not full and not short:
        return {"git_commit": "unknown", "git_commit_short": "unknown", "git_dirty": dirty}

    return {
        "git_commit": full or short or "unknown",
        "git_commit_short": short or (full[:7] if full else "unknown"),
        "git_dirty": dirty,
    }


def log_hparams_and_code_version(
    *, trainer: Any, config_dict: dict[str, Any], repo_root: Path
) -> dict[str, str]:
    """Log flattened hparams + git commit into the active logger(s).

    Returns commit info dict.
    """
    commit = get_git_commit_info(Path(repo_root))

    flat = flatten_for_mlflow(config_dict)
    # Add code version to params.
    flat.update(commit)

    try:
        # Lightning will fan-out to all loggers.
        if getattr(trainer, "logger", None) is not None:
            trainer.logger.log_hyperparams(flat)  # type: ignore[attr-defined]
    except Exception:
        pass

    return commit
