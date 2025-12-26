"""Export MLflow run metrics/params into the repository `plots/` folder.

This satisfies the assignment requirement to keep plots/logs inside the repo.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ExportResult:
    run_id: str
    out_dir: Path
    metric_keys: list[str]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_run_to_plots(
    *,
    tracking_uri: str,
    experiment_name: str,
    run_id: str,
    plots_root: Path,
    metric_keys: Iterable[str] | None = None,
) -> ExportResult:
    """Export MLflow run data to `plots/<experiment>/<run_id>/`.

    Writes:
    - params.json, tags.json, run_info.json
    - metrics/<key>.csv and metrics.csv
    - plots/<key>.png (one per metric)
    """

    # Local import to keep module importable without mlflow in minimal contexts.
    # matplotlib is the simplest portable option for PNG output.
    import matplotlib
    from mlflow.tracking import MlflowClient

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    out_dir = Path(plots_root) / str(experiment_name) / str(run_id)
    _safe_mkdir(out_dir)

    _write_json(
        out_dir / "run_info.json",
        {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
        },
    )
    _write_json(out_dir / "params.json", dict(run.data.params))
    _write_json(out_dir / "tags.json", dict(run.data.tags))

    # Determine metric keys to export.
    keys: list[str]
    if metric_keys is None:
        keys = sorted(run.data.metrics.keys())
    else:
        # Keep only those present.
        want = list(metric_keys)
        keys = [k for k in want if k in run.data.metrics]

    all_rows: list[dict[str, Any]] = []

    metrics_dir = out_dir / "metrics"
    plots_dir = out_dir / "plots"
    _safe_mkdir(metrics_dir)
    _safe_mkdir(plots_dir)

    for key in keys:
        hist = client.get_metric_history(run_id, key)
        if not hist:
            continue

        rows = [
            {
                "metric": key,
                "step": int(m.step),
                "timestamp": int(m.timestamp),
                "value": float(m.value),
            }
            for m in hist
        ]
        all_rows.extend(rows)

        # Write per-metric CSV
        with open(metrics_dir / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "step", "timestamp", "value"])
            w.writeheader()
            w.writerows(rows)

        # Plot
        xs = [r["step"] for r in rows]
        ys = [r["value"] for r in rows]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, linewidth=2)
        plt.title(key)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{key}.png", dpi=140)
        plt.close()

    # Write combined CSV
    if all_rows:
        all_rows.sort(key=lambda r: (r["metric"], r["step"], r["timestamp"]))
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "step", "timestamp", "value"])
            w.writeheader()
            w.writerows(all_rows)

    return ExportResult(run_id=run_id, out_dir=out_dir, metric_keys=keys)
