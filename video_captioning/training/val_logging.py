"""Shared logging helpers for Lightning modules.

Goal: keep MLflow metric keys identical across baseline and advanced training
without duplicating logic in each LightningModule.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from video_captioning.evaluation.metrics import CaptioningMetrics


def log_learning_rate(pl_module: Any) -> None:
    """Log current learning rate as `learning_rate` on epoch end."""
    try:
        opt = pl_module.optimizers()
    except Exception:
        return

    # Lightning may return a list if multiple optimizers are used.
    optimizers: Iterable[Any]
    if isinstance(opt, (list, tuple)):
        optimizers = opt
    else:
        optimizers = [opt]

    for optimizer in optimizers:
        try:
            for pg in optimizer.param_groups:
                lr = pg.get("lr", None)
                if lr is not None:
                    pl_module.log("learning_rate", float(lr), on_epoch=True)
                    return
        except Exception:
            continue


@dataclass
class ValMetricsLogger:
    """Collect generated captions during validation and log BLEU/METEOR once/epoch."""

    tokenizer: Any
    val_generate_captions: bool = True
    val_generation_mode: str = "beam"
    val_max_samples: int = 256
    val_compute_metrics: bool = True
    val_compute_bleu_4: bool = True
    val_compute_meteor: bool = False

    metrics: CaptioningMetrics = field(default_factory=CaptioningMetrics)
    predictions: list[str] = field(default_factory=list)
    references: list[list[str]] = field(default_factory=list)

    _t_after_validation_epoch_end: float | None = None

    def remaining(self) -> int:
        return int(self.val_max_samples) - len(self.predictions)

    def reset(self) -> None:
        self.predictions = []
        self.references = []

    def should_collect(self) -> bool:
        return bool(
            self.val_compute_metrics
            and self.val_generate_captions
            and int(self.val_max_samples) > 0
        )

    def take_count(self, batch_size: int, all_captions: Any) -> int:
        if not self.should_collect():
            return 0
        if not all_captions:
            return 0
        remaining = self.remaining()
        if remaining <= 0:
            return 0
        return min(int(batch_size), int(remaining))

    def add(self, preds: list[str], refs: list[list[str]]) -> None:
        if not preds:
            return
        self.predictions.extend(list(preds))
        self.references.extend(list(refs))

    def on_validation_epoch_end(self, pl_module: Any) -> None:
        """Call from LightningModule.on_validation_epoch_end."""
        pl_module.log("epoch", int(getattr(pl_module, "current_epoch", 0)))

        if not self.should_collect():
            self.reset()
            return
        if not self.predictions:
            return

        # Basic generation diagnostics to quickly spot collapse (e.g., empty captions)
        # or near-deterministic repetition.
        preds = list(self.predictions)
        n = len(preds)
        if n > 0:
            empty = sum(1 for p in preds if not str(p).strip())
            uniq = len(set(str(p) for p in preds))
            lengths = [len(str(p).split()) for p in preds]
            avg_len = float(sum(lengths)) / float(n) if n else 0.0
            empty_frac = float(empty) / float(n) if n else 0.0
            uniq_frac = float(uniq) / float(n) if n else 0.0

            pl_module.log("val_pred_empty_frac", float(empty_frac), on_epoch=True, prog_bar=False)
            pl_module.log("val_pred_avg_len_words", float(avg_len), on_epoch=True, prog_bar=False)
            pl_module.log("val_pred_unique_frac", float(uniq_frac), on_epoch=True, prog_bar=False)

        scores = self.metrics.evaluate_corpus(self.predictions, self.references)
        if self.val_compute_bleu_4:
            pl_module.log("val_bleu_4", float(scores["bleu_4"]), on_epoch=True, prog_bar=True)
        if self.val_compute_meteor:
            pl_module.log("val_meteor", float(scores["meteor"]), on_epoch=True, prog_bar=False)

        self.reset()
        self._t_after_validation_epoch_end = time.perf_counter()

    def on_train_epoch_start(self, pl_module: Any) -> None:
        """Call from LightningModule.on_train_epoch_start to log the gap after val."""
        if self._t_after_validation_epoch_end is None:
            return

        gap = time.perf_counter() - self._t_after_validation_epoch_end
        if getattr(getattr(pl_module, "trainer", None), "is_global_zero", True):
            pl_module.log(
                "val_to_next_train_epoch_gap_sec", float(gap), on_epoch=True, prog_bar=False
            )

        self._t_after_validation_epoch_end = None
