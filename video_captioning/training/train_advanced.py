"""Advanced (main) model training entrypoint.

Implements training for the Transformer-based model with the input format:
  <VIDEO_START> [frame tokens] <VIDEO_END> [text tokens]

Call via:
  python -m video_captioning.commands train_advanced [hydra overrides...]

This mirrors train_baseline.py but composes the `advanced` config.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from video_captioning.data.dataset import VideoDataModule
from video_captioning.models.model import TransformerVideoCaptioningModel, VisionEncoder
from video_captioning.training.lightning_module_advanced import AdvancedVideoCaptioningLightning
from video_captioning.training.train_baseline import setup_logging
from video_captioning.utils.experiment_tracking import log_hparams_and_code_version
from video_captioning.utils.mlflow_export import export_run_to_plots

logger = logging.getLogger(__name__)


def _patch_pl_combined_loader_len() -> None:
    """Work around PyTorch Lightning 2.6.0 regression.

    In PL 2.6.0, CombinedLoader.__len__ raises RuntimeError unless iter(combined_loader)
    was called first. The training loop calls len() during reset before iterating,
    causing a crash.
    """

    try:
        from pytorch_lightning.utilities.combined_loader import (
            CombinedLoader,
            _MaxSize,
            _MaxSizeCycle,
            _MinSize,
            _Sequential,
        )

        if getattr(CombinedLoader, "__len__", None) is None:
            return

        # Only patch the buggy implementation.
        try:
            _ = CombinedLoader.__len__
        except Exception:
            return

        def _safe_len(self: CombinedLoader) -> int:  # type: ignore[name-defined]
            mode = getattr(self, "_mode", "min_size")
            flattened = getattr(self, "_flattened", [])
            limits = getattr(self, "_limits", None)
            mode_cls = {
                "min_size": _MinSize,
                "max_size": _MaxSize,
                "max_size_cycle": _MaxSizeCycle,
                "sequential": _Sequential,
            }.get(mode, _MinSize)
            return len(mode_cls(flattened, limits=limits))

        # Detect the exact 2.6.0 behavior and patch.
        # The buggy version raises: "Please call `iter(combined_loader)` first."
        try:
            # Construct a dummy CombinedLoader and exercise __len__ without iter.
            from torch.utils.data import DataLoader, TensorDataset

            dummy = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)
            cl = CombinedLoader(dummy, mode="min_size")
            try:
                _ = len(cl)
                return  # already ok
            except RuntimeError as exc:
                if "iter(combined_loader)" not in str(exc):
                    return

            CombinedLoader.__len__ = _safe_len  # type: ignore[assignment]
            logger.warning("Applied PL CombinedLoader.__len__ compatibility patch")
        except Exception:
            return
    except Exception:
        return


def _translate_overrides(hydra_overrides: tuple[str, ...]) -> list[str]:
    translated: list[str] = []
    for o in hydra_overrides:
        if o.startswith("dataloader."):
            translated.append("dataset." + o)
        elif o.startswith("tokenizer."):
            translated.append("dataset." + o)
        else:
            translated.append(o)
    return translated


def _setup_loggers(config: DictConfig) -> list[Any]:
    loggers: list[Any] = []

    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if local_rank != 0:
        return loggers

    if config.logging.use_mlflow:
        try:
            mlflow_logger = MLFlowLogger(
                experiment_name=config.logging.experiment_name,
                tracking_uri=config.logging.tracking_uri,
                run_name="advanced",
            )
            loggers.append(mlflow_logger)
            logger.info("MLflow logger initialized")
        except Exception as exc:
            logger.warning(f"Failed to initialize MLflow logger: {exc}")

    return loggers


def _setup_callbacks(config: DictConfig, output_dir: Path, has_logger: bool) -> list[Any]:
    callbacks: list[Any] = []

    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="advanced-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            verbose=True,
        )
    )

    if config.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(config.training.early_stopping_patience),
                verbose=True,
            )
        )

    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return callbacks


def _create_model_and_encoder(config: DictConfig, special_tokens: dict[str, int]):
    logger.info("Creating vision encoder...")
    vision_encoder = VisionEncoder(
        model_name=str(config.model.vision_encoder.name),
        pretrained=bool(config.model.vision_encoder.pretrained),
        freeze=bool(config.model.vision_encoder.freeze),
    )

    logger.info("Creating transformer model...")
    model = TransformerVideoCaptioningModel(
        vocab_size=int(config.dataset.tokenizer.vocab_size),
        d_model=int(config.model.embedding.d_model),
        num_layers=int(config.model.transformer.num_layers),
        num_heads=int(config.model.transformer.num_heads),
        ffn_dim=int(config.model.transformer.ffn_dim),
        dropout=float(config.model.transformer.dropout),
        vision_encoder_output_dim=int(vision_encoder.output_dim),
        max_seq_len=int(config.model.transformer.max_seq_len),
        special_tokens=special_tokens,
    )

    return model, vision_encoder


def train_from_config(config: DictConfig) -> None:
    setup_logging(config.logging.level)

    _patch_pl_combined_loader_len()

    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]

    output_dir = repo_root / str(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    pl.seed_everything(int(config.training.seed), workers=True)

    # Data
    data_module = VideoDataModule(
        json_dir=config.dataset.paths.json_dir,
        videos_dir=config.dataset.paths.videos_dir,
        dataset_type="advanced",
        target_fps=float(config.dataset.advanced.target_fps),
        image_size=int(config.dataset.image.size),
        batch_size=int(config.dataset.dataloader.batch_size),
        num_workers=int(config.dataset.dataloader.num_workers),
        persistent_workers=bool(getattr(config.dataset.dataloader, "persistent_workers", False)),
        prefetch_factor=int(getattr(config.dataset.dataloader, "prefetch_factor", 2)),
        multiprocessing_context=str(
            getattr(config.dataset.dataloader, "multiprocessing_context", "spawn")
        ),
        pin_memory=bool(getattr(config.dataset.dataloader, "pin_memory", True)),
        cache_frames=bool(getattr(config.dataset.dataloader, "cache_frames", False)),
        cache_max_items=int(getattr(config.dataset.dataloader, "cache_max_items", 0)),
        cache_policy=str(getattr(config.dataset.dataloader, "cache_policy", "lru")),
        disk_cache_dir=getattr(config.dataset.dataloader, "disk_cache_dir", None),
        disk_cache_mmap=bool(getattr(config.dataset.dataloader, "disk_cache_mmap", True)),
        max_caption_length=int(config.dataset.dataloader.max_caption_length),
        tokenizer_vocab_size=int(config.dataset.tokenizer.vocab_size),
        tokenizer_model_dir=str(config.dataset.tokenizer.model_dir),
    )

    data_module.setup(stage="fit")
    tokenizer = data_module.tokenizer
    special_tokens = tokenizer.get_special_tokens()

    model, vision_encoder = _create_model_and_encoder(config, special_tokens=special_tokens)

    # Generation settings (same schema as baseline)
    gen = getattr(config, "generation", {})
    beam_cfg = getattr(gen, "beam_search_settings", {})
    topk_cfg = getattr(gen, "topk_settings", {})

    beam_size = int(getattr(beam_cfg, "beam_size", getattr(gen, "beam_size", 5)))
    length_penalty = float(getattr(beam_cfg, "length_penalty", getattr(gen, "length_penalty", 1.2)))
    max_length = int(getattr(gen, "max_length", 100))
    temperature = float(getattr(gen, "temperature", 1.0))
    top_k = int(getattr(topk_cfg, "k", getattr(gen, "top_k", 50)))

    lightning_module = AdvancedVideoCaptioningLightning(
        model=model,
        vision_encoder=vision_encoder,
        vocab_size=int(config.dataset.tokenizer.vocab_size),
        tokenizer=tokenizer,
        learning_rate=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
        optimizer=str(config.training.optimizer),
        pad_token_id=int(getattr(tokenizer, "pad_id", special_tokens.get("pad", 0))),
        beam_size=beam_size,
        max_length=max_length,
        temperature=temperature,
        length_penalty=length_penalty,
        top_k=top_k,
        val_generate_captions=bool(
            getattr(config, "validation", {}).get("generate_captions", True)
        ),
        val_generation_mode=str(getattr(config, "validation", {}).get("generation_mode", "beam")),
        val_max_samples=int(getattr(config, "validation", {}).get("max_samples", 256)),
        val_compute_metrics=bool(getattr(config, "validation", {}).get("compute_metrics", True)),
        val_compute_bleu_4=bool(getattr(config, "validation", {}).get("compute_bleu_4", True)),
        val_compute_meteor=bool(getattr(config, "validation", {}).get("compute_meteor", False)),
    )

    loggers = _setup_loggers(config)
    callbacks = _setup_callbacks(config, output_dir=output_dir, has_logger=len(loggers) > 0)

    trainer = pl.Trainer(
        max_epochs=int(config.training.max_epochs),
        accelerator=str(config.training.accelerator),
        devices=config.training.devices,
        precision=str(config.training.precision),
        strategy=str(config.training.strategy),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=int(config.logging.log_every_n_steps),
        val_check_interval=float(config.training.val_check_interval),
        num_sanity_val_steps=int(getattr(config.training, "num_sanity_val_steps", 0)),
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=float(config.training.gradient_clip_val),
    )

    logger.info(f"Total pre-fit init time: {time.perf_counter() - t0:.3f}s")

    # Log resolved hyperparameters + git commit into MLflow (and any other active loggers).
    config_dict = OmegaConf.to_container(config, resolve=True)
    try:
        _ = log_hparams_and_code_version(
            trainer=trainer, config_dict=dict(config_dict), repo_root=repo_root
        )
    except Exception:
        pass

    # Train and export MLflow plots/logs into `plots/` on completion (rank 0 only).
    run_id: str | None = None
    tracking_uri = str(getattr(config.logging, "tracking_uri", ""))
    experiment_name = str(getattr(config.logging, "experiment_name", "advanced"))
    try:
        for lg in loggers or []:
            if isinstance(lg, MLFlowLogger):
                run_id = str(getattr(lg, "run_id", "") or "")
                break

        trainer.fit(lightning_module, datamodule=data_module)
    finally:
        if run_id and tracking_uri:
            try:
                export_run_to_plots(
                    tracking_uri=tracking_uri,
                    experiment_name=experiment_name,
                    run_id=run_id,
                    plots_root=repo_root / "plots",
                    metric_keys=[
                        "train_loss_epoch",
                        "train_loss_step",
                        "val_loss",
                        "val_bleu_4",
                        "val_meteor",
                        "learning_rate",
                        "val_pred_empty_frac",
                        "val_pred_avg_len_words",
                        "val_pred_unique_frac",
                    ],
                )
                exported_dir = repo_root / "plots" / experiment_name / run_id
                logger.info("Exported MLflow plots/logs to: %s", exported_dir)
            except Exception as exc:
                logger.warning(f"Failed to export MLflow plots/logs: {exc}")

    final_model_path = output_dir / "advanced_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")


def run_train_advanced(*hydra_overrides: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    translated = _translate_overrides(tuple(hydra_overrides))

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="advanced", overrides=translated)

    OmegaConf.resolve(config)
    train_from_config(config)
