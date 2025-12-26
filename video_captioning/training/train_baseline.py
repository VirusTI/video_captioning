"""Baseline training entrypoint.

This replaces the former repository-root `train_baseline.py` script.
It is designed to be called from the single CLI entrypoint:

  python -m video_captioning.commands train_baseline [hydra overrides...]

The implementation composes Hydra config programmatically, avoiding subprocess.
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
from video_captioning.models.model import BaselineVideoCaptioningModel, VisionEncoder
from video_captioning.training.lightning_module import VideoCaptioningLightning
from video_captioning.utils.experiment_tracking import log_hparams_and_code_version
from video_captioning.utils.mlflow_export import export_run_to_plots

logger = logging.getLogger(__name__)


def _patch_pl_combined_loader_len() -> None:
    """Work around PyTorch Lightning 2.6.0 CombinedLoader.__len__ regression."""

    try:
        from pytorch_lightning.utilities.combined_loader import (
            CombinedLoader,
            _MaxSize,
            _MaxSizeCycle,
            _MinSize,
            _Sequential,
        )

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

        # Patch only if the buggy behavior is present.
        from torch.utils.data import DataLoader, TensorDataset

        dummy = DataLoader(TensorDataset(torch.arange(2)), batch_size=1)
        cl = CombinedLoader(dummy, mode="min_size")
        try:
            _ = len(cl)
            return
        except RuntimeError as exc:
            if "iter(combined_loader)" not in str(exc):
                return

        CombinedLoader.__len__ = _safe_len  # type: ignore[assignment]
        logger.warning("Applied PL CombinedLoader.__len__ compatibility patch")
    except Exception:
        return


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, log_level.upper()),
    )


def create_model_and_encoder(
    config: DictConfig,
) -> tuple[BaselineVideoCaptioningModel, VisionEncoder, int]:
    logger.info("Creating vision encoder...")
    vision_encoder = VisionEncoder(
        model_name=config.model.vision_encoder.model,
        pretrained=config.model.vision_encoder.pretrained,
        freeze=config.model.vision_encoder.freeze,
    )
    logger.info(f"Vision encoder output dimension: {vision_encoder.output_dim}")

    logger.info("Creating baseline captioning model...")
    model = BaselineVideoCaptioningModel(
        vocab_size=config.dataset.tokenizer.vocab_size,
        d_model=config.model.embedding.d_model,
        num_layers=config.model.lstm.num_layers,
        dropout=config.model.lstm.dropout,
        vision_encoder_output_dim=vision_encoder.output_dim,
    )

    return model, vision_encoder, config.dataset.tokenizer.vocab_size


def setup_callbacks(config: DictConfig, output_dir: Path, has_logger: bool = True) -> list[Any]:
    callbacks: list[Any] = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="baseline-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    if config.training.get("early_stopping_patience"):
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.training.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

    if has_logger:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    return callbacks


def setup_loggers(config: DictConfig) -> list[Any]:
    loggers: list[Any] = []

    # In DDP, every process executes this module. Only initialize MLflow on rank 0.
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if local_rank != 0:
        return loggers

    if config.logging.use_mlflow:
        try:
            mlflow_logger = MLFlowLogger(
                experiment_name=config.logging.experiment_name,
                tracking_uri=config.logging.tracking_uri,
                run_name="baseline",
            )
            loggers.append(mlflow_logger)
            logger.info("MLflow logger initialized")
        except Exception as exc:
            logger.warning(f"Failed to initialize MLflow logger: {exc}")

    return loggers


def train_from_config(config: DictConfig) -> None:
    setup_logging(config.logging.level)

    _patch_pl_combined_loader_len()

    t0 = time.perf_counter()

    logger.info("=" * 80)
    logger.info("BASELINE VIDEO CAPTIONING - TRAINING")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(config)}")

    repo_root = Path(__file__).resolve().parents[2]

    output_dir = repo_root / str(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    pl.seed_everything(int(config.training.seed), workers=True)

    logger.info("\n" + "=" * 80)
    logger.info("SETTING UP DATA")
    logger.info("=" * 80)

    json_dir = config.dataset.paths.json_dir
    videos_dir = config.dataset.paths.videos_dir

    t_data_init = time.perf_counter()
    data_module = VideoDataModule(
        json_dir=json_dir,
        videos_dir=videos_dir,
        dataset_type="baseline",
        image_size=config.dataset.image.size,
        batch_size=config.dataset.dataloader.batch_size,
        num_workers=config.dataset.dataloader.num_workers,
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
        max_caption_length=config.dataset.dataloader.max_caption_length,
        tokenizer_vocab_size=config.dataset.tokenizer.vocab_size,
        tokenizer_model_dir=config.dataset.tokenizer.model_dir,
    )
    logger.info(f"DataModule init time: {time.perf_counter() - t_data_init:.3f}s")

    logger.info(f"Batch size: {config.dataset.dataloader.batch_size}")
    logger.info(f"Num workers: {config.dataset.dataloader.num_workers}")
    logger.info(f"Max caption length: {config.dataset.dataloader.max_caption_length}")

    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)

    t_model = time.perf_counter()
    model, vision_encoder, vocab_size = create_model_and_encoder(config)
    logger.info(f"Model+encoder creation time: {time.perf_counter() - t_model:.3f}s")

    logger.info("Model type: Baseline (ResNet-50 + LSTM)")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {config.model.embedding.d_model}")
    logger.info(f"LSTM layers: {config.model.lstm.num_layers}")
    logger.info(f"LSTM dropout: {config.model.lstm.dropout}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    logger.info("\n" + "=" * 80)
    logger.info("CREATING LIGHTNING MODULE")
    logger.info("=" * 80)

    t_setup = time.perf_counter()
    data_module.setup(stage="fit")
    logger.info(f"DataModule setup(fit) time: {time.perf_counter() - t_setup:.3f}s")
    tokenizer = data_module.tokenizer

    t_lm = time.perf_counter()

    # Backward-compat for older configs:
    # - generation.beam_size / generation.length_penalty
    # New structure:
    # - generation.beam_search_settings.beam_size / generation.beam_search_settings.length_penalty
    gen = getattr(config, "generation", {})
    beam_cfg = getattr(gen, "beam_search_settings", {})
    topk_cfg = getattr(gen, "topk_settings", {})

    beam_size = int(getattr(beam_cfg, "beam_size", getattr(gen, "beam_size", 5)))
    length_penalty = float(getattr(beam_cfg, "length_penalty", getattr(gen, "length_penalty", 1.2)))
    max_length = int(getattr(gen, "max_length", 100))
    temperature = float(getattr(gen, "temperature", 1.0))
    top_k = int(getattr(topk_cfg, "k", getattr(gen, "top_k", 50)))

    lightning_module = VideoCaptioningLightning(
        model=model,
        vision_encoder=vision_encoder,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,
        pad_token_id=int(getattr(tokenizer, "pad_id", 0)),
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
    logger.info(f"LightningModule init time: {time.perf_counter() - t_lm:.3f}s")

    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Optimizer: {config.training.optimizer}")
    logger.info(f"Weight decay: {config.training.weight_decay}")

    logger.info("\n" + "=" * 80)
    logger.info("SETTING UP CALLBACKS AND LOGGERS")
    logger.info("=" * 80)

    t_loggers = time.perf_counter()
    loggers = setup_loggers(config)
    logger.info(f"Logger setup time: {time.perf_counter() - t_loggers:.3f}s")
    has_logger = len(loggers) > 0
    callbacks = setup_callbacks(config, output_dir, has_logger=has_logger)

    for callback in callbacks:
        logger.info(f"Added callback: {callback.__class__.__name__}")

    for logger_obj in loggers:
        logger.info(f"Added logger: {logger_obj.__class__.__name__}")

    logger.info("\n" + "=" * 80)
    logger.info("CREATING TRAINER")
    logger.info("=" * 80)

    t_trainer = time.perf_counter()
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        strategy=config.training.strategy,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.logging.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        num_sanity_val_steps=int(getattr(config.training, "num_sanity_val_steps", 0)),
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    logger.info(f"Trainer init time: {time.perf_counter() - t_trainer:.3f}s")

    logger.info(f"Accelerator: {config.training.accelerator}")
    logger.info(f"Devices: {config.training.devices}")
    logger.info(f"Precision: {config.training.precision}")
    logger.info(f"Max epochs: {config.training.max_epochs}")

    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80 + "\n")

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
    experiment_name = str(getattr(config.logging, "experiment_name", "baseline"))
    try:
        for lg in loggers or []:
            if isinstance(lg, MLFlowLogger):
                run_id = str(getattr(lg, "run_id", "") or "")
                break

        trainer.fit(lightning_module, datamodule=data_module)
    finally:
        # Best-effort export. Safe to no-op if MLflow isn't enabled.
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
                    ],
                )
                exported_dir = repo_root / "plots" / experiment_name / run_id
                logger.info("Exported MLflow plots/logs to: %s", exported_dir)
            except Exception as exc:
                logger.warning(f"Failed to export MLflow plots/logs: {exc}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    final_model_path = output_dir / "baseline_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")


def run_train_baseline(*hydra_overrides: str) -> None:
    """Compose config via Hydra and run baseline training.

    Args:
        hydra_overrides: Hydra-style overrides, e.g. ("training.max_epochs=3",)
    """

    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    # Backward-compat: translate old top-level overrides (dataloader.*, tokenizer.*)
    translated: list[str] = []
    for o in hydra_overrides:
        if o.startswith("dataloader."):
            translated.append("dataset." + o)
        elif o.startswith("tokenizer."):
            translated.append("dataset." + o)
        else:
            translated.append(o)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="baseline", overrides=translated)

    # Resolve interpolations eagerly for clearer errors
    OmegaConf.resolve(config)

    train_from_config(config)
