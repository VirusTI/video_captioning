"""Inference entrypoints for the video captioning package.

This module is intended to be called via the single CLI entrypoint:

    python -m video_captioning.commands inference baseline /path/to/ckpt.ckpt /path/to/video.mp4
    python -m video_captioning.commands inference advanced /path/to/ckpt.ckpt /path/to/video.mp4

Generation/token selection mode is taken from config (by default from
`validation.generation_mode`, with an optional `inference.generation_mode` override
if present).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from video_captioning.models.model import TransformerVideoCaptioningModel, VisionEncoder
from video_captioning.training.lightning_module import VideoCaptioningLightning
from video_captioning.training.lightning_module_advanced import AdvancedVideoCaptioningLightning
from video_captioning.training.train_baseline import create_model_and_encoder, setup_logging

logger = logging.getLogger(__name__)


def _extract_first_frame(video_path: str, image_size: int) -> torch.Tensor:
    """Extract first frame from a video as a float tensor in [0, 1], shape (3, H, W)."""

    # Keep behavior aligned with dataset.py; avoid OpenCV threading/OpenCL quirks.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            frame = cv2.resize(frame, (image_size, image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            return torch.from_numpy(frame).permute(2, 0, 1)

        dummy = np.zeros((image_size, image_size, 3), dtype=np.float32)
        return torch.from_numpy(dummy).permute(2, 0, 1)

    except Exception as exc:
        logger.warning(f"Failed to extract frame from {video_path}: {exc}")
        dummy = np.zeros((image_size, image_size, 3), dtype=np.float32)
        return torch.from_numpy(dummy).permute(2, 0, 1)


def _decode_frames_uint8(video_path: str, image_size: int, target_fps: float) -> list[np.ndarray]:
    """Decode and resize frames as uint8 RGB HWC.

    Keeps behavior aligned with AdvancedVideoDataset: sequential decode and
    sampling based on true FPS when available.
    """

    # Keep behavior aligned with dataset.py; avoid OpenCV threading/OpenCL quirks.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 1.0:
        src_fps = 30.0

    tfps = float(target_fps)
    if tfps <= 0:
        tfps = 2.0

    frame_interval = max(1, int(round(src_fps / tfps)))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if idx % frame_interval == 0:
            frame = cv2.resize(frame, (image_size, image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        idx += 1

    cap.release()
    return frames


def _extract_frames_sequence(
    video_path: str,
    image_size: int,
    target_fps: float,
    normalization_mean: tuple[float, float, float],
    normalization_std: tuple[float, float, float],
) -> torch.Tensor:
    """Extract a (T, 3, H, W) float tensor normalized like AdvancedVideoDataset."""

    try:
        frames_list = _decode_frames_uint8(video_path, image_size=image_size, target_fps=target_fps)
        if not frames_list:
            frames_arr = np.zeros((1, image_size, image_size, 3), dtype=np.uint8)
        else:
            frames_arr = np.stack(frames_list, axis=0).astype(np.uint8, copy=False)

        frames_f = frames_arr.astype(np.float32) / 255.0
        mean = np.asarray(normalization_mean, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.asarray(normalization_std, dtype=np.float32).reshape(1, 1, 1, 3)
        frames_f = (frames_f - mean) / std

        frames_t = torch.from_numpy(frames_f).permute(0, 3, 1, 2).contiguous()
        return frames_t
    except Exception as exc:
        logger.warning(f"Failed to extract frames from {video_path}: {exc}")
        dummy = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
        frames_t = torch.from_numpy(dummy).permute(0, 3, 1, 2)
        return frames_t


def _create_advanced_model_and_encoder(
    config: DictConfig,
    special_tokens: dict[str, int],
    *,
    force_vision_encoder_pretrained: bool | None = None,
) -> tuple[TransformerVideoCaptioningModel, VisionEncoder]:
    pretrained_cfg = bool(getattr(config.model.vision_encoder, "pretrained", False))
    if force_vision_encoder_pretrained is not None:
        pretrained_cfg = bool(force_vision_encoder_pretrained)

    vision_encoder = VisionEncoder(
        model_name=str(config.model.vision_encoder.name),
        pretrained=pretrained_cfg,
        freeze=bool(getattr(config.model.vision_encoder, "freeze", True)),
    )

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


def _translate_overrides(hydra_overrides: tuple[str, ...]) -> list[str]:
    """Backward-compat for older overrides used in this repo."""

    translated: list[str] = []
    for o in hydra_overrides:
        if o.startswith("dataloader."):
            translated.append("dataset." + o)
        elif o.startswith("tokenizer."):
            translated.append("dataset." + o)
        else:
            translated.append(o)
    return translated


def _load_config(config_name: str, *hydra_overrides: str) -> DictConfig:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name=config_name,
            overrides=_translate_overrides(hydra_overrides),
        )

    OmegaConf.resolve(config)
    return config


def _get_generation_mode(config: DictConfig) -> str:
    # Prefer a dedicated inference section if user adds it later.
    inf = getattr(config, "inference", {})
    if isinstance(inf, dict) and "generation_mode" in inf:
        return str(inf["generation_mode"]).lower()

    val = getattr(config, "validation", {})
    if isinstance(val, dict) and "generation_mode" in val:
        return str(val["generation_mode"]).lower()

    return "beam"


def run_inference(
    model_type: str, weights_path: str, video_path: str, *hydra_overrides: str
) -> str:
    """Run caption generation for a single video.

    Args:
        model_type: Currently supports "baseline".
        weights_path: Path to a PyTorch Lightning checkpoint (.ckpt) or a plain state_dict (.pt).
        video_path: Path to the input video file.
        hydra_overrides: Optional Hydra overrides (same style as training CLI).

    Returns:
        Generated caption string.
    """

    model_type = str(model_type).lower()
    if model_type not in ("baseline", "advanced"):
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Supported: 'baseline', 'advanced'."
        )

    config = _load_config("baseline" if model_type == "baseline" else "advanced", *hydra_overrides)
    setup_logging(config.logging.level)

    ckpt_path = Path(weights_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"weights_path not found: {weights_path}")

    vid_path = Path(video_path)
    if not vid_path.exists():
        raise FileNotFoundError(f"video_path not found: {video_path}")

    # Load tokenizer model (SentencePiece) from configured path.
    # We only need tokenizer for decoding. Keep it lightweight.
    from video_captioning.data.dataset import BPETokenizer

    tokenizer_model_path = Path(config.dataset.tokenizer.model_dir) / "bpe_model.model"
    tokenizer = BPETokenizer(vocab_size=int(config.dataset.tokenizer.vocab_size))
    if tokenizer_model_path.exists():
        tokenizer.load(str(tokenizer_model_path))
    else:
        raise FileNotFoundError(
            f"Tokenizer model not found at {tokenizer_model_path}. "
            "Run training/setup to create it, "
            "or point dataset.tokenizer.model_dir to an existing model."
        )

    # Create modules matching training.
    if model_type == "baseline":
        model, vision_encoder, vocab_size = create_model_and_encoder(config)
    else:
        special_tokens = tokenizer.get_special_tokens()
        # Avoid any network/downloads during inference; checkpoint will overwrite weights.
        model, vision_encoder = _create_advanced_model_and_encoder(
            config,
            special_tokens=special_tokens,
            force_vision_encoder_pretrained=False,
        )
        vocab_size = int(config.dataset.tokenizer.vocab_size)

    # Generation config (re-uses the same schema as training/validation).
    gen = getattr(config, "generation", {})
    beam_cfg = getattr(gen, "beam_search_settings", {})
    topk_cfg = getattr(gen, "topk_settings", {})

    beam_size = int(getattr(beam_cfg, "beam_size", getattr(gen, "beam_size", 5)))
    length_penalty = float(getattr(beam_cfg, "length_penalty", getattr(gen, "length_penalty", 1.2)))
    max_length = int(getattr(gen, "max_length", 100))
    temperature = float(getattr(gen, "temperature", 1.0))
    top_k = int(getattr(topk_cfg, "k", getattr(gen, "top_k", 50)))

    # Create lightning module and load weights.
    if model_type == "baseline":
        if ckpt_path.suffix == ".ckpt":
            lightning_module: torch.nn.Module = VideoCaptioningLightning.load_from_checkpoint(
                str(ckpt_path),
                model=model,
                vision_encoder=vision_encoder,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                learning_rate=float(config.training.learning_rate),
                weight_decay=float(config.training.weight_decay),
                optimizer=str(config.training.optimizer),
                pad_token_id=int(getattr(tokenizer, "pad_id", 0)),
                beam_size=beam_size,
                max_length=max_length,
                temperature=temperature,
                length_penalty=length_penalty,
                top_k=top_k,
                # In inference we don't need val metrics buffers; keep defaults.
            )
        else:
            # Support plain state_dict files (saved by train_baseline as baseline_final.pt)
            state = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state)
            lightning_module = VideoCaptioningLightning(
                model=model,
                vision_encoder=vision_encoder,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                learning_rate=float(config.training.learning_rate),
                weight_decay=float(config.training.weight_decay),
                optimizer=str(config.training.optimizer),
                pad_token_id=int(getattr(tokenizer, "pad_id", 0)),
                beam_size=beam_size,
                max_length=max_length,
                temperature=temperature,
                length_penalty=length_penalty,
                top_k=top_k,
                val_generate_captions=False,
                val_compute_metrics=False,
            )
    else:
        if ckpt_path.suffix != ".ckpt":
            raise ValueError(
                "Advanced inference currently expects a PyTorch Lightning checkpoint (.ckpt)."
            )

        lightning_module = AdvancedVideoCaptioningLightning.load_from_checkpoint(
            str(ckpt_path),
            model=model,
            vision_encoder=vision_encoder,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
            learning_rate=float(config.training.learning_rate),
            weight_decay=float(config.training.weight_decay),
            optimizer=str(config.training.optimizer),
            pad_token_id=int(getattr(tokenizer, "pad_id", 0)),
            beam_size=beam_size,
            max_length=max_length,
            temperature=temperature,
            length_penalty=length_penalty,
            top_k=top_k,
            val_generate_captions=False,
            val_generation_mode=_get_generation_mode(config),
            val_compute_metrics=False,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.to(device)
    lightning_module.eval()

    mode = _get_generation_mode(config)

    if model_type == "baseline":
        frame = _extract_first_frame(str(vid_path), int(config.dataset.image.size))
        frames = frame.unsqueeze(0).to(device)  # (1, 3, H, W)

        with torch.no_grad():
            emb = lightning_module._encode_frames(frames)  # type: ignore[attr-defined]

            if mode == "greedy":
                captions = lightning_module.generate_greedy(  # type: ignore[attr-defined]
                    frame_embeddings=emb, max_length=max_length
                )
            elif mode in ("topk", "top-k", "top_k"):
                captions = lightning_module.generate_topk(  # type: ignore[attr-defined]
                    frame_embeddings=emb,
                    max_length=max_length,
                    top_k=top_k,
                    temperature=temperature,
                )
            else:
                captions = lightning_module.generate(  # type: ignore[attr-defined]
                    frame_embeddings=emb,
                    beam_size=beam_size,
                    max_length=max_length,
                    temperature=temperature,
                    length_penalty=length_penalty,
                )
    else:
        mean = tuple(float(x) for x in config.dataset.normalization.mean)
        std = tuple(float(x) for x in config.dataset.normalization.std)
        frames_seq = _extract_frames_sequence(
            str(vid_path),
            image_size=int(config.dataset.image.size),
            target_fps=float(config.dataset.advanced.target_fps),
            normalization_mean=mean,  # type: ignore[arg-type]
            normalization_std=std,  # type: ignore[arg-type]
        )
        frames_b = frames_seq.unsqueeze(0).to(device)  # (1, T, 3, H, W)
        T = int(frames_b.shape[1])
        frame_mask = torch.ones((1, T), dtype=torch.bool, device=device)
        frame_lengths = torch.tensor([T], dtype=torch.long, device=device)

        with torch.no_grad():
            emb = lightning_module._encode_frames_sequence(  # type: ignore[attr-defined]
                frames_b, frame_mask
            )

            if mode == "greedy":
                captions = lightning_module.generate_greedy(  # type: ignore[attr-defined]
                    frame_embeddings=emb,
                    frame_lengths=frame_lengths,
                    frame_mask=frame_mask,
                    max_length=max_length,
                )
            elif mode in ("topk", "top-k", "top_k"):
                captions = lightning_module.generate_topk(  # type: ignore[attr-defined]
                    frame_embeddings=emb,
                    frame_lengths=frame_lengths,
                    frame_mask=frame_mask,
                    max_length=max_length,
                    top_k=top_k,
                    temperature=temperature,
                )
            else:
                captions = lightning_module.generate(  # type: ignore[attr-defined]
                    frame_embeddings=emb,
                    frame_lengths=frame_lengths,
                    frame_mask=frame_mask,
                    beam_size=beam_size,
                    max_length=max_length,
                    temperature=temperature,
                    length_penalty=length_penalty,
                )

    caption = captions[0] if captions else ""
    return caption
