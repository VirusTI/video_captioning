"""Export trained models to ONNX.

This module intentionally focuses on exporting the *model forward* (vision encoder + captioning
head) rather than text generation (beam search), since decoding is typically handled outside the
ONNX graph.

Currently supported:
- baseline: (frames, tokens) -> logits

Notes:
- For baseline, `frames` is a float tensor of shape (B, 3, H, W) and `tokens` is int64
  of shape (B, L). The ONNX output is `logits` of shape (B, L, vocab_size).
- If you export from a `.ckpt`, weights for both the captioning model and the vision encoder
  are restored from the checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch import nn

from video_captioning.training.train_baseline import create_model_and_encoder
from video_captioning.training.train_advanced import _create_model_and_encoder
from video_captioning.data.dataset import BPETokenizer

logger = logging.getLogger(__name__)


class BaselineOnnxModule(nn.Module):
    def __init__(self, vision_encoder: nn.Module, caption_model: nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.caption_model = caption_model

    def forward(self, frames: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        frame_embeddings = self.vision_encoder(frames)
        return self.caption_model(frame_embeddings, tokens)


class AdvancedOnnxModule(nn.Module):
    """ONNX-friendly forward for the advanced Transformer model.

    This wrapper intentionally uses a *fixed* number of frames (T) for the exported
    graph. Frame validity is provided via `frame_mask`.

    Inputs:
      - frames: (B, T, 3, H, W)
      - tokens: (B, L)
      - frame_mask: (B, T) with 1 for real frames, 0 for padding

    Output:
      - logits: (B, L, vocab_size)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        transformer: nn.Module,
        *,
        pad_token_id: int,
        video_start_id: int,
        video_end_id: int,
        num_frames: int,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.transformer = transformer

        self.pad_token_id = int(pad_token_id)
        self.video_start_id = int(video_start_id)
        self.video_end_id = int(video_end_id)
        self.num_frames = int(num_frames)

    def forward(
        self,
        frames: torch.Tensor,
        tokens: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        # frames: (B, T, 3, H, W)
        # tokens: (B, L)
        # frame_mask: (B, T)
        B, T = frames.shape[:2]
        if T != self.num_frames:
            # Keep export graph stable: require the exported T.
            # (No exception in ONNX runtime, but in eager this is helpful.)
            raise ValueError(
                f"Expected frames with T={self.num_frames}, got T={int(T)}. Re-export with max_frames=T."
            )

        frame_embeddings = self.vision_encoder(frames)  # (B, T, vision_dim)

        # Mirror TransformerVideoCaptioningModel internals, but with fixed offsets.
        model = self.transformer
        frame_emb = model.frame_projection(frame_embeddings)  # (B, T, d_model)
        text_emb = model.embedding(tokens)  # (B, L, d_model)
        text_emb = model.dropout(text_emb)

        device = frame_emb.device

        # Special token embeddings
        vs = model.embedding(torch.tensor([self.video_start_id], device=device)).view(1, 1, -1)
        ve = model.embedding(torch.tensor([self.video_end_id], device=device)).view(1, 1, -1)
        vs = vs.expand(B, 1, -1)
        ve = ve.expand(B, 1, -1)

        # Concatenate: <VIDEO_START> [frames...] <VIDEO_END> [text...]
        sequence = torch.cat([vs, frame_emb, ve, text_emb], dim=1)

        # Key padding mask: True = padding (ignored)
        frame_pad = ~frame_mask.to(dtype=torch.bool)
        text_pad = tokens.eq(int(self.pad_token_id))
        key_padding_mask = torch.cat(
            [
                torch.zeros((B, 1), device=device, dtype=torch.bool),
                frame_pad,
                torch.zeros((B, 1), device=device, dtype=torch.bool),
                text_pad,
            ],
            dim=1,
        )

        sequence = model.positional_encoding(sequence)

        total_seq_len = sequence.shape[1]
        causal_mask = model._build_causal_mask(int(total_seq_len), device)

        decoded = model.transformer_decoder(
            tgt=sequence,
            memory=sequence,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )

        # Extract decoded text segment. Offset is constant: 1 + T + 1.
        t_pos = 1 + self.num_frames + 1
        text_decoded = decoded[:, t_pos:, :]
        logits = model.output_layer(text_decoded)
        return logits


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


def _load_config(config_name: str, *hydra_overrides: str) -> DictConfig:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=_translate_overrides(hydra_overrides))

    OmegaConf.resolve(config)
    return config


def _load_baseline_weights(
    model: nn.Module,
    vision_encoder: nn.Module,
    weights_path: Path,
) -> None:
    if weights_path.suffix == ".ckpt":
        ckpt = torch.load(str(weights_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        model_state: dict[str, torch.Tensor] = {}
        vision_state: dict[str, torch.Tensor] = {}
        other: dict[str, torch.Tensor] = {}

        for k, v in state_dict.items():
            if k.startswith("model."):
                model_state[k[len("model.") :]] = v
            elif k.startswith("vision_encoder."):
                vision_state[k[len("vision_encoder.") :]] = v
            else:
                other[k] = v

        if model_state:
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if unexpected:
                logger.warning("Unexpected keys loading model weights: %s", unexpected)
            if missing:
                logger.warning("Missing keys loading model weights: %s", missing)
        else:
            # Fallback: try direct load.
            model.load_state_dict(state_dict, strict=False)

        if vision_state:
            missing, unexpected = vision_encoder.load_state_dict(vision_state, strict=False)
            if unexpected:
                logger.warning("Unexpected keys loading vision weights: %s", unexpected)
            if missing:
                logger.warning("Missing keys loading vision weights: %s", missing)
        else:
            logger.info("No vision_encoder.* weights found in checkpoint; using default encoder weights")

        return

    # Plain state_dict (.pt) usually contains only the captioning model weights.
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)


def _load_advanced_weights(
    model: nn.Module,
    vision_encoder: nn.Module,
    weights_path: Path,
) -> None:
    if weights_path.suffix == ".ckpt":
        ckpt = torch.load(str(weights_path), map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        model_state: dict[str, torch.Tensor] = {}
        vision_state: dict[str, torch.Tensor] = {}

        for k, v in state_dict.items():
            if k.startswith("model."):
                model_state[k[len("model.") :]] = v
            elif k.startswith("vision_encoder."):
                vision_state[k[len("vision_encoder.") :]] = v

        if model_state:
            model.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        if vision_state:
            vision_encoder.load_state_dict(vision_state, strict=False)
        else:
            logger.info("No vision_encoder.* weights found in checkpoint; using default encoder weights")

        return

    # Plain state_dict (.pt) typically contains only the Transformer weights.
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)


def export_baseline_to_onnx(
    weights_path: str,
    output_path: str,
    *,
    opset: int = 17,
    batch_size: int = 1,
    seq_len: int = 16,
    config_name: str = "baseline",
    hydra_overrides: tuple[str, ...] = (),
) -> Path:
    """Export baseline model to ONNX.

    Args:
        weights_path: Path to a `.ckpt` (Lightning checkpoint) or `.pt` (state_dict) weights file.
        output_path: Where to write the `.onnx` file.
        opset: ONNX opset version.
        batch_size: Dummy batch size for tracing.
        seq_len: Dummy caption length for tracing.
        config_name: Hydra config name (default: baseline).
        hydra_overrides: Optional Hydra overrides (same style as training CLI).

    Returns:
        Path to exported ONNX file.
    """

    try:
        import onnx as _onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ONNX export requires the `onnx` package. "
            "Install extras with `uv sync --extra export` or `pip install -e '.[export]'`."
        ) from exc

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"weights_path not found: {weights_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_name, *hydra_overrides)
    model, vision_encoder, vocab_size = create_model_and_encoder(config)

    _load_baseline_weights(model, vision_encoder, weights)

    wrapper = BaselineOnnxModule(vision_encoder=vision_encoder, caption_model=model)
    wrapper.eval()

    image_size = int(config.dataset.image.size)
    dummy_frames = torch.randn(batch_size, 3, image_size, image_size, dtype=torch.float32)
    dummy_tokens = torch.randint(
        low=0,
        high=int(vocab_size),
        size=(batch_size, seq_len),
        dtype=torch.int64,
    )

    input_names = ["frames", "tokens"]
    output_names = ["logits"]
    dynamic_axes = {
        # LSTM export is most reliable with fixed batch size.
        # Keep sequence length dynamic for convenience.
        "tokens": {1: "seq_len"},
        "logits": {1: "seq_len"},
    }

    logger.info("Exporting baseline model to ONNX: %s", out)
    torch.onnx.export(
        wrapper,
        (dummy_frames, dummy_tokens),
        str(out),
        opset_version=int(opset),
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,
        do_constant_folding=True,
    )

    return out


def export_advanced_to_onnx(
    weights_path: str,
    output_path: str,
    *,
    opset: int = 17,
    batch_size: int = 1,
    num_frames: int = 16,
    seq_len: int = 16,
    config_name: str = "advanced",
    hydra_overrides: tuple[str, ...] = (),
) -> Path:
    """Export advanced Transformer model to ONNX.

    The exported graph is:
      (frames, tokens, frame_mask) -> logits

    where `frames` has a fixed T=`num_frames`.
    """

    try:
        import onnx as _onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ONNX export requires the `onnx` package. "
            "Install extras with `uv sync --extra export` or `pip install -e '.[export]'`."
        ) from exc

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(f"weights_path not found: {weights_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_name, *hydra_overrides)

    # Load tokenizer to get special token ids.
    tokenizer_model_path = Path(config.dataset.tokenizer.model_dir) / "bpe_model.model"
    tokenizer = BPETokenizer(vocab_size=int(config.dataset.tokenizer.vocab_size))
    if not tokenizer_model_path.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found at {tokenizer_model_path}. "
            "Run training/setup to create it, or point dataset.tokenizer.model_dir to an existing model."
        )
    tokenizer.load(str(tokenizer_model_path))
    special = tokenizer.get_special_tokens()

    model, vision_encoder = _create_model_and_encoder(config, special_tokens=special)
    _load_advanced_weights(model, vision_encoder, weights)

    wrapper = AdvancedOnnxModule(
        vision_encoder=vision_encoder,
        transformer=model,
        pad_token_id=int(special.get("pad", 0)),
        video_start_id=int(special.get("video_start", -1)),
        video_end_id=int(special.get("video_end", -1)),
        num_frames=int(num_frames),
    )
    wrapper.eval()

    image_size = int(config.dataset.image.size)

    dummy_frames = torch.randn(
        batch_size,
        int(num_frames),
        3,
        image_size,
        image_size,
        dtype=torch.float32,
    )
    dummy_tokens = torch.randint(
        low=0,
        high=int(config.dataset.tokenizer.vocab_size),
        size=(batch_size, int(seq_len)),
        dtype=torch.int64,
    )
    dummy_mask = torch.ones(batch_size, int(num_frames), dtype=torch.int64)

    input_names = ["frames", "tokens", "frame_mask"]
    output_names = ["logits"]

    # Keep shapes fixed for the exported graph (most reliable for transformer + mask logic).
    dynamic_axes = {}

    logger.info("Exporting advanced model to ONNX: %s", out)
    torch.onnx.export(
        wrapper,
        (dummy_frames, dummy_tokens, dummy_mask),
        str(out),
        opset_version=int(opset),
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,
        do_constant_folding=True,
    )

    return out
