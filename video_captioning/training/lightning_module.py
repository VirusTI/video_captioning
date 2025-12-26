"""
PyTorch Lightning training module for video captioning models.

Integrates model, loss function, and evaluation metrics (BLEU-4, METEOR).
"""

from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam, AdamW

from video_captioning.models.model import VisionEncoder
from video_captioning.training.val_logging import ValMetricsLogger, log_learning_rate


class VideoCaptioningLightning(pl.LightningModule):
    """Lightning module for video captioning training and validation.

    Handles:
    - Forward pass through vision encoder + captioning model
    - Loss computation (cross-entropy)
    - BLEU-4 and METEOR metric calculation during validation
    - Logging to MLflow
    """

    def __init__(
        self,
        model: nn.Module,
        vision_encoder: VisionEncoder,
        vocab_size: int,
        tokenizer: Any,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer: str = "adamw",
        pad_token_id: int = 0,
        beam_size: int = 5,
        max_length: int = 100,
        temperature: float = 1.0,
        length_penalty: float = 1.2,
        top_k: int = 50,
        val_generate_captions: bool = True,
        val_generation_mode: str = "beam",
        val_max_samples: int = 256,
        val_compute_metrics: bool = True,
        val_compute_bleu_4: bool = True,
        val_compute_meteor: bool = False,
    ):
        """Initialize Lightning module.

        Args:
            model: Captioning model (BaselineVideoCaptioningModel or
                   TransformerVideoCaptioningModel)
            vision_encoder: VisionEncoder instance for frame embedding
            vocab_size: Vocabulary size
            tokenizer: BPETokenizer instance for decoding predictions
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization coefficient
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            pad_token_id: Padding token ID (for loss masking)
            beam_size: Beam search size for generation
            max_length: Maximum caption length
            temperature: Sampling temperature (1.0 = no change)
            length_penalty: Length penalty for beam search
        """
        super().__init__()

        self.model = model
        self.vision_encoder = vision_encoder
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer.lower()
        self.pad_token_id = pad_token_id

        # Generation parameters
        self.beam_size = beam_size
        self.max_length = max_length
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.top_k = int(top_k)

        # Validation-time generation/metrics controls
        self.val_generation_mode = str(val_generation_mode).lower()

        self._val_logger = ValMetricsLogger(
            tokenizer=tokenizer,
            val_generate_captions=bool(val_generate_captions),
            val_generation_mode=str(val_generation_mode).lower(),
            val_max_samples=int(val_max_samples),
            val_compute_metrics=bool(val_compute_metrics),
            val_compute_bleu_4=bool(val_compute_bleu_4),
            val_compute_meteor=bool(val_compute_meteor),
        )

        # For logging
        self.train_loss = 0.0
        self.val_loss = 0.0

    def forward(
        self,
        frames: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through vision encoder + captioning model.

        Args:
            frames: Frame tensor
                   (B, 3, H, W) for baseline or (B, T, 3, H, W) for advanced
            tokens: Target caption tokens (B, seq_len)

        Returns:
            Logits: (B, seq_len, vocab_size)
        """
        frame_embeddings = self._encode_frames(frames)
        logits = self.model(frame_embeddings, tokens)  # (B, seq_len, vocab_size)
        return logits

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode raw frames into a single embedding per sample."""
        if frames.dim() == 4:
            is_valid_frame = frames.abs().sum(dim=[1, 2, 3]) > 0  # (B,)
            frame_embeddings = self.vision_encoder(frames)  # (B, encoder_dim)
            frame_embeddings = frame_embeddings.clone()
            frame_embeddings[~is_valid_frame] = 0.0
            return frame_embeddings

        # Advanced: (B, T, 3, H, W)
        B, T = frames.shape[:2]
        is_valid = frames.abs().sum(dim=[2, 3, 4]) > 0  # (B, T)

        frames_flat = frames.view(B * T, *frames.shape[2:])
        embeddings_flat = self.vision_encoder(frames_flat)
        frame_embeddings_seq = embeddings_flat.view(B, T, -1)

        frame_embeddings_seq = frame_embeddings_seq.clone()
        frame_embeddings_seq[~is_valid] = 0.0

        valid_counts = is_valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return frame_embeddings_seq.sum(dim=1) / valid_counts

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step: compute loss and log metrics.

        Args:
            batch: Dictionary with 'frames', 'tokens'
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Ensure all tensor data is on the same device as the model
        device = next(self.model.parameters()).device
        frames = batch["frames"].to(device)
        tokens = batch["tokens"].to(device)
        batch_size = int(tokens.shape[0])

        # Teacher forcing: feed BOS + tokens[:-1] to predict tokens.
        # Without this shift, the model can learn the trivial identity mapping
        # (predict token t from token t), producing unrealistically low loss and
        # degenerate autoregressive generation.
        bos_id = int(getattr(self.tokenizer, "bos_id", 1))
        input_tokens = tokens.clone()
        if input_tokens.shape[1] > 1:
            input_tokens[:, 1:] = tokens[:, :-1]
        input_tokens[:, 0] = bos_id

        # Forward pass
        logits = self(frames, input_tokens)  # (B, seq_len, vocab_size)

        # Compute cross-entropy loss (ignoring pad tokens)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            tokens.view(-1),
            ignore_index=self.pad_token_id,
            reduction="mean",
        )

        # Log training loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.train_loss = loss.item()
        return loss

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Validation step: compute loss and metrics with beam search generation.

        Args:
            batch: Dictionary with 'frames', 'tokens', 'texts', 'all_captions'
            batch_idx: Batch index

        Returns:
            Dictionary with 'loss', 'bleu_4', 'meteor'
        """
        # Ensure all tensor data is on the same device as the model
        device = next(self.model.parameters()).device
        frames = batch["frames"].to(device)
        tokens = batch["tokens"].to(device)
        batch_size = int(tokens.shape[0])

        bos_id = int(getattr(self.tokenizer, "bos_id", 1))
        input_tokens = tokens.clone()
        if input_tokens.shape[1] > 1:
            input_tokens[:, 1:] = tokens[:, :-1]
        input_tokens[:, 0] = bos_id

        # Forward pass for loss (teacher-forcing)
        with torch.no_grad():
            frame_embeddings = self._encode_frames(frames)
            logits = self.model(frame_embeddings, input_tokens)  # (B, seq_len, vocab_size)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            tokens.view(-1),
            ignore_index=self.pad_token_id,
            reduction="mean",
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.val_loss = loss.item()

        all_captions = batch.get("all_captions", [])
        take = self._val_logger.take_count(batch_size=batch_size, all_captions=all_captions)
        if take <= 0:
            return {"loss": loss}

        emb_subset = frame_embeddings[:take].detach()
        refs_subset = list(all_captions)[:take]

        if self.val_generation_mode == "greedy":
            preds = self.generate_greedy(
                frame_embeddings=emb_subset,
                max_length=self.max_length,
            )
        elif self.val_generation_mode in ("topk", "top-k", "top_k"):
            preds = self.generate_topk(
                frame_embeddings=emb_subset,
                max_length=self.max_length,
                top_k=self.top_k,
                temperature=self.temperature,
            )
        else:
            preds = self.generate(
                frame_embeddings=emb_subset,
                beam_size=self.beam_size,
                max_length=self.max_length,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
            )

        self._val_logger.add(preds=list(preds), refs=list(refs_subset))

        return {"loss": loss}

    def generate_greedy(
        self,
        frame_embeddings: torch.Tensor,
        max_length: int = 100,
    ) -> List[str]:
        """Fast greedy decoding (vectorized across batch)."""
        batch_size = frame_embeddings.shape[0]
        device = frame_embeddings.device

        # Project frame embeddings
        frame_emb = self.model.frame_projection(frame_embeddings)  # (B, d_model)

        special = {}
        if hasattr(self.tokenizer, "get_special_tokens"):
            try:
                special = self.tokenizer.get_special_tokens() or {}
            except Exception:
                special = {}

        bos_id = int(special.get("bos", getattr(self.tokenizer, "bos_id", 1)))
        eos_id = int(special.get("eos", getattr(self.tokenizer, "eos_id", 2)))
        pad_id = int(special.get("pad", getattr(self.tokenizer, "pad_id", 0)))

        hidden, cell = self.model.init_decode_state(batch_size=batch_size, device=device)
        token_ids = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        generated: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_length - 1):
            with torch.no_grad():
                logits, hidden, cell = self.model.decode_step(
                    frame_emb=frame_emb,
                    token_ids=token_ids,
                    hidden=hidden,
                    cell=cell,
                )
            next_tokens = torch.argmax(logits, dim=-1)

            # Once finished, keep emitting PAD to avoid changing sequences
            next_tokens = torch.where(finished, torch.tensor(pad_id, device=device), next_tokens)

            for i in range(batch_size):
                t = int(next_tokens[i].item())
                if not finished[i]:
                    if t == eos_id:
                        finished[i] = True
                    elif t != pad_id:
                        generated[i].append(t)

            token_ids = next_tokens
            if bool(finished.all()):
                break

        return [self.tokenizer.decode(seq) for seq in generated]

    def generate_topk(
        self,
        frame_embeddings: torch.Tensor,
        max_length: int = 100,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> List[str]:
        """Top-k sampling (vectorized across batch)."""
        batch_size = frame_embeddings.shape[0]
        device = frame_embeddings.device

        k = max(1, int(top_k))

        frame_emb = self.model.frame_projection(frame_embeddings)  # (B, d_model)

        special = {}
        if hasattr(self.tokenizer, "get_special_tokens"):
            try:
                special = self.tokenizer.get_special_tokens() or {}
            except Exception:
                special = {}

        bos_id = int(special.get("bos", getattr(self.tokenizer, "bos_id", 1)))
        eos_id = int(special.get("eos", getattr(self.tokenizer, "eos_id", 2)))
        pad_id = int(special.get("pad", getattr(self.tokenizer, "pad_id", 0)))

        hidden, cell = self.model.init_decode_state(batch_size=batch_size, device=device)
        token_ids = torch.full((batch_size,), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        generated: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_length - 1):
            with torch.no_grad():
                logits, hidden, cell = self.model.decode_step(
                    frame_emb=frame_emb,
                    token_ids=token_ids,
                    hidden=hidden,
                    cell=cell,
                )

            if temperature != 1.0:
                logits = logits / float(temperature)

            # Choose from top-k per row
            topk_logits, topk_ids = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
            topk_probs = torch.softmax(topk_logits, dim=-1)
            sampled_idx = torch.multinomial(topk_probs, num_samples=1).squeeze(1)
            next_tokens = topk_ids.gather(1, sampled_idx.unsqueeze(1)).squeeze(1)

            next_tokens = torch.where(finished, torch.tensor(pad_id, device=device), next_tokens)

            for i in range(batch_size):
                t = int(next_tokens[i].item())
                if not finished[i]:
                    if t == eos_id:
                        finished[i] = True
                    elif t != pad_id:
                        generated[i].append(t)

            token_ids = next_tokens
            if bool(finished.all()):
                break

        return [self.tokenizer.decode(seq) for seq in generated]

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Test step: same as validation.

        Args:
            batch: Dictionary with batch data
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with 'optimizer' and optional 'lr_scheduler'
        """
        # Parameters to optimize (unfreeze vision encoder parameters if needed)
        params = [
            {"params": self.model.parameters(), "lr": self.learning_rate},
        ]

        # If vision encoder has trainable params, include them
        vision_params = [p for p in self.vision_encoder.parameters() if p.requires_grad]
        if vision_params:
            params.append({"params": vision_params, "lr": self.learning_rate})

        # Choose optimizer
        if self.optimizer_name == "adam":
            optimizer = Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            optimizer = SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Optional: learning rate scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6,
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _decode_predictions(self, logits: torch.Tensor) -> List[str]:
        """Decode logits to caption strings.

        Args:
            logits: Model output (B, seq_len, vocab_size)

        Returns:
            List of decoded caption strings
        """
        # Get argmax predictions
        predictions = torch.argmax(logits, dim=-1)  # (B, seq_len)

        # Move to CPU for tokenizer decoding
        predictions = predictions.cpu()

        # Special tokens
        special = {}
        if hasattr(self.tokenizer, "get_special_tokens"):
            try:
                special = self.tokenizer.get_special_tokens() or {}
            except Exception:
                special = {}
        eos_id = int(special.get("eos", getattr(self.tokenizer, "eos_id", 2)))
        pad_id = int(special.get("pad", getattr(self.tokenizer, "pad_id", 0)))

        # Decode using tokenizer
        captions = []
        for pred_tokens in predictions:
            token_list = pred_tokens.tolist()
            if eos_id in token_list:
                token_list = token_list[: token_list.index(eos_id)]
            token_list = [t for t in token_list if t != pad_id]
            caption = self.tokenizer.decode(token_list)
            captions.append(caption)

        return captions

    def generate(
        self,
        frame_embeddings: torch.Tensor,
        beam_size: int = 5,
        max_length: int = 100,
        temperature: float = 1.0,
        length_penalty: float = 1.2,
    ) -> List[str]:
        """Generate captions using beam search (autoregressive generation).

        This method can be used for validation, testing, and inference.

        Args:
            frame_embeddings: Frame embeddings (B, encoder_dim)
            beam_size: Beam search size
            max_length: Maximum caption length
            temperature: Sampling temperature (1.0 = no change)
            length_penalty: Length penalty for beam search

        Returns:
            List of generated caption strings
        """
        batch_size = frame_embeddings.shape[0]
        device = frame_embeddings.device

        # Project frame embeddings
        frame_emb = self.model.frame_projection(frame_embeddings)  # (B, d_model)

        # Start / end tokens from tokenizer
        special = {}
        if hasattr(self.tokenizer, "get_special_tokens"):
            try:
                special = self.tokenizer.get_special_tokens() or {}
            except Exception:
                special = {}

        bos_id = int(special.get("bos", getattr(self.tokenizer, "bos_id", 1)))
        eos_id = int(special.get("eos", getattr(self.tokenizer, "eos_id", 2)))
        pad_id = int(special.get("pad", getattr(self.tokenizer, "pad_id", 0)))

        # Beam search per sample in batch
        captions = []

        for b in range(batch_size):
            # init recurrent state for this sample
            h0, c0 = self.model.init_decode_state(batch_size=1, device=device)

            # Initialize beam for this sample
            beams = [
                {
                    "logprob": 0.0,
                    "normalized_logprob": 0.0,
                    "tokens": [bos_id],
                    "hidden": h0,
                    "cell": c0,
                }
            ]

            for step in range(max_length - 1):
                # Expand candidates from current beams
                candidates = []

                for beam_idx, beam in enumerate(beams):
                    last_token = beam["tokens"][-1]

                    # If finished, keep beam as-is
                    if last_token in (eos_id, pad_id):
                        # Ensure normalized_logprob exists for sorting
                        norm = beam.get("normalized_logprob")
                        if norm is None:
                            length = max(1, len(beam["tokens"]))
                            norm = beam["logprob"] / (length**length_penalty)
                        candidates.append({**beam, "normalized_logprob": float(norm)})
                        continue

                    token_ids = torch.tensor([last_token], dtype=torch.long, device=device)

                    with torch.no_grad():
                        logits, hidden, cell = self.model.decode_step(
                            frame_emb=frame_emb[b : b + 1],
                            token_ids=token_ids,
                            hidden=beam["hidden"],
                            cell=beam["cell"],
                        )

                    # Apply temperature
                    if temperature != 1.0:
                        logits = logits / temperature

                    # Get log probabilities
                    log_probs = F.log_softmax(logits, dim=-1)  # (1, vocab_size)

                    # Get top beam_size candidates
                    top_log_probs, top_tokens = torch.topk(log_probs, beam_size, dim=-1)

                    for k in range(beam_size):
                        candidate_logprob = beam["logprob"] + top_log_probs[0, k].item()

                        # Length penalty
                        length_factor = ((5.0 + (step + 1)) / 6.0) ** length_penalty
                        normalized_logprob = candidate_logprob / length_factor

                        candidates.append(
                            {
                                "logprob": candidate_logprob,
                                "normalized_logprob": normalized_logprob,
                                "tokens": beam["tokens"] + [top_tokens[0, k].item()],
                                "hidden": hidden.detach(),
                                "cell": cell.detach(),
                            }
                        )

                # Select top beam_size candidates
                candidates.sort(key=lambda x: x["normalized_logprob"], reverse=True)
                beams = candidates[:beam_size]

                # Stop if all beams generated EOS/PAD
                if all(beam["tokens"][-1] in (eos_id, pad_id) for beam in beams):
                    break

            # Select best beam
            best_beam = beams[0]
            caption_tokens = best_beam["tokens"][1:]  # Remove start token
            if eos_id in caption_tokens:
                caption_tokens = caption_tokens[: caption_tokens.index(eos_id)]

            # Decode caption
            caption = self.tokenizer.decode(caption_tokens)
            captions.append(caption)

        return captions

    def on_train_epoch_end(self) -> None:
        """Hook called at the end of training epoch."""
        log_learning_rate(self)

    def on_train_epoch_start(self) -> None:
        """Hook called at the start of training epoch.

        Useful to detect stalls between val end and next epoch start.
        """
        self._val_logger.on_train_epoch_start(self)

    def on_validation_epoch_end(self) -> None:
        """Hook called at the end of validation epoch."""
        self._val_logger.on_validation_epoch_end(self)
