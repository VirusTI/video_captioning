"""Lightning module for the advanced (Transformer) captioning model.

Implements the "main model" training setup:
1) Extract per-frame embeddings with VisionEncoder (ViT-tiny) at target_fps.
2) Feed the Transformer as a single concatenated sequence:
   <VIDEO_START> [frame_tokens...] <VIDEO_END> [text tokens...]

Loss is computed only on the text-token logits (the model returns logits for the
text segment), with teacher forcing (BOS + tokens[:-1] -> predict tokens).
"""

from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from video_captioning.models.model import TransformerVideoCaptioningModel, VisionEncoder
from video_captioning.training.val_logging import ValMetricsLogger, log_learning_rate


class AdvancedVideoCaptioningLightning(pl.LightningModule):
    def __init__(
        self,
        model: TransformerVideoCaptioningModel,
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
        super().__init__()
        self.model = model
        self.vision_encoder = vision_encoder
        self.vocab_size = int(vocab_size)
        self.tokenizer = tokenizer
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer_name = str(optimizer).lower()
        self.pad_token_id = int(pad_token_id)

        # Generation parameters
        self.beam_size = int(beam_size)
        self.max_length = int(max_length)
        self.temperature = float(temperature)
        self.length_penalty = float(length_penalty)
        self.top_k = int(top_k)

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

    def _encode_frames_sequence(
        self, frames: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode frames into (B, T, D) embeddings, zeroing out padding."""
        # frames: (B, T, 3, H, W)
        B, T = frames.shape[:2]
        frames_flat = frames.view(B * T, *frames.shape[2:])
        emb_flat = self.vision_encoder(frames_flat)  # (B*T, D)
        emb = emb_flat.view(B, T, -1)

        mask_bool = frame_mask.bool() if frame_mask.dtype != torch.bool else frame_mask
        emb = emb.clone()
        emb[~mask_bool] = 0.0
        return emb

    def _special_ids(self) -> tuple[int, int, int]:
        special = {}
        if hasattr(self.tokenizer, "get_special_tokens"):
            try:
                special = self.tokenizer.get_special_tokens() or {}
            except Exception:
                special = {}
        bos_id = int(special.get("bos", getattr(self.tokenizer, "bos_id", 2)))
        eos_id = int(special.get("eos", getattr(self.tokenizer, "eos_id", 3)))
        pad_id = int(special.get("pad", getattr(self.tokenizer, "pad_id", 0)))
        return bos_id, eos_id, pad_id

    def _decode_token_ids(self, token_ids: list[int]) -> str:
        _, eos_id, pad_id = self._special_ids()
        trimmed: list[int] = []
        for tid in token_ids:
            if tid in (eos_id, pad_id):
                break
            trimmed.append(int(tid))
        try:
            return str(self.tokenizer.decode(trimmed))
        except Exception:
            return ""

    def _next_token_logits(
        self,
        frame_embeddings: torch.Tensor,
        frame_lengths: torch.Tensor,
        frame_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for the next token given current prefix tokens.

        tokens: (B, L) prefix tokens starting with BOS.
        Returns: (B, vocab_size) logits for next token.
        """
        logits = self.model(
            frame_embeddings=frame_embeddings,
            tokens=tokens,
            frame_lengths=frame_lengths,
            frame_mask=frame_mask,
        )
        return logits[:, -1, :]

    def generate_greedy(
        self,
        frame_embeddings: torch.Tensor,
        frame_lengths: torch.Tensor,
        frame_mask: torch.Tensor,
        max_length: int = 100,
    ) -> list[str]:
        bos_id, eos_id, pad_id = self._special_ids()
        device = frame_embeddings.device
        batch_size = int(frame_embeddings.shape[0])

        tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        generated: list[list[int]] = [[] for _ in range(batch_size)]
        for _ in range(max(1, int(max_length)) - 1):
            with torch.no_grad():
                step_logits = self._next_token_logits(
                    frame_embeddings, frame_lengths, frame_mask, tokens
                )
                # Never generate PAD during decoding; PAD is only for padding targets.
                step_logits[:, pad_id] = float("-inf")
                next_ids = torch.argmax(step_logits, dim=-1)

            # Once finished, keep emitting PAD (so sequences don't change).
            next_ids = torch.where(finished, torch.tensor(pad_id, device=device), next_ids)

            for i in range(batch_size):
                if finished[i]:
                    continue
                tid = int(next_ids[i].item())
                if tid == eos_id:
                    finished[i] = True
                else:
                    generated[i].append(tid)

            tokens = torch.cat([tokens, next_ids.view(batch_size, 1)], dim=1)
            if bool(finished.all().item()):
                break

        return [self._decode_token_ids(ids) for ids in generated]

    def generate_topk(
        self,
        frame_embeddings: torch.Tensor,
        frame_lengths: torch.Tensor,
        frame_mask: torch.Tensor,
        max_length: int = 100,
        top_k: int = 50,
        temperature: float = 1.0,
    ) -> list[str]:
        bos_id, eos_id, pad_id = self._special_ids()
        device = frame_embeddings.device
        batch_size = int(frame_embeddings.shape[0])

        tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(batch_size)]

        k = max(1, int(top_k))
        temp = float(temperature)
        if temp <= 0:
            temp = 1.0

        for _ in range(max(1, int(max_length)) - 1):
            with torch.no_grad():
                step_logits = self._next_token_logits(
                    frame_embeddings, frame_lengths, frame_mask, tokens
                )
                step_logits = step_logits / temp
                # Never sample PAD.
                step_logits[:, pad_id] = float("-inf")
                topk_vals, topk_idx = torch.topk(
                    step_logits, k=min(k, step_logits.shape[-1]), dim=-1
                )
                probs = torch.softmax(topk_vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_ids = topk_idx.gather(1, choice.view(batch_size, 1)).squeeze(-1)

            next_ids = torch.where(finished, torch.tensor(pad_id, device=device), next_ids)

            for i in range(batch_size):
                if finished[i]:
                    continue
                tid = int(next_ids[i].item())
                if tid == eos_id:
                    finished[i] = True
                else:
                    generated[i].append(tid)

            tokens = torch.cat([tokens, next_ids.view(batch_size, 1)], dim=1)
            if bool(finished.all().item()):
                break

        return [self._decode_token_ids(ids) for ids in generated]

    def generate(
        self,
        frame_embeddings: torch.Tensor,
        frame_lengths: torch.Tensor,
        frame_mask: torch.Tensor,
        beam_size: int = 5,
        max_length: int = 100,
        temperature: float = 1.0,
        length_penalty: float = 1.2,
    ) -> list[str]:
        """Beam search generation (per-sample).

        This is intentionally simple and used only for validation metrics.
        """
        bos_id, eos_id, pad_id = self._special_ids()
        device = frame_embeddings.device
        bsz = int(frame_embeddings.shape[0])

        out: list[str] = []
        beam_n = max(1, int(beam_size))
        temp = float(temperature)
        if temp <= 0:
            temp = 1.0

        for i in range(bsz):
            fe = frame_embeddings[i : i + 1]
            fl = frame_lengths[i : i + 1]
            fm = frame_mask[i : i + 1]

            beams = [
                {
                    "tokens": [bos_id],
                    "logprob": 0.0,
                    "finished": False,
                }
            ]

            for _ in range(max(1, int(max_length)) - 1):
                candidates: list[dict[str, Any]] = []
                for beam in beams:
                    last = int(beam["tokens"][-1])
                    if beam["finished"] or last == eos_id:
                        candidates.append(beam)
                        continue

                    tok = torch.tensor([beam["tokens"]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = self._next_token_logits(fe, fl, fm, tok) / temp
                        log_probs = torch.log_softmax(logits, dim=-1)
                        top_log_probs, top_ids = torch.topk(log_probs, k=beam_n, dim=-1)

                    for k in range(beam_n):
                        tid = int(top_ids[0, k].item())
                        lp = float(top_log_probs[0, k].item())
                        new_tokens = beam["tokens"] + [tid]
                        finished = tid == eos_id
                        candidates.append(
                            {
                                "tokens": new_tokens,
                                "logprob": float(beam["logprob"]) + lp,
                                "finished": finished,
                            }
                        )

                # length-normalize
                def score(b: dict[str, Any]) -> float:
                    length = max(1, len(b["tokens"]) - 1)
                    return float(b["logprob"]) / (length ** float(length_penalty))

                candidates.sort(key=score, reverse=True)
                beams = candidates[:beam_n]
                if all(bool(b["finished"]) for b in beams):
                    break

            best = beams[0]["tokens"][1:]
            out.append(self._decode_token_ids([int(t) for t in best]))

        return out

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        device = next(self.model.parameters()).device

        frames = batch["frames"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        frame_lengths = batch["frame_lengths"].to(device)
        tokens = batch["tokens"].to(device)
        batch_size = int(tokens.shape[0])

        bos_id = int(getattr(self.tokenizer, "bos_id", 2))
        input_tokens = tokens.clone()
        if input_tokens.shape[1] > 1:
            input_tokens[:, 1:] = tokens[:, :-1]
        input_tokens[:, 0] = bos_id

        frame_embeddings = self._encode_frames_sequence(frames, frame_mask)
        logits = self.model(
            frame_embeddings=frame_embeddings,
            tokens=input_tokens,
            frame_lengths=frame_lengths,
            frame_mask=frame_mask,
        )

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction="mean",
        )

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        device = next(self.model.parameters()).device

        frames = batch["frames"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        frame_lengths = batch["frame_lengths"].to(device)
        tokens = batch["tokens"].to(device)
        batch_size = int(tokens.shape[0])

        bos_id = int(getattr(self.tokenizer, "bos_id", 2))
        input_tokens = tokens.clone()
        if input_tokens.shape[1] > 1:
            input_tokens[:, 1:] = tokens[:, :-1]
        input_tokens[:, 0] = bos_id

        with torch.no_grad():
            frame_embeddings = self._encode_frames_sequence(frames, frame_mask)
            logits = self.model(
                frame_embeddings=frame_embeddings,
                tokens=input_tokens,
                frame_lengths=frame_lengths,
                frame_mask=frame_mask,
            )

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction="mean",
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        all_captions = batch.get("all_captions", [])
        take = self._val_logger.take_count(batch_size=batch_size, all_captions=all_captions)
        if take <= 0:
            return loss

        emb_subset = frame_embeddings[:take].detach()
        fl_subset = frame_lengths[:take].detach()
        fm_subset = frame_mask[:take].detach()
        refs_subset = list(all_captions)[:take]

        mode = self.val_generation_mode
        if mode == "greedy":
            preds = self.generate_greedy(
                emb_subset, fl_subset, fm_subset, max_length=self.max_length
            )
        elif mode in ("topk", "top-k", "top_k"):
            preds = self.generate_topk(
                emb_subset,
                fl_subset,
                fm_subset,
                max_length=self.max_length,
                top_k=self.top_k,
                temperature=self.temperature,
            )
        else:
            preds = self.generate(
                emb_subset,
                fl_subset,
                fm_subset,
                beam_size=self.beam_size,
                max_length=self.max_length,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
            )

        self._val_logger.add(preds=list(preds), refs=list(refs_subset))

        return loss

    def on_validation_epoch_end(self) -> None:
        self._val_logger.on_validation_epoch_end(self)

    def on_train_epoch_start(self) -> None:
        self._val_logger.on_train_epoch_start(self)

    def on_train_epoch_end(self) -> None:
        log_learning_rate(self)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            opt = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        return opt
