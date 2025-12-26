"""Video captioning neural network models.

Models: nn.Module (no Lightning).
Vision encoder: Separate module.
Lightning integration: Done in train.py
"""

import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torchvision.models as models


class ProjectionLayer(nn.Module):
    """Project high-dimensional features to embedding dimension."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        if hidden_dim is None:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to embeddings."""
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class VisionEncoder(nn.Module):
    """Vision encoder for extracting frame embeddings.
    
    Supports ResNet-50 or ViT-tiny from timm based on config.
    Output dimensions are automatically determined by model architecture.
    """
    
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True, freeze: bool = True):
        super().__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.encoder = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            self.output_dim = 2048  # ResNet-50 avg pooled output dimension
        elif model_name == 'vit_tiny':
            # ViT-tiny from timm: timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k
            try:
                import timm
                self.encoder = timm.create_model(
                    'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0  # Remove classification head, get features only
                )
                # ViT-tiny patch16 has embedding dimension of 192
                self.output_dim = 192
            except ImportError:
                raise ValueError("ViT-tiny requires timm. Install: pip install timm")
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'resnet50' or 'vit_tiny'")
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from frames.
        
        Args:
            frames: (T, 3, 224, 224) or (B, T, 3, 224, 224)
        
        Returns:
            embeddings: (T, output_dim) or (B, T, output_dim)
        """
        if frames.dim() == 4:
            # (T, 3, 224, 224) - single sample
            T = frames.shape[0]
            
            if self.model_name == 'resnet50':
                embeddings = self.encoder(frames)  # (T, 2048, 1, 1)
                embeddings = embeddings.view(T, -1)
            else:
                embeddings = self.encoder(frames)  # (T, 384)
            
            return embeddings
        else:
            # (B, T, 3, 224, 224) - batch
            B, T = frames.shape[:2]
            frames_reshaped = frames.view(B*T, *frames.shape[2:])
            
            if self.model_name == 'resnet50':
                embeddings = self.encoder(frames_reshaped)
                embeddings = embeddings.view(B*T, -1)
            else:
                embeddings = self.encoder(frames_reshaped)
            
            embeddings = embeddings.view(B, T, -1)
            return embeddings


class BaselineVideoCaptioningModel(nn.Module):
    """Baseline LSTM-based video captioning model.
    
    At each timestep:
      input = [frame_embedding | text_embedding]
      hidden, cell = LSTM(input, (hidden, cell))
      output = Linear(hidden) → vocab
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        dropout: float,
        vision_encoder_output_dim: int,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embed text tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Project vision encoder output to d_model if needed
        if vision_encoder_output_dim != d_model:
            self.frame_projection = ProjectionLayer(vision_encoder_output_dim, d_model)
        else:
            self.frame_projection = nn.Identity()
        
        # LSTM decoder: input is [frame_emb | text_emb] = 2*d_model
        self.lstm = nn.LSTM(
            input_size=2 * d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        frame_embeddings: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            frame_embeddings: (batch_size, vision_output_dim) from encoder
            tokens: (batch_size, seq_len) - target sequence
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        
        # Embed text tokens
        text_embeddings = self.embedding(tokens)  # (batch_size, seq_len, d_model)
        
        # Project frame embeddings
        frame_emb = self.frame_projection(frame_embeddings)  # (batch_size, d_model)
        
        # Repeat frame embedding for each timestep
        frame_emb_expanded = frame_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Concatenate frame and text embeddings
        lstm_input = torch.cat([frame_emb_expanded, text_embeddings], dim=-1)  # (batch_size, seq_len, 2*d_model)
        lstm_input = self.dropout(lstm_input)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(lstm_input)  # (batch_size, seq_len, d_model)
        
        # Project to vocabulary
        logits = self.output_layer(lstm_output)  # (batch_size, seq_len, vocab_size)
        
        return logits

    def init_decode_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize (hidden, cell) state for autoregressive decoding."""
        state_dtype = dtype or next(self.parameters()).dtype
        h0 = torch.zeros(self.num_layers, batch_size, self.d_model, device=device, dtype=state_dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.d_model, device=device, dtype=state_dtype)
        return h0, c0

    def decode_step(
        self,
        frame_emb: torch.Tensor,
        token_ids: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step decode.

        Args:
            frame_emb: (B, d_model) frame embedding (already projected)
            token_ids: (B,) current token ids
            hidden: (num_layers, B, d_model)
            cell: (num_layers, B, d_model)

        Returns:
            logits: (B, vocab_size)
            next_hidden, next_cell
        """
        text_emb = self.embedding(token_ids).unsqueeze(1)  # (B, 1, d_model)
        frame_expanded = frame_emb.unsqueeze(1)  # (B, 1, d_model)
        lstm_input = torch.cat([frame_expanded, text_emb], dim=-1)  # (B, 1, 2*d_model)
        lstm_input = self.dropout(lstm_input)

        lstm_output, (next_hidden, next_cell) = self.lstm(lstm_input, (hidden, cell))
        logits = self.output_layer(lstm_output[:, 0, :])  # (B, vocab_size)
        return logits, next_hidden, next_cell


class TransformerVideoCaptioningModel(nn.Module):
    """Transformer-based video captioning model.
    
    Input sequence format:
      [<VIDEO_START>, frame_emb_1, ..., frame_emb_T, <VIDEO_END>, text_tokens]
    
    All embeddings are d_model dimension.
    Transformer decoder with causal masking for autoregressive generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        vision_encoder_output_dim: int,
        max_seq_len: int,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.special_tokens = special_tokens or {
            'video_start': 1,
            'video_end': 2,
            'pad': 0,
        }
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.special_tokens.get('pad', 0))
        
        # Project vision encoder output
        if vision_encoder_output_dim != d_model:
            self.frame_projection = ProjectionLayer(vision_encoder_output_dim, d_model)
        else:
            self.frame_projection = nn.Identity()
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal attention mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        frame_embeddings: torch.Tensor,
        tokens: torch.Tensor,
        frame_lengths: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            frame_embeddings: (batch_size, max_T, vision_output_dim)
            tokens: (batch_size, seq_len)
            frame_lengths: (batch_size,) - actual number of frames
            frame_mask: (batch_size, max_T) - 1 for real, 0 for padding
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size = frame_embeddings.shape[0]
        max_T = frame_embeddings.shape[1]
        text_seq_len = tokens.shape[1]

        # Project frame embeddings
        frame_emb = self.frame_projection(frame_embeddings)  # (B, max_T, d_model)

        # Embed text tokens
        text_emb = self.embedding(tokens)  # (B, L, d_model)
        text_emb = self.dropout(text_emb)

        device = frame_emb.device
        pad_id = int(self.special_tokens.get("pad", 0))

        # Build per-sample sequences so that the text segment starts immediately after
        # each sample's *true* number of frames. This avoids shifting text positions
        # depending on other samples in the batch (which hurts convergence).
        total_seq_len = 1 + int(max(frame_lengths).item()) + 1 + text_seq_len

        # Safety: positional encoding is constructed with a fixed maximum length.
        # If we exceed it, training/inference will be invalid (or crash). Prefer to
        # fail loudly with an actionable message.
        max_pe_len = int(self.positional_encoding.pe.shape[1])
        if total_seq_len > max_pe_len:
            raise ValueError(
                f"Transformer sequence length {total_seq_len} exceeds max_seq_len={max_pe_len}. "
                f"Reduce target_fps / max_caption_length or increase model.transformer.max_seq_len."
            )

        sequence = torch.zeros((batch_size, total_seq_len, self.d_model), device=device, dtype=frame_emb.dtype)
        key_padding_mask = torch.ones((batch_size, total_seq_len), device=device, dtype=torch.bool)

        video_start_id = int(self.special_tokens["video_start"])
        video_end_id = int(self.special_tokens["video_end"])
        video_start_emb = self.embedding(torch.tensor([video_start_id], device=device)).view(1, 1, -1)
        video_end_emb = self.embedding(torch.tensor([video_end_id], device=device)).view(1, 1, -1)

        frame_mask_bool = frame_mask.bool() if frame_mask.dtype != torch.bool else frame_mask
        text_pad = tokens.eq(pad_id)

        # Fill sequences
        for i in range(batch_size):
            Ti = int(frame_lengths[i].item())
            Ti = max(0, min(Ti, max_T))

            # <VIDEO_START>
            sequence[i, 0:1, :] = video_start_emb
            key_padding_mask[i, 0] = False

            # Frames (only real frames)
            if Ti > 0:
                sequence[i, 1 : 1 + Ti, :] = frame_emb[i, :Ti, :]
                # Mark padded frames as padding (ignored)
                key_padding_mask[i, 1 : 1 + Ti] = ~frame_mask_bool[i, :Ti]

            # <VIDEO_END>
            ve_pos = 1 + Ti
            sequence[i, ve_pos : ve_pos + 1, :] = video_end_emb
            key_padding_mask[i, ve_pos] = False

            # Text
            t_pos = ve_pos + 1
            sequence[i, t_pos : t_pos + text_seq_len, :] = text_emb[i, :, :]
            key_padding_mask[i, t_pos : t_pos + text_seq_len] = text_pad[i, :]

        # Apply positional encoding
        sequence = self.positional_encoding(sequence)

        causal_mask = self._build_causal_mask(total_seq_len, device)

        decoded = self.transformer_decoder(
            tgt=sequence,
            memory=sequence,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )

        # Extract text part per sample (variable offset) and project to vocab.
        logits = torch.zeros((batch_size, text_seq_len, self.vocab_size), device=device, dtype=decoded.dtype)
        for i in range(batch_size):
            Ti = int(frame_lengths[i].item())
            Ti = max(0, min(Ti, max_T))
            t_pos = 1 + Ti + 1
            text_decoded = decoded[i, t_pos : t_pos + text_seq_len, :]
            logits[i] = self.output_layer(text_decoded)

        return logits


def create_model(
    model_type: str,
    vocab_size: int,
    config: Dict[str, Any],
) -> nn.Module:
    """Create a video captioning model.
    
    Args:
        model_type: 'baseline' or 'advanced'
        vocab_size: Size of vocabulary
        config: Configuration dict with model hyperparameters
                Must contain keys matching model requirements
    
    Returns:
        Model instance (nn.Module)
    """
    
    # All parameters must be in config - no magic number defaults
    d_model = config['embedding']['d_model']
    vision_output_dim = config['vision_encoder']['output_dim']
    
    if model_type == 'baseline':
        return BaselineVideoCaptioningModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=config['lstm']['num_layers'],
            dropout=config['lstm']['dropout'],
            vision_encoder_output_dim=vision_output_dim,
        )
    
    elif model_type == 'advanced':
        return TransformerVideoCaptioningModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=config['transformer']['num_layers'],
            num_heads=config['transformer']['num_heads'],
            ffn_dim=config['transformer']['ffn_dim'],
            dropout=config['transformer']['dropout'],
            vision_encoder_output_dim=vision_output_dim,
            max_seq_len=config['transformer']['max_seq_len'],
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
