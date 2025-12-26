"""Data loading for video captioning - returns RAW FRAMES (no embeddings).

Vision encoding happens in the model/training loop via Lightning.
This allows proper parallelization across GPU workers.
"""

import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import logging
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import sentencepiece as spm


logger = logging.getLogger(__name__)


def _configure_runtime_for_data_loading() -> None:
    """Configure runtime knobs for stable data loading.

    The assignment rubric forbids executable code at file import level.
    Keep all side-effects behind an explicit call.
    """

    # Albumentations performs an update/version check on import which can trigger
    # network calls. When DataLoader uses multiprocessing (especially with
    # "spawn"), those imports happen inside each worker and can cause long stalls.
    #
    # Albumentations (v2+) uses NO_ALBUMENTATIONS_UPDATE.
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")  # legacy/compat

    # OpenCV can deadlock or stall when used from multiple forked DataLoader workers
    # (especially with internal threading/OpenCL enabled). Make it deterministic and
    # more stable by disabling OpenCV threading and OpenCL.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Initialize DataLoader worker process.

    Must be defined at module scope to be picklable under multiprocessing 'spawn'
    (used by PyTorch DataLoader in some configs / DDP setups).

    We also clamp common BLAS/OpenMP thread pools to avoid CPU oversubscription.
    """

    _configure_runtime_for_data_loading()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


class BPETokenizer:
    """Byte-Pair Encoding tokenizer using SentencePiece."""

    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 8000):
        """Initialize BPE tokenizer.
        
        Args:
            model_path: Path to existing SentencePiece model
            vocab_size: Vocabulary size for training new model
        """
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = None

        # Default ids used when training a new SentencePiece model below.
        # If loading an existing model, these values are overwritten from the model.
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

    def train(self, captions: List[str], output_dir: str = "models"):
        """Train SentencePiece BPE model on captions."""
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        model_prefix = str(Path(output_dir) / "bpe_model")
        
        temp_file = f"{model_prefix}_temp.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(captions))
        
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            # Ensure we have an explicit PAD token and stable special ids.
            # This is important for padding + ignore_index in loss.
            pad_id=int(self.pad_id),
            unk_id=int(self.unk_id),
            bos_id=int(self.bos_id),
            eos_id=int(self.eos_id),
            pad_piece='<pad>',
            user_defined_symbols=['<VIDEO_START>', '<VIDEO_END>'],
        )
        
        self.model_path = f"{model_prefix}.model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.model_path)

        # Refresh ids from the trained model to be safe.
        self.pad_id = int(self.sp.pad_id())
        self.unk_id = int(self.sp.unk_id())
        self.bos_id = int(self.sp.bos_id())
        self.eos_id = int(self.sp.eos_id())
        
        Path(temp_file).unlink()
        return self.model_path

    def load(self, model_path: str):
        """Load pre-trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path

        # Read ids from model. Note: some older models may have pad_id == -1
        # if PAD wasn't configured. In that case, keep pad_id=0 as fallback.
        pad_id = int(self.sp.pad_id())
        if pad_id >= 0:
            self.pad_id = pad_id
        self.unk_id = int(self.sp.unk_id())
        self.bos_id = int(self.sp.bos_id())
        self.eos_id = int(self.sp.eos_id())

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded/trained")
        return self.sp.EncodeAsIds(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not loaded/trained")
        return self.sp.DecodeIds(token_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp is None:
            return self.vocab_size
        return self.sp.GetPieceSize()

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad': self.pad_id,
            'bos': self.bos_id,
            'eos': self.eos_id,
            'unk': self.unk_id,
            'video_start': self.sp.PieceToId('<VIDEO_START>') if self.sp else -1,
            'video_end': self.sp.PieceToId('<VIDEO_END>') if self.sp else -1,
        }


class BaselineVideoDataset(Dataset):
    """Baseline: Single raw frame per video.
    
    Returns: (frame, tokens, text, all_captions)
      - frame: (3, image_size, image_size) - raw RGB frame tensor
      - tokens: (max_length,) - BPE tokenized caption
      - text: str - caption text
      - all_captions: List[str] - all captions for evaluation
    """

    def __init__(
        self,
        json_path: str,
        videos_dir: str,
        tokenizer: BPETokenizer,
        image_size: int = 224,
        max_caption_length: int = 50,
        stage: str = 'train',
        cache_frames: bool = False,
        cache_max_items: int = 0,
        cache_policy: str = "lru",
    ):
        self.json_path = Path(json_path)
        self.videos_dir = Path(videos_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.stage = stage
        self.cache_frames = bool(cache_frames)
        self.cache_max_items = int(cache_max_items)
        self.cache_policy = str(cache_policy).lower()

        # LRU cache for decoded/resized frame tensors.
        # Important: with DataLoader persistent_workers=True, this cache persists
        # across epochs inside each worker process.
        self._frame_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        
        with open(self.json_path) as f:
            loaded = json.load(f)
            # Support both list and dict formats
            # If dict with 'video_files' key, use that
            if isinstance(loaded, dict) and 'video_files' in loaded:
                self.data = loaded['video_files']
            else:
                # Otherwise assume it's already a list
                self.data = loaded if isinstance(loaded, list) else [loaded]

    def __len__(self) -> int:
        return len(self.data)

    def _extract_frame(self, video_path: str) -> torch.Tensor:
        """Extract first frame from video (normalized to [0, 1])."""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                return torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)
            
            dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            return torch.from_numpy(dummy).permute(2, 0, 1)
        
        except Exception as e:
            print(f"Error extracting frame from {video_path}: {e}")
            dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            return torch.from_numpy(dummy).permute(2, 0, 1)

    def _get_cached_frame(self, key: str, video_path: str) -> torch.Tensor:
        if not self.cache_frames or self.cache_max_items <= 0:
            return self._extract_frame(video_path)

        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached

        frame = self._extract_frame(video_path)
        self._frame_cache[key] = frame
        self._frame_cache.move_to_end(key)
        if self.cache_policy != "all":
            if len(self._frame_cache) > self.cache_max_items:
                self._frame_cache.popitem(last=False)
        return frame

    def _encode_caption(self, caption: str) -> torch.Tensor:
        """Encode caption to BPE tokens."""
        max_len = int(self.max_caption_length)
        eos_id = int(getattr(self.tokenizer, "eos_id", 3))
        pad_id = int(getattr(self.tokenizer, "pad_id", 0))

        token_ids = self.tokenizer.encode(caption)
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.long)

        # Reserve 1 slot for EOS so the model learns to stop.
        token_ids = token_ids[: max(0, max_len - 1)]
        token_ids = token_ids + [eos_id]

        if len(token_ids) < max_len:
            token_ids = token_ids + [pad_id] * (max_len - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)

    def _get_captions(self, all_captions: List[str]) -> Tuple[str, List[str]]:
        """Get caption for this dataset."""
        if self.stage == 'train':
            return random.choice(all_captions), all_captions
        else:
            return all_captions[0], all_captions

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, List[str]]:
        item = self.data[idx]
        video_id = item['video_id']
        # Support both 'caption' and 'captions' keys
        captions = item.get('captions', item.get('caption', []))
        
        caption, all_captions = self._get_captions(captions)
        
        video_path = self.videos_dir / f"{video_id}.mp4"
        frame = self._get_cached_frame(str(video_id), str(video_path))  # (3, 224, 224)
        tokens = self._encode_caption(caption)
        
        return frame, tokens, caption, all_captions


class AdvancedVideoDataset(Dataset):
    """Advanced: Multiple frames (FPS-based) with augmentations.
    
    Returns: (frames, tokens, text, all_captions, num_frames)
      - frames: (T, 3, image_size, image_size) - variable number of raw frames
      - tokens: (max_length,) - BPE tokens
      - text: str - caption text
      - all_captions: List[str] - all captions
      - num_frames: int - actual number of frames T
    """

    def __init__(
        self,
        json_path: str,
        videos_dir: str,
        tokenizer: BPETokenizer,
        target_fps: float = 2.0,
        image_size: int = 224,
        max_caption_length: int = 50,
        normalization_mean: tuple = (0.485, 0.456, 0.406),
        normalization_std: tuple = (0.229, 0.224, 0.225),
        augment: bool = True,
        stage: str = 'train',
        cache_frames: bool = False,
        cache_max_items: int = 0,
        cache_policy: str = "lru",
        disk_cache_dir: Optional[str] = None,
        disk_cache_mmap: bool = True,
    ):
        self.json_path = Path(json_path)
        self.videos_dir = Path(videos_dir)
        self.tokenizer = tokenizer
        self.target_fps = target_fps
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.stage = stage
        self.cache_frames = bool(cache_frames)
        self.cache_max_items = int(cache_max_items)
        self.cache_policy = str(cache_policy).lower()

        self.disk_cache_dir = Path(disk_cache_dir).expanduser() if disk_cache_dir else None
        self.disk_cache_mmap = bool(disk_cache_mmap)
        if self.disk_cache_dir is not None:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # In-worker RAM cache for decoded frames (pre-augmentation).
        # NOTE: In DDP, this cache is duplicated across ranks/workers and can
        # easily explode memory. When disk_cache_dir is set, we avoid storing
        # decoded frames in RAM by default and rely on the disk cache instead.
        self._frames_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        
        with open(self.json_path) as f:
            loaded = json.load(f)
            # Support both list and dict formats
            # If dict with 'video_files' key, use that
            if isinstance(loaded, dict) and 'video_files' in loaded:
                self.data = loaded['video_files']
            else:
                # Otherwise assume it's already a list
                self.data = loaded if isinstance(loaded, list) else [loaded]
        
        # Augmentation pipeline (lazy-import albumentations to avoid heavy imports
        # in DataLoader worker spawn when augmentations aren't used).
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if augment and stage == 'train':
            transforms = [
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.GaussianBlur(blur_limit=5, p=0.2),
                A.RandomBrightnessContrast(p=0.2),
            ]
        else:
            transforms = []

        transforms.extend(
            [
                A.Normalize(mean=normalization_mean, std=normalization_std),
                ToTensorV2(),
            ]
        )

        # Use ReplayCompose so we can apply the *same* random transform to all
        # frames, without hard-coding a max number of frames via additional_targets.
        self.augmentation = A.ReplayCompose(transforms)

    def __len__(self) -> int:
        return len(self.data)

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """Sample frame indices based on target FPS.

        Note: This helper is currently unused by the sequential decoder.
        """
        # If you need this helper in the future, prefer the true FPS read from the file.
        # Keep a conservative fallback.
        fallback_src_fps = 30.0
        frame_interval = max(1, int(round(float(fallback_src_fps) / float(self.target_fps))))
        return list(range(0, total_frames, frame_interval))

    def _decode_frames_uint8(self, video_path: str) -> list[np.ndarray]:
        """Decode and resize frames as uint8 RGB HWC (no augmentation).

        Important for performance: avoid random seeking via CAP_PROP_POS_FRAMES,
        which is very expensive with many short seeks per sample.
        Instead, read sequentially and keep frames at the desired interval.
        """
        cap = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []

        # Read sequentially; sample every `frame_interval` frames.
        # Prefer the true FPS from the video container; fall back to config.
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 1.0:
            # If container metadata is missing/unreliable, fall back to a sane default.
            src_fps = 30.0
        frame_interval = max(1, int(round(src_fps / float(self.target_fps))))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if idx % frame_interval == 0:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            idx += 1

        cap.release()
        return frames

    def _cache_key_for_video(self, video_id: str) -> str:
        # Cache contents depend on sampling and resize params; include them.
        # We use the video_id (unique) plus parameters. We do NOT include
        # assumed video_fps here because decoding prefers the true FPS from the file.
        tfps = str(self.target_fps).replace(".", "p")
        return f"{video_id}__tfps{tfps}__sz{self.image_size}"

    def _disk_cache_path(self, cache_key: str) -> Optional[Path]:
        if self.disk_cache_dir is None:
            return None
        return self.disk_cache_dir / f"{cache_key}.npy"

    def _load_frames_from_disk_cache(self, cache_key: str) -> Optional[np.ndarray]:
        path = self._disk_cache_path(cache_key)
        if path is None or not path.exists():
            return None

        mmap_mode = "r" if self.disk_cache_mmap else None
        try:
            arr = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            # Expect (T, H, W, 3) uint8
            if not isinstance(arr, np.ndarray) or arr.ndim != 4 or arr.shape[-1] != 3:
                return None
            if arr.dtype != np.uint8:
                # Be tolerant; convert if needed.
                arr = arr.astype(np.uint8, copy=False)
            return arr
        except Exception:
            return None

    def _save_frames_to_disk_cache(self, cache_key: str, frames_arr: np.ndarray) -> None:
        path = self._disk_cache_path(cache_key)
        if path is None:
            return
        if path.exists():
            return

        try:
            if not isinstance(frames_arr, np.ndarray) or frames_arr.size == 0:
                return

            arr = frames_arr.astype(np.uint8, copy=False)
            tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
            # Atomic-ish write: write to tmp then rename.
            # IMPORTANT: pass a file handle so numpy won't append a ".npy" suffix.
            with open(tmp_path, "wb") as f:
                np.save(f, arr, allow_pickle=False)
            try:
                os.replace(tmp_path, path)
            except FileExistsError:
                # Another worker won the race.
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass
        except Exception:
            # Best-effort cache.
            try:
                if path is not None:
                    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
                    tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    def _get_decoded_frames_array(self, key: str, video_path: str) -> np.ndarray:
        """Return decoded resized RGB frames as (T, H, W, 3) uint8.

        Disk cache (when enabled) is the primary cache because it avoids rank×worker
        RAM duplication in DDP and leverages the OS page cache.
        """
        # Disk cache first.
        if self.disk_cache_dir is not None:
            loaded = self._load_frames_from_disk_cache(key)
            if loaded is not None:
                return loaded

            frames_list = self._decode_frames_uint8(video_path)
            if not frames_list:
                empty = np.zeros((0, self.image_size, self.image_size, 3), dtype=np.uint8)
                return empty

            frames_arr = np.stack(frames_list, axis=0).astype(np.uint8, copy=False)
            self._save_frames_to_disk_cache(key, frames_arr)

            # If mmap is enabled, prefer re-loading from disk to avoid holding the full
            # decoded array in RAM longer than needed.
            if self.disk_cache_mmap:
                reloaded = self._load_frames_from_disk_cache(key)
                if reloaded is not None:
                    return reloaded
            return frames_arr

        # RAM cache (per-worker) when disk cache is disabled.
        if self.cache_frames and self.cache_max_items > 0:
            cached = self._frames_cache.get(key)
            if cached is not None:
                self._frames_cache.move_to_end(key)
                return cached

        frames_list = self._decode_frames_uint8(video_path)
        if not frames_list:
            empty = np.zeros((0, self.image_size, self.image_size, 3), dtype=np.uint8)
            return empty
        frames_arr = np.stack(frames_list, axis=0).astype(np.uint8, copy=False)

        if self.cache_frames and self.cache_max_items > 0:
            self._frames_cache[key] = frames_arr
            self._frames_cache.move_to_end(key)
            if self.cache_policy != "all" and len(self._frames_cache) > self.cache_max_items:
                self._frames_cache.popitem(last=False)
        return frames_arr

    def _extract_frames(self, cache_key: str, video_path: str) -> torch.Tensor:
        """Extract frames from video based on target FPS."""
        try:
            # Decode/cached frames *before* augmentation to keep train-time randomness
            # intact while still avoiding repeated video decode.
            frames_arr = self._get_decoded_frames_array(cache_key, video_path)

            if frames_arr.shape[0] == 0:
                dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                out = self.augmentation(image=dummy)
                return out["image"].unsqueeze(0)

            # Apply the same random augmentation to all frames via ReplayCompose.
            first = self.augmentation(image=frames_arr[0])
            replay = first.get("replay")
            augmented_frames = [first["image"]]

            if replay is not None:
                import albumentations as A

                for i in range(1, frames_arr.shape[0]):
                    out = A.ReplayCompose.replay(replay, image=frames_arr[i])
                    augmented_frames.append(out["image"])
            else:
                for i in range(1, frames_arr.shape[0]):
                    out = self.augmentation(image=frames_arr[i])
                    augmented_frames.append(out["image"])

            return torch.stack(augmented_frames)
        
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            return torch.from_numpy(dummy).permute(2, 0, 1).unsqueeze(0)

    def _encode_caption(self, caption: str) -> torch.Tensor:
        """Encode caption to BPE tokens."""
        max_len = int(self.max_caption_length)
        eos_id = int(getattr(self.tokenizer, "eos_id", 3))
        pad_id = int(getattr(self.tokenizer, "pad_id", 0))

        token_ids = self.tokenizer.encode(caption)
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.long)

        token_ids = token_ids[: max(0, max_len - 1)]
        token_ids = token_ids + [eos_id]

        if len(token_ids) < max_len:
            token_ids = token_ids + [pad_id] * (max_len - len(token_ids))

        return torch.tensor(token_ids, dtype=torch.long)

    def _get_captions(self, all_captions: List[str]) -> Tuple[str, List[str]]:
        """Get caption for this dataset."""
        if self.stage == 'train':
            return random.choice(all_captions), all_captions
        else:
            return all_captions[0], all_captions

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, List[str], int]:
        item = self.data[idx]
        video_id = item['video_id']
        # Support both 'caption' and 'captions' keys
        captions = item.get('captions', item.get('caption', []))
        
        caption, all_captions = self._get_captions(captions)
        
        video_path = self.videos_dir / f"{video_id}.mp4"
        cache_key = self._cache_key_for_video(str(video_id))
        frames = self._extract_frames(cache_key, str(video_path))  # (T, 3, 224, 224)
        
        tokens = self._encode_caption(caption)
        
        return frames, tokens, caption, all_captions, frames.shape[0]


def collate_fn_baseline(batch: List[Tuple]) -> Dict[str, Any]:
    """Collate for baseline: single frame."""
    frames, tokens, texts, all_captions_list = zip(*batch)
    
    return {
        'frames': torch.stack(frames),  # (batch_size, 3, 224, 224)
        'tokens': torch.stack(tokens),  # (batch_size, max_len)
        'texts': list(texts),
        'all_captions': list(all_captions_list),
    }


def collate_fn_advanced(batch: List[Tuple]) -> Dict[str, Any]:
    """Collate for advanced: variable frames with padding."""
    frames_list, tokens_list, texts, all_captions_list, frame_lengths = zip(*batch)
    
    # Find max frame length
    max_T = max(frames.shape[0] for frames in frames_list)
    
    # Pad frame sequences
    padded_frames = []
    frame_masks = []
    
    for frames in frames_list:
        T = frames.shape[0]
        if T < max_T:
            pad_shape = (max_T - T, *frames.shape[1:])
            pad = torch.zeros(pad_shape, dtype=frames.dtype)
            padded = torch.cat([frames, pad], dim=0)
        else:
            padded = frames

        padded_frames.append(padded)

        # Mask: True for real frames, False for padding
        mask = torch.zeros(max_T, dtype=torch.bool)
        mask[:T] = True
        frame_masks.append(mask)
    
    return {
        'frames': torch.stack(padded_frames),  # (batch_size, max_T, 3, 224, 224)
        'frame_mask': torch.stack(frame_masks),  # (batch_size, max_T)
        'tokens': torch.stack(tokens_list),  # (batch_size, max_len)
        'texts': list(texts),
        'all_captions': list(all_captions_list),
        'frame_lengths': torch.tensor(frame_lengths, dtype=torch.long),
    }


class VideoDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for video captioning."""

    def __init__(
        self,
        json_dir: str,
        videos_dir: str,
        dataset_type: str = 'baseline',
        target_fps: float = 2.0,
        image_size: int = 224,
        normalization_mean: tuple = (0.485, 0.456, 0.406),
        normalization_std: tuple = (0.229, 0.224, 0.225),
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        multiprocessing_context: str = "spawn",
        pin_memory: bool = True,
        cache_frames: bool = False,
        cache_max_items: int = 0,
        cache_policy: str = "lru",
        disk_cache_dir: Optional[str] = None,
        disk_cache_mmap: bool = True,
        max_caption_length: int = 50,
        tokenizer_vocab_size: int = 8000,
        tokenizer_model_dir: str = 'models',
    ):
        super().__init__()

        _configure_runtime_for_data_loading()
        
        self.json_dir = Path(json_dir)
        self.videos_dir = Path(videos_dir)
        self.dataset_type = dataset_type
        self.target_fps = target_fps
        self.image_size = image_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.multiprocessing_context = multiprocessing_context
        self.pin_memory = bool(pin_memory)
        self.cache_frames = bool(cache_frames)
        self.cache_max_items = int(cache_max_items)
        self.cache_policy = str(cache_policy).lower()
        self.disk_cache_dir = disk_cache_dir
        self.disk_cache_mmap = bool(disk_cache_mmap)
        self.max_caption_length = max_caption_length
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.tokenizer_model_dir = tokenizer_model_dir
        
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and tokenizer."""

        t0 = time.perf_counter()
        
        # Initialize or load tokenizer
        tokenizer_model_path = Path(self.tokenizer_model_dir) / "bpe_model.model"
        self.tokenizer = BPETokenizer(vocab_size=self.tokenizer_vocab_size)
        
        if tokenizer_model_path.exists():
            logger.info(f"Loading tokenizer from: {tokenizer_model_path}")
            t_tok = time.perf_counter()
            self.tokenizer.load(str(tokenizer_model_path))
            logger.info(f"Tokenizer load time: {time.perf_counter() - t_tok:.3f}s")
        else:
            logger.info(f"Tokenizer model not found at {tokenizer_model_path}; training new tokenizer...")
            # Train tokenizer - find the correct train file
            captions = []
            
            # Try different train file names
            train_json = None
            for fname in ['train.json', 'train_split.json', 'train_9k_metadata.json']:
                path = self.json_dir / fname
                if path.exists():
                    train_json = path
                    break
            
            if train_json is None:
                raise FileNotFoundError(f"Could not find training JSON file in {self.json_dir}")
            
            with open(train_json) as f:
                train_data = json.load(f)
                
                # Handle both formats:
                # 1. List of dicts: [{"video_id": "...", "captions": [...]}, ...]
                # 2. Dict with 'video_files': {"video_files": [...], ...}
                if isinstance(train_data, dict) and 'video_files' in train_data:
                    items = train_data['video_files']
                elif isinstance(train_data, list):
                    items = train_data
                else:
                    items = [train_data]
                
                for item in items:
                    captions.extend(item.get('captions', item.get('caption', [])))
            
            t_train = time.perf_counter()
            self.tokenizer.train(captions, self.tokenizer_model_dir)
            logger.info(f"Tokenizer train time: {time.perf_counter() - t_train:.3f}s")
        
        # Helper to find the right file
        def find_json_file(dir_path, patterns):
            """Find first existing file from patterns list."""
            for pattern in patterns:
                path = Path(dir_path) / pattern
                if path.exists():
                    return str(path)
            raise FileNotFoundError(f"Could not find any of {patterns} in {dir_path}")
        
        # Create datasets
        if stage in ['fit', 'train', None]:
            train_json_path = find_json_file(self.json_dir, ['train.json', 'train_split.json', 'train_9k_metadata.json'])
            
            if self.dataset_type == 'baseline':
                self.train_dataset = BaselineVideoDataset(
                    json_path=train_json_path,
                    videos_dir=str(self.videos_dir),
                    tokenizer=self.tokenizer,
                    image_size=self.image_size,
                    max_caption_length=self.max_caption_length,
                    stage='train',
                    cache_frames=self.cache_frames,
                    cache_max_items=self.cache_max_items,
                    cache_policy=self.cache_policy,
                )
            else:  # advanced
                self.train_dataset = AdvancedVideoDataset(
                    json_path=train_json_path,
                    videos_dir=str(self.videos_dir),
                    tokenizer=self.tokenizer,
                    target_fps=self.target_fps,
                    image_size=self.image_size,
                    max_caption_length=self.max_caption_length,
                    normalization_mean=self.normalization_mean,
                    normalization_std=self.normalization_std,
                    augment=True,
                    stage='train',
                    cache_frames=self.cache_frames,
                    cache_max_items=self.cache_max_items,
                    cache_policy=self.cache_policy,
                    disk_cache_dir=self.disk_cache_dir,
                    disk_cache_mmap=self.disk_cache_mmap,
                )
        
        if stage in ['fit', 'validate', None]:
            val_json_path = find_json_file(self.json_dir, ['val.json', 'validation_split.json', 'val_split.json'])
            
            if self.dataset_type == 'baseline':
                self.val_dataset = BaselineVideoDataset(
                    json_path=val_json_path,
                    videos_dir=str(self.videos_dir),
                    tokenizer=self.tokenizer,
                    image_size=self.image_size,
                    max_caption_length=self.max_caption_length,
                    stage='val',
                    cache_frames=self.cache_frames,
                    cache_max_items=self.cache_max_items,
                    cache_policy=self.cache_policy,
                )
            else:  # advanced
                self.val_dataset = AdvancedVideoDataset(
                    json_path=val_json_path,
                    videos_dir=str(self.videos_dir),
                    tokenizer=self.tokenizer,
                    target_fps=self.target_fps,
                    image_size=self.image_size,
                    max_caption_length=self.max_caption_length,
                    normalization_mean=self.normalization_mean,
                    normalization_std=self.normalization_std,
                    augment=False,
                    stage='val',
                    cache_frames=self.cache_frames,
                    cache_max_items=self.cache_max_items,
                    cache_policy=self.cache_policy,
                    disk_cache_dir=self.disk_cache_dir,
                    disk_cache_mmap=self.disk_cache_mmap,
                )
        
        if stage in ['test', None]:
            test_json_path = find_json_file(self.json_dir, ['test.json', 'test_split.json', 'test_1k_metadata.json'])
            
            if self.dataset_type == 'baseline':
                self.test_dataset = BaselineVideoDataset(
                    json_path=test_json_path,
                    videos_dir=str(self.videos_dir),
                    tokenizer=self.tokenizer,
                    image_size=self.image_size,
                    max_caption_length=self.max_caption_length,
                    stage='test',
                    cache_frames=self.cache_frames,
                    cache_max_items=self.cache_max_items,
                    cache_policy=self.cache_policy,
                )
            else:  # advanced
                self.test_dataset = AdvancedVideoDataset(
                    json_path=test_json_path,
                    videos_dir=str(self.videos_dir),
                    tokenizer=self.tokenizer,
                    target_fps=self.target_fps,
                    image_size=self.image_size,
                    max_caption_length=self.max_caption_length,
                    normalization_mean=self.normalization_mean,
                    normalization_std=self.normalization_std,
                    augment=False,
                    stage='test',
                    cache_frames=self.cache_frames,
                    cache_max_items=self.cache_max_items,
                    cache_policy=self.cache_policy,
                    disk_cache_dir=self.disk_cache_dir,
                    disk_cache_mmap=self.disk_cache_mmap,
                )

        logger.info(f"DataModule setup total time: {time.perf_counter() - t0:.3f}s (stage={stage})")

    def train_dataloader(self) -> DataLoader:
        collate_fn = collate_fn_baseline if self.dataset_type == 'baseline' else collate_fn_advanced
        loader_kwargs: Dict[str, Any] = {
            "pin_memory": bool(self.pin_memory),
        }

        if self.num_workers and self.num_workers > 0:
            loader_kwargs.update(
                {
                    "persistent_workers": bool(self.persistent_workers),
                    "prefetch_factor": int(self.prefetch_factor),
                    "multiprocessing_context": str(self.multiprocessing_context),
                }
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=dataloader_worker_init_fn if self.num_workers and self.num_workers > 0 else None,
            **loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        collate_fn = collate_fn_baseline if self.dataset_type == 'baseline' else collate_fn_advanced
        loader_kwargs: Dict[str, Any] = {
            "pin_memory": bool(self.pin_memory),
        }

        if self.num_workers and self.num_workers > 0:
            loader_kwargs.update(
                {
                    "persistent_workers": bool(self.persistent_workers),
                    "prefetch_factor": int(self.prefetch_factor),
                    "multiprocessing_context": str(self.multiprocessing_context),
                }
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=dataloader_worker_init_fn if self.num_workers and self.num_workers > 0 else None,
            **loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        collate_fn = collate_fn_baseline if self.dataset_type == 'baseline' else collate_fn_advanced
        loader_kwargs: Dict[str, Any] = {
            "pin_memory": bool(self.pin_memory),
        }

        if self.num_workers and self.num_workers > 0:
            loader_kwargs.update(
                {
                    "persistent_workers": bool(self.persistent_workers),
                    "prefetch_factor": int(self.prefetch_factor),
                    "multiprocessing_context": str(self.multiprocessing_context),
                }
            )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            worker_init_fn=dataloader_worker_init_fn if self.num_workers and self.num_workers > 0 else None,
            **loader_kwargs,
        )

    def get_tokenizer(self) -> BPETokenizer:
        """Get tokenizer instance."""
        return self.tokenizer
