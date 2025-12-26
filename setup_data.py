#!/usr/bin/env python3
"""
Unified data setup script for MSR-VTT dataset.
Downloads videos and metadata from HuggingFace, prepares train/val/test splits.
DVC tracks the resulting split JSON files to detect changes.
"""

import hashlib
import json
import os
import random
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import get_hf_file_metadata, hf_hub_download

# ============================================================================
# VIDEO DOWNLOAD
# ============================================================================


def ensure_videos_exist(video_dir: Path, min_videos: int = 10000) -> bool:
    """Check if videos directory exists and has sufficient videos."""
    if not video_dir.exists():
        return False

    mp4_count = len(list(video_dir.glob("*.mp4")))
    return mp4_count >= min_videos


def download_videos_archive(output_dir: Path):
    """Download MSRVTT_Videos.zip from HuggingFace and extract."""
    print("\n📥 Downloading MSR-VTT videos archive (2.1 GB)...")
    print("   This may take a few minutes...")

    # Create videos parent directory
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Download
    repo_id = "friedrichor/MSR-VTT"
    filename = "MSRVTT_Videos.zip"
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=None,
        local_dir=str(videos_dir.parent),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # Best-effort integrity check.
    # HuggingFace often exposes the LFS SHA256 as the file etag.
    try:
        meta = get_hf_file_metadata(
            url=f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}",
            timeout=30,
        )
        etag = str(getattr(meta, "etag", "") or "").strip('"')
        if len(etag) == 64 and all(c in "0123456789abcdef" for c in etag.lower()):
            sha256 = hashlib.sha256()
            with open(zip_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    sha256.update(chunk)
            digest = sha256.hexdigest()
            if digest.lower() != etag.lower():
                raise RuntimeError(
                    "Downloaded archive hash mismatch: "
                    f"expected sha256={etag}, got sha256={digest}"
                )
    except Exception as e:
        print(f"⚠️  Hash verification skipped/failed (non-fatal): {e}")

    print("✅ Download complete, extracting...")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(videos_dir)

    # Cleanup
    os.remove(zip_path)
    print("✅ Videos extracted to", videos_dir / "video")


# ============================================================================
# METADATA DOWNLOAD
# ============================================================================


def download_msr_vtt(output_dir: str = "data", split: str = "train_9k", num_samples: int = None):
    """Download MSR-VTT dataset from HuggingFace.

    Args:
        output_dir: Output directory for dataset metadata
        split: Dataset configuration ('train_9k', 'train_7k', 'test_1k')
        num_samples: Number of samples to download (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Downloading MSR-VTT {split} metadata from HuggingFace")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_path}")

    # Load dataset from HuggingFace
    print("\nLoading dataset from HuggingFace...")
    dataset_dict = load_dataset(
        "friedrichor/MSR-VTT",
        name=split,
        cache_dir=str(output_path / ".cache"),
    )

    # Get the appropriate split from the DatasetDict
    if "train" in dataset_dict:
        dataset = dataset_dict["train"]
        split_name = "train"
    elif "test" in dataset_dict:
        dataset = dataset_dict["test"]
        split_name = "test"
    else:
        # For other structures, get first available split
        split_name = list(dataset_dict.keys())[0]
        dataset = dataset_dict[split_name]

    print(f"  Using split: {split_name}")

    total_samples = len(dataset)
    if num_samples:
        total_samples = min(num_samples, total_samples)

    print(f"Total samples available: {len(dataset)}")
    print(f"Processing: {total_samples} samples")

    # Prepare dataset info
    dataset_info = {
        "split": split,
        "num_samples": total_samples,
        "source": f"friedrichor/MSR-VTT ({split})",
        "columns": dataset.column_names,
        "video_files": [],
        "seed": 42,  # For reproducibility
    }

    # Process each sample
    print(f"\nExtracting metadata from {total_samples} samples...")
    for idx in range(total_samples):
        if idx % max(1, total_samples // 10) == 0:
            print(f"  Progress: {idx}/{total_samples}")

        sample = dataset[idx]
        video_id = sample.get("video_id", f"video_{idx:05d}")

        # Extract metadata
        dataset_info["video_files"].append(
            {
                "video_id": video_id,
                "caption": sample.get("caption", ""),
                "source": sample.get("source", ""),
                "category": sample.get("category", 0),
                "url": sample.get("url", ""),
                "start_time": sample.get("start time", 0.0),
                "end_time": sample.get("end time", 0.0),
            }
        )

    print(f"  Progress: {total_samples}/{total_samples}")

    # Save dataset metadata (this will be tracked by DVC)
    metadata_path = output_path / f"{split}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\n✅ Download completed!")
    print(f"   Metadata saved to: {metadata_path}")
    print(f"   Total samples: {total_samples}")

    return metadata_path


# ============================================================================
# DATASET SPLITTING
# ============================================================================


def prepare_splits(data_dir: str = "data", seed: int = 42):
    """
    Prepare dataset splits:
    - Train: 8000 samples from train_9k
    - Validation: 1000 samples from train_9k
    - Test: 1000 samples from test_1k
    """
    random.seed(seed)
    data_path = Path(data_dir)

    print("\n" + "=" * 80)
    print("PREPARING DATASET SPLITS")
    print("=" * 80)

    # Load train_9k metadata
    train_metadata_path = data_path / "train_9k_metadata.json"
    if not train_metadata_path.exists():
        print(f"❌ {train_metadata_path} not found!")
        return False

    with open(train_metadata_path) as f:
        train_metadata = json.load(f)

    actual_train_samples = len(train_metadata["video_files"])
    print(f"✓ Loaded train_9k metadata: {actual_train_samples} samples")

    # Load test_1k metadata
    test_metadata_path = data_path / "test_1k_metadata.json"
    if not test_metadata_path.exists():
        print(f"⚠️  {test_metadata_path} not found yet (still downloading?)")
        test_videos = []
    else:
        with open(test_metadata_path) as f:
            test_metadata = json.load(f)
        test_videos = test_metadata["video_files"]
        print(f"✓ Loaded test_1k metadata: {len(test_videos)} samples")

    # Split train_9k into train and validation
    all_train_videos = train_metadata["video_files"]
    random.shuffle(all_train_videos)

    # Use 8000 for train, remaining for validation (up to 1000)
    train_videos = all_train_videos[:8000]
    val_videos = all_train_videos[8000 : min(9000, len(all_train_videos))]

    print("✓ Split train_9k into:")
    print(f"    - Train: {len(train_videos)} samples")
    print(f"    - Validation: {len(val_videos)} samples")

    # Prepare split metadata files
    splits = {
        "train": {
            "split": "train",
            "num_samples": len(train_videos),
            "source": "friedrichor/MSR-VTT (train_9k, indices 0-8000)",
            "columns": train_metadata["columns"],
            "video_files": train_videos,
            "seed": seed,
        },
        "validation": {
            "split": "validation",
            "num_samples": len(val_videos),
            "source": "friedrichor/MSR-VTT (train_9k, indices 8000-9000)",
            "columns": train_metadata["columns"],
            "video_files": val_videos,
            "seed": seed,
        },
        "test": {
            "split": "test",
            "num_samples": len(test_videos),
            "source": "friedrichor/MSR-VTT (test_1k)",
            "columns": train_metadata["columns"],
            "video_files": test_videos,
            "seed": seed,
        },
    }

    # Save split files
    for split_name, split_data in splits.items():
        output_path = data_path / f"{split_name}_split.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"✓ Saved {split_name} split: {output_path}")

    # Create summary
    summary = {
        "dataset": "MSR-VTT",
        "total_samples": len(train_videos) + len(val_videos) + len(test_videos),
        "splits": {
            "train": len(train_videos),
            "validation": len(val_videos),
            "test": len(test_videos),
        },
        "seed": seed,
        "split_files": {
            "train": "train_split.json",
            "validation": "validation_split.json",
            "test": "test_split.json",
        },
    }

    summary_path = data_path / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Dataset summary saved to: {summary_path}")
    print("\n" + "=" * 80)
    print("DATASET READY!")
    print("=" * 80)
    print(f"Total samples: {summary['total_samples']}")
    print(f"  - Train:      {summary['splits']['train']:>5}")
    print(f"  - Validation: {summary['splits']['validation']:>5}")
    print(f"  - Test:       {summary['splits']['test']:>5}")

    return True


# ============================================================================
# MAIN SETUP
# ============================================================================


def setup_data(output_dir: str = "data"):
    """Main data setup function - coordinates all data preparation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MSR-VTT Data Setup")
    print("=" * 80)

    # Step 1: Check if videos exist
    video_dir = output_path / "videos" / "video"
    if not ensure_videos_exist(video_dir):
        download_videos_archive(output_path)
    else:
        mp4_count = len(list(video_dir.glob("*.mp4")))
        print(f"✅ Videos already present ({mp4_count} files)")

    # Step 2: Download metadata for train split
    print("\n📥 Downloading train metadata...")
    metadata_path = Path(output_dir) / "train_9k_metadata.json"
    if not metadata_path.exists():
        download_msr_vtt(output_dir, split="train_9k")
    else:
        print("✅ Train metadata exists")

    # Step 3: Download metadata for test split
    print("\n📥 Downloading test metadata...")
    test_metadata_path = Path(output_dir) / "test_1k_metadata.json"
    if not test_metadata_path.exists():
        download_msr_vtt(output_dir, split="test_1k")
    else:
        print("✅ Test metadata exists")

    # Step 4: Prepare splits
    print("\n🔀 Preparing train/validation/test splits...")
    prepare_splits(output_dir)

    print("\n" + "=" * 80)
    print("✅ Data setup complete!")
    print("=" * 80)


if __name__ == "__main__":
    setup_data()
