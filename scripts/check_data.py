#!/usr/bin/env python3
"""
Ensure dataset is available. If not, download it.
This script checks if data splits exist and downloads if needed.
"""

import json
import subprocess
import sys
from pathlib import Path


def check_data_exists() -> bool:
    """Check if all required data files exist."""
    data_dir = Path("data")
    required_files = [
        "train_split.json",
        "validation_split.json",
        "test_split.json",
        "dataset_summary.json",
    ]

    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"❌ Missing: {file_path}")
            return False

    # Check videos
    video_dir = data_dir / "videos" / "video"
    if not video_dir.exists():
        print(f"❌ Missing videos directory: {video_dir}")
        return False

    mp4_files = list(video_dir.glob("*.mp4"))
    if len(mp4_files) < 10000:
        print(f"❌ Incomplete videos: found {len(mp4_files)}/10000 MP4 files")
        return False

    print("✅ All data files and videos found!")
    return True


def download_data() -> bool:
    """Download dataset if missing."""
    print("\n" + "=" * 80)
    print("DOWNLOADING DATASET")
    print("=" * 80)

    try:
        # Download train_9k
        print("\n1. Downloading train_9k (9000 samples)...")
        subprocess.run(
            [
                "python",
                "download_msr_vtt.py",
                "--output-dir",
                "data",
                "--split",
                "train_9k",
                "--num-samples",
                "9000",
            ],
            check=True,
            cwd=Path(__file__).parent,
        )

        # Download test_1k
        print("\n2. Downloading test_1k (1000 samples)...")
        subprocess.run(
            ["python", "download_msr_vtt.py", "--output-dir", "data", "--split", "test_1k"],
            check=True,
            cwd=Path(__file__).parent,
        )

        # Prepare splits
        print("\n3. Preparing dataset splits...")
        subprocess.run(["python", "prepare_splits.py"], check=True, cwd=Path(__file__).parent)

        print("\n✅ Dataset downloaded successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading dataset: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def print_data_info():
    """Print dataset information."""
    summary_path = Path("data/dataset_summary.json")

    if not summary_path.exists():
        print("❌ Dataset summary not found")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Dataset: {summary['dataset']}")
    print(f"Total samples: {summary['total_samples']:,}")
    print("Splits:")
    print(f"  - Train:      {summary['splits']['train']:>6,}")
    print(f"  - Validation: {summary['splits']['validation']:>6,}")
    print(f"  - Test:       {summary['splits']['test']:>6,}")
    print(f"Seed: {summary['seed']}")
    print("=" * 80)


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("DATASET AVAILABILITY CHECK")
    print("=" * 80)

    # Check if data exists
    if check_data_exists():
        print_data_info()
        return 0

    # If not, download
    print("\n⚠️  Dataset not found. Starting download...")
    if download_data():
        print_data_info()
        return 0
    else:
        print("\n❌ Failed to download dataset")
        return 1


if __name__ == "__main__":
    sys.exit(main())
