"""Single CLI entrypoint using Fire."""

import fire

from video_captioning.inference.inference import run_inference
from video_captioning.onnx.export import export_advanced_to_onnx, export_baseline_to_onnx
from video_captioning.training.train_advanced import run_train_advanced
from video_captioning.training.train_baseline import run_train_baseline


def train_baseline(*hydra_args: str):
    """Run baseline training with optional Hydra overrides.

    Example:
            python -m video_captioning.commands train_baseline \
                training.max_epochs=3
    """
    run_train_baseline(*hydra_args)


def train_advanced(*hydra_args: str):
    """Run advanced (main) Transformer training with optional Hydra overrides.

    Example:
            python -m video_captioning.commands train_advanced training.max_epochs=3
    """
    run_train_advanced(*hydra_args)


def inference(model_type: str, weights_path: str, video_path: str, *hydra_args: str):
    """Run inference for a single video.

    Example:
            python -m video_captioning.commands inference baseline /path/to/model.ckpt \
                /path/to/video.mp4

            python -m video_captioning.commands inference advanced /path/to/model.ckpt \
                /path/to/video.mp4
    """

    caption = run_inference(model_type, weights_path, video_path, *hydra_args)
    print(caption)


def convert_onnx(
    model_type: str,
    weights_path: str,
    output_path: str,
    *hydra_args: str,
    opset: int = 17,
    batch_size: int = 1,
    seq_len: int = 16,
    max_frames: int = 16,
):
    """Convert a trained model to ONNX.

    Currently supported model types:
    - baseline

    Example:
            python -m video_captioning.commands convert_onnx baseline \
                artifacts/checkpoints/baseline.ckpt artifacts/onnx/baseline.onnx
    """

    model_type = str(model_type).lower()
    if model_type == "baseline":
        out = export_baseline_to_onnx(
            weights_path=weights_path,
            output_path=output_path,
            opset=opset,
            batch_size=batch_size,
            seq_len=seq_len,
            hydra_overrides=hydra_args,
        )
    elif model_type == "advanced":
        out = export_advanced_to_onnx(
            weights_path=weights_path,
            output_path=output_path,
            opset=opset,
            batch_size=batch_size,
            num_frames=max_frames,
            seq_len=seq_len,
            hydra_overrides=hydra_args,
        )
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Use 'baseline' or 'advanced' for ONNX export."
        )

    print(f"Saved ONNX model to: {out}")


def main():
    fire.Fire(
        {
            "train_baseline": train_baseline,
            "train_advanced": train_advanced,
            "inference": inference,
            "convert_onnx": convert_onnx,
        }
    )


if __name__ == "__main__":
    main()
