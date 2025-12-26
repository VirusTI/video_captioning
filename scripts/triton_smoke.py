#!/usr/bin/env python3
"""Minimal Triton smoke test (HTTP v2) without extra deps.

Checks:
- server readiness
- model readiness
- runs a single inference with binary tensors

Usage:
    python scripts/triton_smoke.py main baseline --url http://localhost:8000
    python scripts/triton_smoke.py main advanced --url http://localhost:8000

Or (equivalent Fire form):
    python scripts/triton_smoke.py main --model=baseline --url=http://localhost:8000

Notes:
- Uses Triton HTTP binary tensor extension via `Inference-Header-Content-Length`.
- Inputs are simple (zeros / small integers). This is for "is it alive" checks,
  not quality evaluation.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

import fire
import numpy as np


@dataclass
class InferOutput:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    data: np.ndarray


def _http_get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _http_get_text(url: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        raw = resp.read()
    return status, raw.decode("utf-8", errors="replace")


def _http_post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200))
        raw = resp.read()
    return status, json.loads(raw.decode("utf-8"))


def _np_to_triton_dtype(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return "FP32"
    if dtype == np.int64:
        return "INT64"
    if dtype == np.int32:
        return "INT32"
    raise ValueError(f"Unsupported dtype for Triton HTTP: {dtype}")


def _parse_triton_dtype(triton_dtype: str) -> np.dtype:
    if triton_dtype == "FP32":
        return np.float32
    if triton_dtype == "INT64":
        return np.int64
    if triton_dtype == "INT32":
        return np.int32
    raise ValueError(f"Unsupported output dtype from Triton HTTP: {triton_dtype}")


def _infer_http_binary(
    *,
    base_url: str,
    model_name: str,
    inputs: dict[str, np.ndarray],
    output_names: list[str],
    timeout: float,
) -> list[InferOutput]:
    infer_url = f"{base_url}/v2/models/{model_name}/infer"

    # Build JSON header.
    in_specs = []
    binary_inputs: list[bytes] = []
    for name, arr in inputs.items():
        arr = np.ascontiguousarray(arr)
        inputs[name] = arr
        bin_bytes = arr.tobytes(order="C")
        binary_inputs.append(bin_bytes)
        in_specs.append(
            {
                "name": name,
                "datatype": _np_to_triton_dtype(arr.dtype),
                "shape": list(arr.shape),
                "parameters": {"binary_data_size": len(bin_bytes)},
            }
        )

    out_specs = []
    for name in output_names:
        out_specs.append({"name": name, "parameters": {"binary_data": True}})

    header = {"inputs": in_specs, "outputs": out_specs}
    header_bytes = json.dumps(header).encode("utf-8")

    body = header_bytes + b"".join(binary_inputs)

    req = urllib.request.Request(infer_url, data=body, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    req.add_header("Inference-Header-Content-Length", str(len(header_bytes)))

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_body = resp.read()
        header_len = int(resp.headers.get("Inference-Header-Content-Length", "0"))

    if header_len <= 0:
        raise RuntimeError(
            "Triton response missing Inference-Header-Content-Length; "
            "cannot parse binary outputs."
        )

    resp_header = json.loads(resp_body[:header_len].decode("utf-8"))
    bin_blob = memoryview(resp_body)[header_len:]

    outputs: list[InferOutput] = []
    offset = 0
    for out in resp_header.get("outputs", []):
        name = out["name"]
        shape = tuple(int(x) for x in out["shape"])
        dtype = _parse_triton_dtype(out["datatype"])
        nbytes = int(out.get("parameters", {}).get("binary_data_size", 0))
        if nbytes <= 0:
            raise RuntimeError(f"Output {name} has no binary_data_size")

        chunk = bin_blob[offset : offset + nbytes]
        offset += nbytes
        arr = np.frombuffer(chunk, dtype=dtype).reshape(shape)
        outputs.append(InferOutput(name=name, shape=shape, dtype=dtype, data=arr))

    return outputs


def main(
    model: str,
    url: str = "http://localhost:8000",
    timeout: float = 10.0,
    seq_len: int = 16,
    num_frames: int = 16,
) -> int:
    model_name = str(model).lower()
    if model_name not in {"baseline", "advanced"}:
        raise ValueError("model must be 'baseline' or 'advanced'")

    base_url = str(url).rstrip("/")

    # 1) Server readiness
    try:
        ready_status, ready_text = _http_get_text(f"{base_url}/v2/health/ready", timeout)
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach Triton at {base_url}: {e}")
        return 2

    # Some Triton builds return an empty body with 200 OK for health endpoints.
    if ready_status != 200:
        print(f"ERROR: Triton not ready (HTTP {ready_status}): {ready_text!r}")
        return 2

    # 2) Model readiness
    model_ready_status, model_ready_text = _http_get_text(
        f"{base_url}/v2/models/{model_name}/ready", timeout
    )
    if model_ready_status != 200:
        # Helpful context: list what's in the repository.
        try:
            idx_status, idx = _http_post_json(
                f"{base_url}/v2/repository/index", {"ready": True}, timeout
            )
            idx_str = json.dumps(idx)
        except Exception:
            idx_status, idx_str = 0, "<failed to query /v2/repository/index>"

        print(
            f"ERROR: model not ready (HTTP {model_ready_status}): {model_ready_text!r}\n"
            f"Repository index (HTTP {idx_status}): {idx_str}"
        )
        return 2

    # 3) Inference
    seq_len = int(seq_len)

    if model_name == "baseline":
        frames = np.zeros((1, 3, 224, 224), dtype=np.float32)
        tokens = np.ones((1, seq_len), dtype=np.int64)
        inputs = {"frames": frames, "tokens": tokens}
    else:
        num_frames = int(num_frames)
        frames = np.zeros((1, num_frames, 3, 224, 224), dtype=np.float32)
        tokens = np.ones((1, seq_len), dtype=np.int64)
        frame_mask = np.ones((1, num_frames), dtype=np.int64)
        inputs = {"frames": frames, "tokens": tokens, "frame_mask": frame_mask}

    outputs = _infer_http_binary(
        base_url=base_url,
        model_name=model_name,
        inputs=inputs,
        output_names=["logits"],
        timeout=timeout,
    )

    logits = next(o for o in outputs if o.name == "logits").data
    finite = np.isfinite(logits).all()
    print(f"server=ready model=ready model={model_name} logits.shape={tuple(logits.shape)} dtype={logits.dtype}")
    print(
        "logits stats:",
        f"min={np.nanmin(logits):.6g}",
        f"max={np.nanmax(logits):.6g}",
        f"mean={np.nanmean(logits):.6g}",
        f"all_finite={bool(finite)}",
    )

    return 0 if finite else 1


def entrypoint(
    model: str,
    url: str = "http://localhost:8000",
    timeout: float = 10.0,
    seq_len: int = 16,
    num_frames: int = 16,
) -> None:
    raise SystemExit(
        main(model=model, url=url, timeout=timeout, seq_len=seq_len, num_frames=num_frames)
    )


if __name__ == "__main__":
    fire.Fire({"main": entrypoint})
