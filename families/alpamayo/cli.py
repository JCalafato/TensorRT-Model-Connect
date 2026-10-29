# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned Alpamayo driving bundle construction and execution selection."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    max_batch_size: int = 1
    max_sequence_length: int = 4096
    max_new_tokens: int = 128
    workspace_bytes: int = 4 << 30
    precision: str = "fp16"
    task: str = "structured_decision"
    backend: str = "trt"
    output_path: Path | None = None
    execution_variant: str = "auto"
    verbose: bool = False

    def __post_init__(self):
        if self.execution_variant not in ("auto", "native"):
            raise ValueError("Alpamayo execution_variant must be auto or native")
        if (self.precision, self.task, self.backend) != ("fp16", "structured_decision", "trt"):
            raise ValueError("Alpamayo driving requires FP16, structured_decision and TensorRT")
        if type(self.max_batch_size) is not int or self.max_batch_size not in (1, 6):
            raise ValueError("Documented Alpamayo batch capacities are one and six")
        for name in ("max_sequence_length", "max_new_tokens", "workspace_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_sequence_length <= self.max_new_tokens:
            raise ValueError("Sequence capacity must leave room for the driving prompt")


def normalize_build_request(request) -> BuildRequest:
    """Map the existing generic contract; all model-specific admission stays here."""
    if isinstance(request, BuildRequest):
        return request
    # Compatibility is structural; declared family builders must not import
    # or depend on the shared request class.
    fields = (
        "family",
        "model_dir",
        "output_path",
        "task",
        "precision",
        "backend",
        "max_batch_size",
        "max_sequence_length",
        "verbose",
        "image_height",
        "image_width",
        "video_num_frames",
        "quantization",
        "mtp_seq_len",
        "tensor_parallel_size",
        "context_parallel_size",
        "fp32_layers",
        "dynamic_kv_cache",
    )
    if not all(hasattr(request, name) for name in fields):
        raise TypeError("Alpamayo requires its family request or compatible build fields")
    if request.family != "alpamayo":
        raise ValueError("Alpamayo cannot build another family's request")
    unsupported = {
        "image_height": request.image_height,
        "image_width": request.image_width,
        "video_num_frames": request.video_num_frames,
        "quantization": request.quantization,
        "mtp_seq_len": request.mtp_seq_len,
    }
    if any(value is not None for value in unsupported.values()):
        raise ValueError(
            "Alpamayo uses trained camera geometry and FP16 without speculative decoding"
        )
    if (
        request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.fp32_layers
        or request.dynamic_kv_cache
    ):
        raise ValueError("Alpamayo requires single-device FP16 and its owning KV capacity policy")
    return BuildRequest(
        model_dir=request.model_dir,
        max_batch_size=request.max_batch_size,
        max_sequence_length=request.max_sequence_length or 4096,
        precision=request.precision,
        task=request.task,
        backend=request.backend,
        output_path=request.output_path,
        verbose=request.verbose,
    )


def build(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    max_batch_size: int = 1,
    max_sequence_length: int = 4096,
    max_new_tokens: int = 128,
    execution_variant: str = "auto",
):
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from tensorrt_model_connect.model_support import resolve_model
    from .model import build as build_model

    request = BuildRequest(
        resolve_model(model, revision),
        max_batch_size,
        max_sequence_length,
        max_new_tokens,
        output_path=Path(output),
        execution_variant=execution_variant,
    )
    writer = BundleWriter(output)
    writer.record_build(request)
    try:
        build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return 0
