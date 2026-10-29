# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the complete native driving task, not a standalone backbone task."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if (
        metadata.model_type == "alpamayo_r1"
        and "AlpamayoR1" in metadata.architectures
        and not metadata.config.get("quantization_config")
    ):
        return FamilySupport(("structured_decision",), "structured_decision", "fp16")
    return None
