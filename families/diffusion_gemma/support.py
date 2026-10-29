# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact DiffusionGemma ownership, separate from autoregressive Gemma."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("diffusion_gemma", "diffusiongemma"),
    tasks=("text_generation", "vision_language_generation", "images_text_to_text"),
    default_task="text_generation",
    default_precision="fp16",
)
