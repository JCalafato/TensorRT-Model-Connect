# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checkpoint discovery and semantic task ownership."""

import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize("model_type", ["diffusion_gemma", "diffusiongemma"])
def test_diffusion_gemma_identity_and_default_task(model_type):
    family, support = resolve_family(
        ModelMetadata(config={"model_type": model_type}, model_index={})
    )
    assert family == "diffusion_gemma"
    assert support.default_task == "text_generation"
    assert {"text_generation", "vision_language_generation", "images_text_to_text"} <= set(
        support.tasks
    )
