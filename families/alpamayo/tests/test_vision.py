# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visual encoder math and cross-image/frame isolation against pinned fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest

from families.alpamayo.config import VisionConfig
from families.alpamayo.tests.test_action import (
    FixtureWeights,
    environment,
    execute,
    validate_fixture,
)


DATA = Path(__file__).parent / "data"


def test_vision_fixture_provenance():
    provenance = json.loads((DATA / "vision.json").read_text())
    assert provenance["package"] == "transformers==4.57.1"
    assert provenance["revision"] == "v4.57.1"
    validate_fixture("vision.npz", provenance)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_visual_features_deepstack_and_frame_isolation(precision):
    from families.alpamayo.vision_engine import build_vision_engine

    provenance = json.loads((DATA / "vision.json").read_text())
    config = VisionConfig.parse(provenance["config"])
    trt, logger, _, _ = environment()
    tolerance = dict(rtol=2e-5, atol=2e-6) if precision == "fp32" else dict(rtol=2e-3, atol=2e-4)
    with np.load(DATA / "vision.npz") as fixture:
        plan = build_vision_engine(
            config, FixtureWeights(fixture), precision, patch_capacity=32, workspace_bytes=1 << 28
        )
        for mode in ("single", "mixed"):
            inputs = {
                name: fixture[f"{mode}_{name}"]
                for name in (
                    "pixels",
                    "position_indices",
                    "position_weights",
                    "mask",
                    "cosine",
                    "sine",
                )
            }
            actual = execute(trt, logger, plan, inputs)
            assert set(actual) == {"features", "deepstack_0", "deepstack_1", "deepstack_2"}
            for name, value in actual.items():
                np.testing.assert_allclose(
                    value, fixture[f"{precision}_{mode}_{name}"], **tolerance
                )
            if mode == "mixed":
                changed = dict(inputs)
                changed["pixels"] = inputs["pixels"].copy()
                changed["pixels"][8:] *= -1
                other = execute(trt, logger, plan, changed)
                for name in actual:
                    # The first image contains eight patches / two merged
                    # tokens, and cannot attend to any later frame's pixels.
                    np.testing.assert_array_equal(actual[name][:2], other[name][:2])
