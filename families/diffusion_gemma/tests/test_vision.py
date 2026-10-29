# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual TensorRT vision encoding versus independently generated HF outputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from .test_graph import _environment, _execute

DATA = Path(__file__).parent / "data" / "tiny-vision"


def test_vision_fixture_provenance():
    metadata = json.loads((DATA / "provenance.json").read_text())
    assert metadata["package"] == "transformers==5.11.0"
    assert metadata["source_revision"] == "v5.11.0"
    assert metadata["fixture_file"] == "reference.npz"
    for name in metadata["files"]:
        assert (DATA / name).is_file()
    with np.load(DATA / metadata["fixture_file"], allow_pickle=False) as reference:
        assert set(reference.files) == set(metadata["arrays"])
        for name, layout in metadata["arrays"].items():
            assert list(reference[name].shape) == layout["shape"]
            assert str(reference[name].dtype) == layout["dtype"]
            assert np.isfinite(reference[name]).all()


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_vision_projection_matches_independent_hf(precision):
    from families.diffusion_gemma.config import ModelConfig
    from families.diffusion_gemma.engine import build_vision_engine
    from families.diffusion_gemma.weights import Checkpoint

    trt, logger, _, _ = _environment()
    config = ModelConfig.from_directory(DATA)
    with Checkpoint(DATA) as weights:
        plan = build_vision_engine(config, weights, precision=precision)
    reference = np.load(DATA / "reference.npz")
    for label in ("square", "padded"):
        actual = _execute(
            trt,
            logger,
            plan,
            {
                "pixel_values": reference[label + "_pixels"],
                "pixel_position_ids": reference[label + "_positions"],
            },
        )
        np.testing.assert_array_equal(actual["image_feature_mask"], reference[label + "_mask"])
        selected = actual["image_features"][actual["image_feature_mask"].astype(bool)]
        np.testing.assert_allclose(
            selected, reference[label + "_" + precision], rtol=2e-3, atol=2e-4
        )
