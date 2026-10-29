# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint mapping must not conflate phase scalars or unpacked tensors."""

import json

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

from families.diffusion_gemma.weights import Checkpoint


def test_exact_indexed_tensor_mapping(tmp_path):
    values = {
        "model.decoder.layers.0.layer_scalar": np.array([2.0], np.float32),
        "model.encoder.language_model.layers.0.layer_scalar": np.array([3.0], np.float32),
        "expert.weight": np.array([[0x81, 0x72]], np.uint8),
    }
    save_file(values, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(values, "part.safetensors")})
    )
    with Checkpoint(tmp_path) as reader:
        np.testing.assert_array_equal(
            reader.floating(reader.layer_scale_name(0, decoder=True)), [2]
        )
        np.testing.assert_array_equal(
            reader.floating(reader.layer_scale_name(0, decoder=False)), [3]
        )
        np.testing.assert_array_equal(reader.raw("expert.weight"), [[0x81, 0x72]])
        with pytest.raises(ValueError, match="explicit decoding"):
            reader.floating("expert.weight")
    assert not reader.readers


def test_bf16_and_fp8_preserve_original_storage(tmp_path):
    import torch
    from safetensors.torch import save_file as save_torch

    save_torch(
        {
            "bfloat": torch.tensor([1.125, -0.125], dtype=torch.bfloat16),
            "scale": torch.tensor([0.5, 448.0], dtype=torch.float8_e4m3fn),
        },
        tmp_path / "model.safetensors",
    )
    with Checkpoint(tmp_path) as reader:
        assert reader.raw("bfloat").dtype == np.dtype(ml_dtypes.bfloat16)
        assert reader.raw("scale").dtype == np.dtype(ml_dtypes.float8_e4m3fn)
        np.testing.assert_array_equal(reader.floating("bfloat"), [1.125, -0.125])
        np.testing.assert_array_equal(reader.floating("scale"), [0.5, 448.0])


def test_reject_index_escaping_checkpoint(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    with pytest.raises(ValueError, match="within its checkpoint"):
        Checkpoint(tmp_path)


def test_reject_index_claiming_absent_tensor(tmp_path):
    save_file({"actual": np.ones(1, np.float32)}, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"missing": "part.safetensors"}})
    )
    with pytest.raises(ValueError, match="indexed tensor is missing"):
        Checkpoint(tmp_path)
