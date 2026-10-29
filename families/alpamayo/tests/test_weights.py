# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy storage, source dtypes and family namespace mapping."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from families.alpamayo import weights


def checkpoint(directory):
    first = {
        "vlm.model.language_model.embed_tokens.weight": torch.arange(12)
        .reshape(4, 3)
        .to(torch.bfloat16),
        "vlm.lm_head.weight": torch.ones(4, 3, dtype=torch.float16),
        "vlm.model.visual.norm.weight": torch.ones(3, dtype=torch.float16),
    }
    second = {
        "expert.norm.weight": torch.ones(8, dtype=torch.float16),
        "action_in_proj.norm.weight": torch.ones(2, dtype=torch.float32),
        "action_out_proj.bias": torch.zeros(2, dtype=torch.float16),
        "action_space.accel_std": torch.tensor(0.6810426712036133, dtype=torch.float32),
    }
    mapping = {}
    for name, values in (("part1.safetensors", first), ("part2.safetensors", second)):
        save_file(values, directory / name)
        mapping.update({key: name for key in values})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return mapping


def test_lazy_shards_and_cached_conversion(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    opened = []
    original = weights.safe_open

    def counted(path, **kwargs):
        opened.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(weights, "safe_open", counted)
    with weights.Weights(tmp_path) as reader:
        assert opened == []
        assert reader.info("text.embed_tokens.weight") == ("BF16", (4, 3))
        array = reader.array("text.embed_tokens.weight", np.float16)
        assert reader.array("text.embed_tokens.weight", np.float16) is array
        np.testing.assert_array_equal(array, np.arange(12, dtype=np.float16).reshape(4, 3))
        assert reader.source_name("text.lm_head.weight") == "vlm.lm_head.weight"
        assert reader.info("vision.norm.weight") == ("F16", (3,))
        assert opened == ["part1.safetensors"]
        value = reader.array("action_space.accel_std")
        assert value.dtype == np.float32
        assert value.item() == 0.6810426712036133
        assert value.item() != 0.6796875  # The evaluator's BF16-rounded constant.
        assert opened == ["part1.safetensors", "part2.safetensors"]
    np.testing.assert_array_equal(array, np.arange(12, dtype=np.float16).reshape(4, 3))
    with pytest.raises(RuntimeError, match="closed"):
        reader.info("expert.norm.weight")


def test_missing_weight_and_unsupported_conversion(tmp_path):
    checkpoint(tmp_path)
    with weights.Weights(tmp_path) as reader:
        assert len(reader.names()) == 7
        with pytest.raises(KeyError):
            reader.array("expert.missing")
        with pytest.raises(ValueError, match="float16 or float32"):
            reader.array("expert.norm.weight", np.int8)


@pytest.mark.parametrize(
    "mapping",
    [
        {"expert.norm.weight": "../outside.safetensors"},
        {"unknown.weight": "part1.safetensors"},
        {},
    ],
)
def test_reject_invalid_index(tmp_path, mapping):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError):
        weights.Weights(tmp_path)
