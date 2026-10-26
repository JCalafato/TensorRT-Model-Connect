# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reasoner checkpoint identity, validation, and indexed BF16 weight loading."""

import json

import numpy as np
import pytest

from families.cosmos3_edge.config import ModelConfig
from families.cosmos3_edge.checkpoint_mapper import CheckpointWeights


def _raw():
    # Model shapes follow the documented checkpoint, without its weights.
    return {
        "model_type": "cosmos3_edge",
        "text_config": {
            "model_type": "cosmos3_edge_text", "hidden_act": "relu2", "hidden_size": 2048,
            "intermediate_size": 9216, "num_hidden_layers": 28, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 131072,
            "max_position_embeddings": 131072, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e8,
                                "mrope_section": [24, 20, 20]},
        },
        "vision_config": {
            "hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27,
            "num_attention_heads": 16, "num_channels": 3, "num_patches": 256,
            "patch_size": 16, "spatial_merge_size": 2, "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh",
        },
        "projector_config": {
            "input_hidden_size": 1152, "out_hidden_size": 2048, "spatial_merge_size": 2,
            "use_postshuffle_norm": False, "merger_intermediate_size": 11520,
        },
    }


def test_reasoner_configuration():
    config = ModelConfig.from_dict(_raw())
    assert config.num_hidden_layers == 28
    assert config.mrope_section == (24, 20, 20)
    assert config.vision["num_hidden_layers"] == 27


@pytest.mark.parametrize("field,value", [
    ("hidden_act", "silu"), ("num_hidden_layers", True), ("hidden_size", 2049),
    ("num_key_value_heads", 3), ("attention_bias", True), ("qk_norm_for_text", True),
])
def test_incompatible_text_contract_rejected(field, value):
    raw = _raw()
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


@pytest.mark.parametrize("field,value", [
    ("rope_theta", float("nan")), ("mrope_section", [24, 20, 21]), ("rope_type", "linear"),
])
def test_incompatible_rotary_contract_rejected(field, value):
    raw = _raw()
    raw["text_config"]["rope_parameters"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw)


def test_generator_metadata_is_not_reasoner_metadata():
    raw = _raw()
    raw["model_type"] = "cosmos3_nano"
    with pytest.raises(ValueError, match="reasoning checkpoint"):
        ModelConfig.from_dict(raw)


def test_indexed_weights_preserve_bfloat16_range(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    shard = tmp_path / "transformer/weights.safetensors"
    shard.parent.mkdir()
    expected = torch.tensor([[1e6, -1e6, 0.25]], dtype=torch.bfloat16)
    save_file({"layers.0.mlp.up_proj.weight": expected, "unrelated.policy.weight": torch.ones(1)}, shard)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layers.0.mlp.up_proj.weight": "transformer/weights.safetensors"},
    }))
    weights = CheckpointWeights(tmp_path)
    actual = weights["layers.0.mlp.up_proj.weight"]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected.float().numpy())
    with pytest.raises(KeyError):
        weights["unrelated.policy.weight"]


def test_index_cannot_escape_checkpoint(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (tmp_path / "outside.safetensors").write_bytes(b"not a checkpoint")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"norm.weight": "../outside.safetensors"},
    }))
    with pytest.raises(ValueError, match="non-contained"):
        CheckpointWeights(root)


def test_hf_cache_shard_links_are_read_without_copying(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    blobs = tmp_path / "models--example--reasoner/blobs"
    blobs.mkdir(parents=True)
    root = blobs.parent / "snapshots/revision"
    (root / "transformer").mkdir(parents=True)
    save_file({"norm.weight": torch.ones(2)}, blobs / "hash")
    (root / "transformer/part.safetensors").symlink_to(blobs / "hash")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"norm.weight": "transformer/part.safetensors"},
    }))
    np.testing.assert_array_equal(CheckpointWeights(root)["norm.weight"], np.ones(2))


def test_arbitrary_external_shard_link_is_rejected(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"not a checkpoint")
    (root / "part.safetensors").symlink_to(outside)
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"norm.weight": "part.safetensors"},
    }))
    with pytest.raises(ValueError, match="non-contained"):
        CheckpointWeights(root)
