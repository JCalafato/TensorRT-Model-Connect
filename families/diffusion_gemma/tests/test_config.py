# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact published checkpoint defaults; not full-model qualification."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from families.diffusion_gemma.config import ModelConfig

DATA = Path(__file__).parent / "data"


def source():
    return json.loads((DATA / "config.json").read_text()), json.loads(
        (DATA / "generation_config.json").read_text()
    )


def test_published_attention_and_diffusion_defaults():
    config = ModelConfig.from_directory(DATA)
    assert config.text["hidden_size"] == 2816
    assert config.text["num_hidden_layers"] == 30
    assert config.attention_geometry(0) == (256, 8)
    assert config.attention_geometry(5) == (512, 2)
    assert config.text["attention_k_eq_v"] and config.text["enable_moe_block"]
    assert config.generation.canvas_length == 256
    assert config.generation.max_denoising_steps == 48
    assert (config.generation.t_max, config.generation.t_min) == (0.8, 0.4)
    assert config.generation.entropy_bound == 0.1
    assert config.generation.confidence_threshold == 0.005
    assert config.generation.stability_threshold == 1
    assert config.generation.eos_token_ids == (1, 106, 50)


def test_documented_alias_does_not_change_architecture():
    raw, generation = source()
    raw["model_type"] = "diffusiongemma"
    config = ModelConfig.from_dict(raw, generation)
    assert config.attention_geometry(5) == (512, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_hidden_layers", 0),
        ("hidden_size", True),
        ("num_key_value_heads", 3),
        ("head_dim", 255),
        ("top_k_experts", 129),
        ("layer_types", ["full_attention"]),
        ("attention_bias", True),
        ("hidden_activation", "silu"),
        ("rms_norm_eps", float("nan")),
    ],
)
def test_malformed_architecture_rejected(field, value):
    raw, generation = source()
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw, generation)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_denoising_steps", 0),
        ("max_denoising_steps", True),
        ("t_min", 0),
        ("t_min", 0.9),
        ("t_max", float("inf")),
        ("stability_threshold", -1),
        ("confidence_threshold", float("nan")),
        ("eos_token_id", [262144]),
        ("sampler_config", {"_cls_name": "AutoregressiveSampler"}),
    ],
)
def test_invalid_denoising_settings_rejected(field, value):
    raw, generation = source()
    generation[field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(raw, generation)


def test_config_does_not_mutate_checkpoint_metadata():
    raw, generation = source()
    before = deepcopy(raw), deepcopy(generation)
    ModelConfig.from_dict(raw, generation)
    assert (raw, generation) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_logit_softcapping", 0),
        ("final_logit_softcapping", float("inf")),
        ("use_bidirectional_attention", "all"),
    ],
)
def test_reject_unmapped_text_semantics(field, value):
    from pathlib import Path
    import json

    root = json.loads((Path(__file__).parent / "data/config.json").read_text())
    root["text_config"][field] = value
    with pytest.raises(ValueError):
        ModelConfig.from_dict(root)
