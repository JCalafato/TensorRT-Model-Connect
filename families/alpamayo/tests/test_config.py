# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architectural contracts; these synthetic cases are not model qualification."""

import json

import pytest

from families.alpamayo.config import ModelConfig
from families.alpamayo.support import describe


def configs():
    text = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        hidden_act="silu",
        rope_scaling=dict(rope_type="default", mrope_section=[2, 1, 1], mrope_interleaved=True),
    )
    vision = dict(
        depth=3,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=16,
        deepstack_visual_indexes=[0, 2],
        hidden_act="gelu_pytorch_tanh",
    )
    root = dict(
        model_type="alpamayo_r1",
        vocab_size=32,
        traj_token_start_idx=10,
        traj_vocab_size=8,
        min_pixels=64,
        max_pixels=256,
        traj_token_ids=dict(
            history=21, future=22, history_start=23, history_end=24, future_start=25, future_end=26
        ),
        expert_cfg=dict(hidden_size=8, intermediate_size=12, num_attention_heads=1),
        action_space_cfg=dict(
            _target_="alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace",
            n_waypoints=4,
            dt=0.1,
        ),
        expert_non_causal_attention=True,
        diffusion_cfg=dict(int_method="euler"),
        tokens_per_history_traj=6,
        tokens_per_future_traj=8,
        action_in_proj_cfg=dict(
            hidden_size=8, num_enc_layers=2, num_fourier_feats=2, max_freq=100.0
        ),
    )
    return root, dict(model_type="qwen3_vl", text_config=text, vision_config=vision)


def test_distinct_expert_width_preserves_shared_kv():
    model = ModelConfig.parse(*configs())
    assert model.text.hidden_size == 16
    assert model.expert.hidden_size == 8
    assert model.text.cache_layout == model.expert.cache_layout == (2, 1, 8)
    assert model.vocab_size == 32
    assert model.action.future_tokens == 2 * model.action.num_waypoints
    assert model.vision.deepstack_visual_indexes == (0, 2)


@pytest.mark.parametrize(
    "change",
    [
        lambda root, vlm: root["expert_cfg"].update(num_hidden_layers=3),
        lambda root, vlm: vlm["vision_config"].update(out_hidden_size=8),
        lambda root, vlm: root.update(tokens_per_future_traj=6),
        lambda root, vlm: root["action_in_proj_cfg"].update(num_fourier_feats=3),
        lambda root, vlm: root.update(model_type="alpamayo_v2"),
        lambda root, vlm: root.update(quantization_config={"quant_method": "unknown"}),
        lambda root, vlm: root.update(min_pixels=512),
        lambda root, vlm: root["traj_token_ids"].update(future=21),
        lambda root, vlm: root["action_space_cfg"].update(_target_="untrusted.Factory"),
        lambda root, vlm: vlm["text_config"]["rope_scaling"].update(mrope_section=[1, 1, 1]),
    ],
)
def test_reject_inconsistent_architecture(change):
    root, vlm = configs()
    change(root, vlm)
    with pytest.raises(ValueError):
        ModelConfig.parse(root, vlm)


def test_local_backbone_metadata_without_network(tmp_path):
    root, vlm = configs()
    (tmp_path / "config.json").write_text(json.dumps(root))
    with pytest.raises(FileNotFoundError, match="base VLM metadata"):
        ModelConfig.from_directory(tmp_path)
    (tmp_path / "base_vlm").mkdir()
    (tmp_path / "base_vlm/config.json").write_text(json.dumps(vlm))
    assert ModelConfig.from_directory(tmp_path) == ModelConfig.parse(root, vlm)


def test_embedded_backbone_metadata(tmp_path):
    root, vlm = configs()
    root["vlm_config"] = vlm
    (tmp_path / "config.json").write_text(json.dumps(root))
    assert ModelConfig.from_directory(tmp_path) == ModelConfig.parse(root, vlm)


def test_complete_driving_task_has_distinct_owner():
    from tensorrt_model_connect.model_support import ModelMetadata

    metadata = ModelMetadata({"model_type": "alpamayo_r1", "architectures": ["AlpamayoR1"]}, {})
    support = describe(metadata)
    assert support.tasks == ("structured_decision",)
    assert support.default_precision == "fp16"
    assert (
        describe(
            ModelMetadata(
                {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}, {}
            )
        )
        is None
    )
    assert describe(ModelMetadata({"model_type": "alpamayo_r1", "architectures": []}, {})) is None
    assert (
        describe(
            ModelMetadata({**metadata.config, "quantization_config": {"quant_method": "awq"}}, {})
        )
        is None
    )
