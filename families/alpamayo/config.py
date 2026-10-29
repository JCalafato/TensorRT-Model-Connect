# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo's model parameters, independent of task APIs and GPU targets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _positive(raw: dict, key: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_float(raw: dict, key: str) -> float:
    value = float(raw[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be positive and finite")
    return value


@dataclass(frozen=True)
class DecoderConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool

    @classmethod
    def parse(cls, raw: dict) -> DecoderConfig:
        sizes = {
            name: _positive(raw, name)
            for name in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            )
        }
        if sizes["hidden_size"] != sizes["num_attention_heads"] * sizes["head_dim"]:
            raise ValueError("Decoder hidden size must match query-head dimensions")
        if sizes["num_attention_heads"] % sizes["num_key_value_heads"]:
            raise ValueError("Query heads must be divisible by KV heads")
        if raw.get("hidden_act") != "silu" or raw.get("attention_bias", False):
            raise ValueError("Alpamayo requires unbiased attention and SiLU decoder MLPs")
        rope = raw["rope_scaling"]
        sections = tuple(rope["mrope_section"])
        if (
            len(sections) != 3
            or any(type(n) is not int or n <= 0 for n in sections)
            or sum(sections) * 2 != sizes["head_dim"]
        ):
            raise ValueError("Three mRoPE sections must span the rotary head")
        if rope.get("rope_type") != "default":
            raise ValueError("Unsupported Alpamayo rotary scaling")
        return cls(
            **sizes,
            rms_norm_eps=_positive_float(raw, "rms_norm_eps"),
            rope_theta=_positive_float(raw, "rope_theta"),
            mrope_section=sections,
            mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
        )

    @property
    def cache_layout(self) -> tuple[int, int, int]:
        return self.num_hidden_layers, self.num_key_value_heads, self.head_dim


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...]

    @classmethod
    def parse(cls, raw: dict) -> VisionConfig:
        sizes = {
            name: _positive(raw, name)
            for name in cls.__dataclass_fields__
            if name != "deepstack_visual_indexes"
        }
        if sizes["hidden_size"] % sizes["num_heads"]:
            raise ValueError("Vision width must be divisible by the attention heads")
        if raw.get("hidden_act") != "gelu_pytorch_tanh":
            raise ValueError("Alpamayo vision requires tanh-approximate GELU")
        count = sizes["num_position_embeddings"]
        if math.isqrt(count) ** 2 != count:
            raise ValueError("Vision positions must form a square interpolation grid")
        levels = tuple(raw["deepstack_visual_indexes"])
        if (
            not levels
            or any(type(n) is not int or not 0 <= n < sizes["depth"] for n in levels)
            or tuple(sorted(set(levels))) != levels
        ):
            raise ValueError("DeepStack indexes must be distinct ordered vision layers")
        return cls(**sizes, deepstack_visual_indexes=levels)


@dataclass(frozen=True)
class ActionConfig:
    num_waypoints: int
    dt: float
    history_tokens: int
    future_tokens: int
    projection_hidden_size: int
    projection_layers: int
    fourier_features: int
    max_frequency: float

    @classmethod
    def parse(cls, raw: dict) -> ActionConfig:
        space = raw["action_space_cfg"]
        if space.get("_target_") != "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace":
            raise ValueError("Unsupported Alpamayo action space")
        if not raw.get("expert_non_causal_attention", False):
            raise ValueError("Alpamayo action queries require non-causal expert attention")
        if raw["diffusion_cfg"].get("int_method") != "euler":
            raise ValueError("Unsupported Alpamayo flow integration method")
        projection = raw["action_in_proj_cfg"]
        result = cls(
            num_waypoints=_positive(space, "n_waypoints"),
            dt=_positive_float(space, "dt"),
            history_tokens=_positive(raw, "tokens_per_history_traj"),
            future_tokens=_positive(raw, "tokens_per_future_traj"),
            projection_hidden_size=_positive(projection, "hidden_size"),
            projection_layers=_positive(projection, "num_enc_layers"),
            fourier_features=_positive(projection, "num_fourier_feats"),
            max_frequency=_positive_float(projection, "max_freq"),
        )
        if result.fourier_features % 2:
            raise ValueError("Action Fourier features must split evenly into sine and cosine")
        if result.history_tokens % 3 or result.future_tokens != 2 * result.num_waypoints:
            raise ValueError("History XYZ and future two-channel action dimensions disagree")
        return result


@dataclass(frozen=True)
class ModelConfig:
    text: DecoderConfig
    vision: VisionConfig
    expert: DecoderConfig
    action: ActionConfig
    vocab_size: int
    trajectory_vocab_start: int
    trajectory_vocab_size: int
    trajectory_token_ids: dict[str, int]
    min_pixels: int
    max_pixels: int

    @classmethod
    def parse(cls, root: dict, vlm: dict) -> ModelConfig:
        if root.get("model_type") != "alpamayo_r1" or vlm.get("model_type") != "qwen3_vl":
            raise ValueError("Expected Alpamayo-R1 and its dense Qwen3-VL backbone metadata")
        if root.get("quantization_config") or vlm.get("quantization_config"):
            raise ValueError("Quantized Alpamayo checkpoints are not documented")
        text = DecoderConfig.parse(vlm["text_config"])
        expert = DecoderConfig.parse({**vlm["text_config"], **root["expert_cfg"]})
        vision = VisionConfig.parse(vlm["vision_config"])
        if text.cache_layout != expert.cache_layout:
            raise ValueError("Expert and VLM must share the same prefix KV layout")
        if vision.out_hidden_size != text.hidden_size:
            raise ValueError("Vision outputs must match the VLM hidden size")
        result = cls(
            text=text,
            vision=vision,
            expert=expert,
            action=ActionConfig.parse(root),
            vocab_size=_positive(root, "vocab_size"),
            trajectory_vocab_start=_positive(root, "traj_token_start_idx"),
            trajectory_vocab_size=_positive(root, "traj_vocab_size"),
            trajectory_token_ids=dict(root["traj_token_ids"]),
            min_pixels=_positive(root, "min_pixels"),
            max_pixels=_positive(root, "max_pixels"),
        )
        if result.min_pixels > result.max_pixels:
            raise ValueError("Image pixel bounds are reversed")
        if result.trajectory_vocab_start + result.trajectory_vocab_size >= result.vocab_size:
            raise ValueError("Trajectory vocabulary leaves no room for special tokens")
        required = {
            "history",
            "future",
            "history_start",
            "history_end",
            "future_start",
            "future_end",
        }
        ids = result.trajectory_token_ids
        if set(ids) != required or any(
            type(v) is not int or not 0 <= v < result.vocab_size for v in ids.values()
        ):
            raise ValueError("Invalid trajectory special-token IDs")
        if len(set(ids.values())) != len(ids):
            raise ValueError("Trajectory special-token IDs must be distinct")
        return result

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> ModelConfig:
        directory = Path(model_dir)
        root = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        embedded = root.get("vlm_config") or root.get("vlm_cfg")
        if isinstance(embedded, dict):
            return cls.parse(root, embedded)
        for name in ("vlm_config.json", "config_vlm.json", "base_vlm/config.json"):
            path = directory / name
            if path.is_file():
                return cls.parse(root, json.loads(path.read_text(encoding="utf-8")))
        raise FileNotFoundError(
            "Alpamayo requires its base VLM metadata, such as base_vlm/config.json"
        )
