# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-owned architecture and block-diffusion generation settings."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"DiffusionGemma {name} must be a positive integer")
    return value


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"DiffusionGemma {name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"DiffusionGemma {name} is outside its supported range")
    return float(value)


@dataclass(frozen=True)
class DiffusionSettings:
    canvas_length: int
    max_denoising_steps: int
    t_max: float
    t_min: float
    entropy_bound: float
    confidence_threshold: float
    stability_threshold: int
    eos_token_ids: tuple[int, ...]

    @classmethod
    def parse(cls, root, generation, vocab):
        sampler = generation.get("sampler_config") or {}
        if (
            not isinstance(sampler, dict)
            or sampler.get("_cls_name", "EntropyBoundSamplerConfig") != "EntropyBoundSamplerConfig"
        ):
            raise ValueError("DiffusionGemma requires the checkpoint entropy-bound sampler")
        canvas = _positive_int(root.get("canvas_length", 256), "canvas_length")
        steps = _positive_int(generation.get("max_denoising_steps", 48), "max_denoising_steps")
        high = _finite(generation.get("t_max", 0.8), "t_max", positive=True)
        low = _finite(generation.get("t_min", 0.4), "t_min", positive=True)
        if low > high:
            raise ValueError("DiffusionGemma temperature schedule must not increase")
        bound = _finite(sampler.get("entropy_bound", 0.1), "entropy_bound")
        confidence = _finite(generation.get("confidence_threshold", 0.005), "confidence_threshold")
        stability = generation.get("stability_threshold", 2)
        if type(stability) is not int or stability < 0:
            raise ValueError("DiffusionGemma stability_threshold must be a nonnegative integer")
        eos = generation.get("eos_token_id", root.get("eos_token_id", []))
        eos = [eos] if type(eos) is int else eos
        if not isinstance(eos, list) or any(type(i) is not int or not 0 <= i < vocab for i in eos):
            raise ValueError("DiffusionGemma EOS IDs must be valid vocabulary indices")
        return cls(
            canvas, steps, high, low, bound, confidence, stability, tuple(dict.fromkeys(eos))
        )


@dataclass(frozen=True)
class ModelConfig:
    raw: dict
    text: dict
    generation: DiffusionSettings

    @classmethod
    def from_dict(cls, root, generation=None):
        if not isinstance(root, dict) or root.get("model_type") not in {
            "diffusion_gemma",
            "diffusiongemma",
        }:
            raise ValueError(
                "Expected a DiffusionGemma checkpoint, not an autoregressive Gemma model"
            )
        text = dict(root.get("text_config") or root)
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "num_global_key_value_heads",
            "head_dim",
            "global_head_dim",
            "intermediate_size",
            "vocab_size",
            "num_experts",
            "top_k_experts",
            "moe_intermediate_size",
            "max_position_embeddings",
            "sliding_window",
        ):
            _positive_int(text.get(key), key)
        if text["top_k_experts"] > text["num_experts"]:
            raise ValueError("DiffusionGemma selected expert count exceeds the expert bank")
        kinds = text.get("layer_types")
        if (
            not isinstance(kinds, list)
            or len(kinds) != text["num_hidden_layers"]
            or not set(kinds) <= {"sliding_attention", "full_attention"}
        ):
            raise ValueError("DiffusionGemma attention schedule must identify every layer")
        for kind in set(kinds):
            full = kind == "full_attention"
            dim = text["global_head_dim" if full else "head_dim"]
            kv = text["num_global_key_value_heads" if full else "num_key_value_heads"]
            if dim % 2 or text["num_attention_heads"] % kv:
                raise ValueError("DiffusionGemma attention geometry is inconsistent")
            rope = text.get("rope_parameters", {}).get(kind, {})
            if rope.get("rope_type", "default") not in {"default", "proportional"}:
                raise ValueError("Unsupported DiffusionGemma rotary embedding type")
            _finite(rope.get("rope_theta"), "rope_theta", positive=True)
            _finite(rope.get("factor", 1.0), "rope factor", positive=True)
            proportion = _finite(
                rope.get("partial_rotary_factor", 1.0), "partial_rotary_factor", positive=True
            )
            if proportion > 1 or int(dim * proportion) < 2:
                raise ValueError("DiffusionGemma rotary embedding proportion is invalid")
        _finite(text.get("rms_norm_eps"), "rms_norm_eps", positive=True)
        if text.get("final_logit_softcapping") is not None:
            _finite(text["final_logit_softcapping"], "final_logit_softcapping", positive=True)
        if text.get("use_bidirectional_attention") not in {None, "vision"}:
            raise ValueError("DiffusionGemma encoder requires causal text attention")
        if text.get("hidden_activation") != "gelu_pytorch_tanh" or text.get(
            "attention_bias", False
        ):
            raise ValueError("DiffusionGemma requires unbiased attention and checkpoint GEGLU")
        if not root.get("tie_word_embeddings", text.get("tie_word_embeddings", True)):
            raise ValueError("DiffusionGemma requires its shared input/output embedding")
        if text.get("num_kv_shared_layers", 0):
            raise ValueError(
                "DiffusionGemma does not share attention caches between decoder layers"
            )
        # Full-attention K=V and the routed branch are architecture semantics,
        # not GPU- or TensorRT-dependent implementation choices.
        text["attention_k_eq_v"] = True
        text["enable_moe_block"] = True
        text["self_conditioning_size"] = _positive_int(
            text.get("self_conditioning_size", text["intermediate_size"]), "self_conditioning_size"
        )
        return cls(
            dict(root), text, DiffusionSettings.parse(root, generation or {}, text["vocab_size"])
        )

    @classmethod
    def from_directory(cls, directory):
        directory = Path(directory)
        generation = directory / "generation_config.json"
        return cls.from_dict(
            json.loads((directory / "config.json").read_text()),
            json.loads(generation.read_text()) if generation.is_file() else {},
        )

    def attention_geometry(self, layer):
        if type(layer) is not int or not 0 <= layer < self.text["num_hidden_layers"]:
            raise ValueError("DiffusionGemma attention layer index is out of range")
        full = self.text["layer_types"][layer] == "full_attention"
        return (
            self.text["global_head_dim" if full else "head_dim"],
            self.text["num_global_key_value_heads" if full else "num_key_value_heads"],
        )
