# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned native checkpoint to bundle construction."""

from dataclasses import asdict
from pathlib import Path

from .config import ModelConfig
from .edge_llm.dispatch import build as dispatch_build
from .engine import build_text_engine, build_vision_engine
from .weights import Checkpoint


def build(request, writer):
    if request.task not in {"text_generation", "vision_language_generation", "images_text_to_text"}:
        raise ValueError("DiffusionGemma supports text or vision-language generation")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
    ):
        raise ValueError("DiffusionGemma requires batch one and TP/CP one")
    if request.dynamic_kv_cache:
        raise NotImplementedError("diffusion_gemma does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise ValueError(
            "DiffusionGemma uses a fixed prefix cache and block diffusion, not speculative decoding"
        )
    if request.graph_transform is not None or request.fp32_layers:
        raise ValueError(
            "DiffusionGemma does not expose graph transforms or per-layer precision overrides"
        )
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError(
            "DiffusionGemma image geometry is checkpoint-owned; build overrides are unsupported"
        )
    if request.quantization not in {None, "none", "nvfp4"}:
        raise ValueError(
            "DiffusionGemma preserves prepared NVFP4; it does not calibrate new quantization"
        )

    def native(request, writer):
        model_dir = Path(request.model_dir)
        config = ModelConfig.from_directory(model_dir)
        tokenizer = model_dir / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError("DiffusionGemma requires checkpoint tokenizer.json")
        capacity = request.max_sequence_length or min(4096, config.text["max_position_embeddings"])
        conditioned = request.task in {"vision_language_generation", "images_text_to_text"}
        image = None
        if conditioned:
            raw = config.raw["vision_config"]
            image = {
                "patch_size": raw["patch_size"],
                "pooling_kernel": raw["pooling_kernel_size"],
                "soft_tokens": config.raw.get(
                    "vision_soft_tokens_per_image", raw.get("default_output_length", 280)
                ),
                "hidden_size": config.text["hidden_size"],
                "image_token": config.raw["image_token_id"],
                "begin_token": config.raw["boi_token_id"],
                "end_token": config.raw["eoi_token_id"],
            }
            if any(
                type(image[k]) is not int or not 0 <= image[k] < config.text["vocab_size"]
                for k in ("image_token", "begin_token", "end_token")
            ):
                raise ValueError("DiffusionGemma image markers must be valid vocabulary IDs")
            if image["soft_tokens"] * image["pooling_kernel"] > raw["position_embedding_size"]:
                raise ValueError("DiffusionGemma image budget exceeds its position embedding table")
        with Checkpoint(model_dir) as weights:
            # Exact prepared-source validation is owned by the quantized graph;
            # requests never trigger implicit calibration or repacking campaigns.
            if request.quantization == "nvfp4" and not config.raw.get("quantization_config"):
                raise ValueError(
                    "DiffusionGemma --quantization nvfp4 requires a prepared source checkpoint"
                )
            plan = build_text_engine(
                config,
                weights,
                capacity,
                precision=request.precision,
                conditioned=conditioned,
                verbose=request.verbose,
            )
            vision_plan = (
                build_vision_engine(
                    config, weights, precision=request.precision, verbose=request.verbose
                )
                if conditioned
                else None
            )
        writer.set_header(family="diffusion_gemma", task=request.task, backend=request.backend)
        writer.add_bytes("engine.plan", plan)
        if vision_plan is not None:
            writer.add_bytes("vision.plan", vision_plan)
            writer.add_json("image_config.json", image)
        writer.add_json(
            "runtime.json",
            {
                "schema_version": 1,
                "hidden_size": config.text["hidden_size"],
                "vocab_size": config.text["vocab_size"],
                "bos_token_id": config.raw.get("bos_token_id", config.text.get("bos_token_id", 2)),
                "precision": request.precision,
                "max_sequence_length": capacity,
                "cache_widths": [
                    dimension * heads
                    for dimension, heads in (
                        config.attention_geometry(i)
                        for i in range(config.text["num_hidden_layers"])
                    )
                ],
                "generation": asdict(config.generation),
            },
        )
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ):
            path = model_dir / name
            if path.is_file():
                writer.add_bytes(name, path.read_bytes())

    return dispatch_build(request, writer, native)
