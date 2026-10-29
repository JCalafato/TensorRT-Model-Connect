# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native text engine construction; no custom kernels or platform selection."""

import tensorrt as trt

from .text import TextGraph


def build_text_engine(
    config,
    weights,
    capacity,
    *,
    precision,
    conditioned=False,
    workspace_bytes=None,
    verbose=False,
):
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("DiffusionGemma native compute precision must be fp32, fp16 or bf16")
    if (
        type(capacity) is not int
        or not config.generation.canvas_length <= capacity <= config.text["max_position_embeddings"]
    ):
        raise ValueError(
            "DiffusionGemma capacity must fit a canvas and stay within checkpoint positions"
        )
    if workspace_bytes is not None and (type(workspace_bytes) is not int or workspace_bytes <= 0):
        raise ValueError("DiffusionGemma builder workspace must be a positive byte count")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = TextGraph(network, config, weights, precision)
    graph.build(capacity, conditioned=conditioned)
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    # TensorRT owns tactic selection and the default workspace budget. A small
    # family-imposed cap can reject valid full-model tactics despite ample GPU
    # memory. Honor an explicit caller cap, otherwise retain the SDK default.
    if workspace_bytes is not None:
        options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    profile = builder.create_optimization_profile()
    canvas = config.generation.canvas_length
    preferred = min(canvas, capacity)
    for name in ("token_id", "position_id"):
        profile.set_shape(name, (1,), (preferred,), (capacity,))
    profile.set_shape(
        "previous_soft_embeddings",
        (1, config.text["hidden_size"]),
        (preferred, config.text["hidden_size"]),
        (capacity, config.text["hidden_size"]),
    )
    # Encoder selects one discarded vocabulary row; denoising selects a full
    # canvas. A prompt-length vocabulary output is never needed by the runtime.
    profile.set_shape("logits_indices", (1,), (canvas,), (canvas,))
    if conditioned:
        profile.set_shape("media_blocks", (1,), (preferred,), (capacity,))
        profile.set_shape(
            "media_features",
            (1, config.text["hidden_size"]),
            (preferred, config.text["hidden_size"]),
            (capacity, config.text["hidden_size"]),
        )
    options.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError(
            "DiffusionGemma native text build failed; see the TensorRT diagnostic output"
        )
    # graph holds constant storage through serialization; weights need not be
    # reopened or copied into a second full-size checkpoint representation.
    return bytes(plan)


def build_vision_engine(config, weights, *, precision, workspace_bytes=None, verbose=False):
    """Serialize one image's fixed-capacity patch encoder and language projection."""
    from .vision import VisionGraph

    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("DiffusionGemma vision precision must be fp32, fp16 or bf16")
    if workspace_bytes is not None and (type(workspace_bytes) is not int or workspace_bytes <= 0):
        raise ValueError("DiffusionGemma builder workspace must be a positive byte count")
    raw = config.raw.get("vision_config")
    if not isinstance(raw, dict) or raw.get("model_type") != "gemma4_vision":
        raise ValueError("DiffusionGemma requires its Gemma4 vision checkpoint")
    soft_tokens = config.raw.get(
        "vision_soft_tokens_per_image", raw.get("default_output_length", 280)
    )
    if type(soft_tokens) is not int or soft_tokens <= 0:
        raise ValueError("DiffusionGemma vision token budget must be positive")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = VisionGraph(network, raw, weights, precision, soft_tokens)
    graph.build()
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    # TensorRT owns tactic selection and the default workspace budget. A small
    # family-imposed cap can reject valid full-model tactics despite ample GPU
    # memory. Honor an explicit caller cap, otherwise retain the SDK default.
    if workspace_bytes is not None:
        options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError(
            "DiffusionGemma native vision build failed; see TensorRT diagnostic output"
        )
    return bytes(plan)
