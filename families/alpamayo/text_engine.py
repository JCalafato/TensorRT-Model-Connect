# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native VLM engine for visual prefill and incremental reasoning tokens."""

import numpy as np
import tensorrt as trt

from .text import TextGraph


def build_text_engine(
    config,
    weights,
    precision,
    *,
    max_batch,
    input_capacity,
    prefix_capacity,
    workspace_bytes,
    last_logits_only=True,
):
    """Build one decoder plan; empty KV inputs represent the initial prefill.

    The engine returns compact growing KV for each layer, and normally only
    the final token's logits. The family runtime owns tokenization, media
    placement, causal masking, rotary positions, sampling, and total-capacity
    admission. This component is not the complete trajectory task.
    """
    for name, value in (
        ("max_batch", max_batch),
        ("input_capacity", input_capacity),
        ("prefix_capacity", prefix_capacity),
        ("workspace_bytes", workspace_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    text = config.text
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = TextGraph(network, weights, text, config.vocab_size, precision)
    inputs = {}

    def add(name, dtype, shape, sequence_axis):
        value = network.add_input(name, dtype, shape)
        value.set_dimension_name(0, "batch")
        value.set_dimension_name(
            sequence_axis, "past_tokens" if name.startswith("past_") else "input_tokens"
        )
        inputs[name] = value
        return value

    tokens = add("tokens", trt.int32, (-1, -1), 1)
    visual = add("visual", graph.dtype, (-1, -1, text.hidden_size), 1)
    # MC exposes integer tensor storage; keep Boolean selection inside the graph.
    visual_mask = graph.cast(add("visual_mask", trt.int32, (-1, -1, 1), 1), trt.bool)
    deepstack = [
        add(f"deepstack_{i}", graph.dtype, (-1, -1, text.hidden_size), 1)
        for i in range(len(config.vision.deepstack_visual_indexes))
    ]
    mask = add("mask", trt.float32, (-1, 1, -1, -1), 2)
    cosine = add("cosine", graph.dtype, (-1, 1, -1, text.head_dim), 2)
    sine = add("sine", graph.dtype, (-1, 1, -1, text.head_dim), 2)
    kv_shape = (-1, text.num_key_value_heads, -1, text.head_dim)
    keys = [add(f"past_key_{i}", graph.dtype, kv_shape, 2) for i in range(text.num_hidden_layers)]
    values = [
        add(f"past_value_{i}", graph.dtype, kv_shape, 2) for i in range(text.num_hidden_layers)
    ]

    def dimension(value, axis):
        shape = network.add_shape(value).get_output(0)
        return network.add_gather(shape, graph.const(np.array([axis], np.int32)), 0).get_output(0)

    total = graph.op(dimension(tokens, 1), dimension(keys[0], 2), trt.ElementWiseOperation.SUM)
    same = graph.op(total, dimension(mask, 3), trt.ElementWiseOperation.EQUAL)
    network.add_assertion(same, "Alpamayo causal mask must span past and current tokens")
    logits, keys, values = graph.forward(
        tokens,
        visual,
        visual_mask,
        deepstack,
        keys,
        values,
        mask,
        cosine,
        sine,
        last_logits_only=last_logits_only,
    )
    outputs = [("logits", logits)]
    outputs += [
        (f"{kind}_{i}", value)
        for kind, tensors in (("key", keys), ("value", values))
        for i, value in enumerate(tensors)
    ]
    for name, value in outputs:
        value.name = name
        network.mark_output(value)
    profile = builder.create_optimization_profile()
    for name, value in inputs.items():
        shapes = []
        for batch, length, past in (
            (1, 1, 0),
            (1, min(128, input_capacity), 0),
            (max_batch, input_capacity, prefix_capacity),
        ):
            shape = list(value.shape)
            shape[0] = batch
            if name.startswith("past_"):
                shape[2] = past
            elif name == "mask":
                shape[2:] = (length, past + length)
            elif name in ("cosine", "sine"):
                shape[2] = length
            else:
                shape[1] = length
            shapes.append(tuple(shape))
        profile.set_shape(name, *shapes)
        if [tuple(shape) for shape in profile.get_shape(name)] != shapes:
            raise ValueError(f"Invalid VLM-engine optimization profile for {name}")
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    options.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Alpamayo VLM decoder")
    return plan
