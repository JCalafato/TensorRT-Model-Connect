# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native combined action flow step, consuming the VLM's immutable prefix KV."""

import tensorrt as trt

from .action import ActionGraph
from .expert import ExpertGraph


def build_action_engine(config, weights, precision, *, max_batch, prefix_capacity, workspace_bytes):
    """Build action projection -> non-causal expert -> normalized flow velocity.

    All action tokens attend to the complete action sequence and the valid VLM
    prefix. The caller supplies an additive mask for post-EOS prefix padding,
    and rotary tables for each sample's trajectory-start position plus its VLM
    rotary delta. No action KV is written back into the prefix.

    This engine alone does not generate reasoning or integrate physical poses.
    """
    for name, value in (
        ("max_batch", max_batch),
        ("prefix_capacity", prefix_capacity),
        ("workspace_bytes", workspace_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    action = ActionGraph(network, weights, config.action, config.expert.hidden_size, precision)
    expert = ExpertGraph(network, weights, config.expert, precision)
    waypoints, head_dim = config.action.num_waypoints, config.expert.head_dim
    kv_heads = config.expert.num_key_value_heads
    inputs = {}

    def add(name, dtype, shape):
        value = network.add_input(name, dtype, shape)
        value.set_dimension_name(0, "batch")
        inputs[name] = value
        return value

    noise = add("actions", trt.float32, (-1, waypoints, 2))
    timestep = add("timesteps", trt.float32, (-1, 1, 1))
    mask = add("prefix_mask", trt.float32, (-1, 1, waypoints, -1))
    cosine = add("cosine", trt.float32, (-1, 1, waypoints, head_dim))
    sine = add("sine", trt.float32, (-1, 1, waypoints, head_dim))
    keys, values = [], []
    for index in range(config.expert.num_hidden_layers):
        key = add(f"prefix_key_{index}", expert.dtype, (-1, kv_heads, -1, head_dim))
        value = add(f"prefix_value_{index}", expert.dtype, (-1, kv_heads, -1, head_dim))
        key.set_dimension_name(2, "prefix_length")
        value.set_dimension_name(2, "prefix_length")
        keys.append(key)
        values.append(value)
    embedded = action.input_projection(noise, timestep)
    hidden = expert.forward(embedded, keys, values, mask, cosine, sine)
    velocity = action.cast(action.output_projection(hidden), trt.float32)
    velocity.name = "velocity"
    network.mark_output(velocity)

    profile = builder.create_optimization_profile()
    for name, value in inputs.items():
        shapes = []
        for batch, prefix in ((1, 1), (1, min(256, prefix_capacity)), (max_batch, prefix_capacity)):
            shape = list(value.shape)
            shape[0] = batch
            if name.startswith(("prefix_key_", "prefix_value_")):
                shape[2] = prefix
            elif name == "prefix_mask":
                shape[-1] = prefix + waypoints
            shapes.append(tuple(shape))
        profile.set_shape(name, *shapes)
        if [tuple(shape) for shape in profile.get_shape(name)] != shapes:
            raise ValueError(f"Invalid action-engine optimization profile for {name}")
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    options.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Alpamayo action expert")
    return plan
