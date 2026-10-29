# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native visual engine with final and intermediate language-width features."""

import tensorrt as trt

from .vision import VisionGraph


def build_vision_engine(config, weights, precision, *, patch_capacity, workspace_bytes):
    unit = config.spatial_merge_size**2
    if type(patch_capacity) is not int or patch_capacity <= 0 or patch_capacity % unit:
        raise ValueError("Visual patch capacity must be a positive multiple of the merge unit")
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be a positive integer")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = VisionGraph(network, weights, config, precision)
    inputs = {}

    def add(name, dtype, shape, patch_axes):
        value = network.add_input(name, dtype, shape)
        for axis in patch_axes:
            value.set_dimension_name(axis, "patches")
        inputs[name] = value
        return value

    flat = config.in_channels * config.temporal_patch_size * config.patch_size**2
    pixels = add("pixels", trt.float32, (-1, flat), (0,))
    indices = add("position_indices", trt.int32, (4, -1), (1,))
    weights_input = add("position_weights", trt.float32, (4, -1), (1,))
    mask = add("mask", trt.float32, (1, -1, -1), (1, 2))
    head = config.hidden_size // config.num_heads
    cosine = add("cosine", trt.float32, (-1, 1, head), (0,))
    sine = add("sine", trt.float32, (-1, 1, head), (0,))
    final, deepstack = graph.forward(pixels, indices, weights_input, mask, cosine, sine)
    for name, value in [
        ("features", final),
        *[(f"deepstack_{i}", x) for i, x in enumerate(deepstack)],
    ]:
        value.name = name
        network.mark_output(value)
    profile = builder.create_optimization_profile()
    for name, value in inputs.items():
        shapes = [
            tuple(count if axis == -1 else axis for axis in value.shape)
            for count in (unit, min(256 * unit, patch_capacity), patch_capacity)
        ]
        profile.set_shape(name, *shapes)
        if [tuple(shape) for shape in profile.get_shape(name)] != shapes:
            raise ValueError(f"Invalid visual-engine profile for {name}")
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    options.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Alpamayo visual encoder")
    return plan
