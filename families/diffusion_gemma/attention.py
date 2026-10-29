# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider grouped attention and full-width proportional rotary embeddings."""

import numpy as np
import tensorrt as trt


def rotary_tables(raw, kind, dimension, capacity):
    params = raw["rope_parameters"][kind]
    fraction = (
        params.get("partial_rotary_factor", 1.0)
        if params.get("rope_type") == "proportional"
        else 1.0
    )
    count = int(dimension * fraction // 2)
    inverse = np.zeros(dimension // 2, np.float32)
    inverse[:count] = 1 / (
        params["rope_theta"] ** (np.arange(count, dtype=np.float32) * 2 / dimension)
    )
    inverse /= params.get("factor", 1.0)
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inverse[None, :]
    return np.cos(angles), np.sin(angles)


def _heads(graph, value, count, dimension):
    layer = graph.net.add_shuffle(value)
    layer.reshape_dims = (-1, count, dimension)
    layer.second_transpose = (1, 0, 2)
    return graph.reshape(layer.get_output(0), (1, count, -1, dimension))


def _rotate(graph, value, positions, count, dimension, tables):
    # Rotate half of the entire head; proportional RoPE sets unused angles to
    # zero rather than changing the half-rotation pairing.
    value = graph.reshape(value, (-1, count, dimension))
    cosine, sine = [graph.net.add_gather(table, positions, 0).get_output(0) for table in tables]
    cosine = graph.reshape(graph.concat([cosine, cosine], 1), (-1, 1, dimension))
    sine = graph.reshape(graph.concat([sine, sine], 1), (-1, 1, dimension))
    indices = graph.const(
        np.r_[np.arange(dimension // 2, dimension), np.arange(dimension // 2)].astype(np.int32)
    )
    rotated = graph.net.add_gather(value, indices, 2).get_output(0)
    signs = graph.const(
        np.r_[-np.ones(dimension // 2), np.ones(dimension // 2)]
        .astype(np.float32)
        .reshape(1, 1, -1),
        graph.dtype,
    )
    rotated = graph.op(rotated, signs, trt.ElementWiseOperation.PROD)

    # Eager reference rounds BOTH FP16 products before the addition. A single
    # compute-dtype fused multiply-add skips one rounding boundary. Explicit
    # accumulator/cast boundaries preserve the two products independently.
    def product(left, right):
        return graph.cast(
            graph.op(
                graph.cast(left, trt.float32),
                graph.cast(right, trt.float32),
                trt.ElementWiseOperation.PROD,
            ),
            graph.dtype,
        )

    result = graph.cast(
        graph.op(
            graph.cast(product(value, cosine), trt.float32),
            graph.cast(product(rotated, sine), trt.float32),
            trt.ElementWiseOperation.SUM,
        ),
        graph.dtype,
    )
    return graph.reshape(result, (-1, count * dimension))


def add_attention(graph, state, positions, mask, layer, tables, capacity):
    n, raw = graph.net, graph.raw
    dimension, kv_heads = graph.config.attention_geometry(layer)
    heads = raw["num_attention_heads"]
    prefix = f"model.decoder.layers.{layer}.self_attn"

    def normalized(value, count, name=None):
        return graph.reshape(
            graph.norm(graph.reshape(value, (-1, count, dimension)), name), (-1, count * dimension)
        )

    query = normalized(graph.linear(state, prefix + ".q_proj"), heads, prefix + ".q_norm")
    query = _rotate(graph, query, positions, heads, dimension, tables)
    key_raw = graph.linear(state, prefix + ".k_proj")
    value_raw = (
        key_raw
        if raw["layer_types"][layer] == "full_attention"
        else graph.linear(state, prefix + ".v_proj")
    )
    key = _rotate(
        graph,
        normalized(key_raw, kv_heads, prefix + ".k_norm"),
        positions,
        kv_heads,
        dimension,
        tables,
    )
    value = normalized(value_raw, kv_heads)
    combined = []
    for label, current in (("k", key), ("v", value)):
        current.name = f"present_{label}_{layer}"
        n.mark_output(current)
        cached = n.add_input(
            f"cache_{label}_{layer}", graph.dtype, (capacity, kv_heads * dimension)
        )
        full = _heads(graph, graph.concat([cached, current], 0), kv_heads, dimension)
        copies = graph.const((np.arange(heads) // (heads // kv_heads)).astype(np.int32))
        combined.append(n.add_gather(full, copies, 1).get_output(0))
    query = _heads(graph, query, heads, dimension)
    scores = graph.matmul(query, combined[0], transpose=True)
    # The provider uses scale=1 and rounds the QK result to compute precision.
    scores = graph.op(graph.cast(scores, trt.float32), mask, trt.ElementWiseOperation.SUM)
    softmax = n.add_softmax(scores)
    softmax.axes = 1 << 3
    probabilities = graph.cast(softmax.get_output(0), graph.dtype)
    context = graph.matmul(probabilities, combined[1])
    transpose = n.add_shuffle(context)
    transpose.first_transpose = (0, 2, 1, 3)
    transpose.reshape_dims = (-1, heads * dimension)
    return graph.linear(transpose.get_output(0), prefix + ".o_proj")
