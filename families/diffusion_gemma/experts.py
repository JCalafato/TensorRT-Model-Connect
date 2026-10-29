# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routed GEGLU branch; execution is entirely owned by TensorRT operators."""

import numpy as np
import tensorrt as trt


def add_experts(graph, residual, layer):
    n, raw, op = graph.net, graph.raw, trt.ElementWiseOperation
    prefix = f"model.decoder.layers.{layer}"
    hidden, width = raw["hidden_size"], raw["moe_intermediate_size"]
    count, selected = raw["num_experts"], raw["top_k_experts"]
    router = graph.norm(residual)
    router = graph.op(
        router, graph.reshape(graph.weight(prefix + ".router.scale"), (1, hidden)), op.PROD
    )
    router = graph.scale(router, hidden**-0.5)
    logits = graph.cast(graph.linear(router, prefix + ".router.proj"), trt.float32)
    softmax = n.add_softmax(logits)
    softmax.axes = 1 << 1
    top = n.add_topk(softmax.get_output(0), trt.TopKOperation.MAX, selected, 1 << 1)
    weights, indices = top.get_output(0), top.get_output(1)
    total = n.add_reduce(weights, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    weights = graph.op(weights, total, op.DIV)
    per_expert = graph.cast(graph.weight(prefix + ".router.per_expert_scale"), trt.float32)
    weights = graph.op(weights, n.add_gather(per_expert, indices, 0).get_output(0), op.PROD)
    normalized = graph.norm(residual, prefix + ".pre_feedforward_layernorm_2")
    loop = n.add_loop()
    loop.add_trip_limit(
        graph.reshape(graph.const(np.array([count], np.int32)), ()), trt.TripLimit.COUNT
    )
    expert_id = loop.add_iterator(
        graph.const(np.arange(count, dtype=np.int32)), 0, False
    ).get_output(0)
    if graph.weights.has(prefix + ".experts.gate_up_proj"):
        gate_up = loop.add_iterator(
            graph.weight(prefix + ".experts.gate_up_proj"), 0, False
        ).get_output(0)
        down = loop.add_iterator(graph.weight(prefix + ".experts.down_proj"), 0, False).get_output(
            0
        )
        if gate_up.shape != (2 * width, hidden) or down.shape != (hidden, width):
            raise ValueError("DiffusionGemma expert bank has inconsistent geometry")
        projected = graph.matmul(normalized, gate_up, transpose=True)
        gate = n.add_gather(projected, graph.const(np.arange(width, dtype=np.int32)), 1).get_output(
            0
        )
        up = n.add_gather(
            projected, graph.const(np.arange(width, width * 2, dtype=np.int32)), 1
        ).get_output(0)
        product = graph.op(graph.gelu(gate), up, op.PROD)
        result = graph.matmul(product, down, transpose=True)
    else:
        from .quantization import PreparedExperts, validate_prepared_source

        if not getattr(graph, "_prepared_source_validated", False):
            validate_prepared_source(graph.config, graph.weights)
            graph._prepared_source_validated = True
        prepared = PreparedExperts(graph, prefix, loop)
        # Do not assume the independently calibrated gate/up scales are equal.
        gate = prepared.linear(normalized, "gate_proj")
        up = prepared.linear(normalized, "up_proj")
        product = graph.op(graph.gelu(gate), up, op.PROD)
        result = prepared.linear(product, "down_proj")
    chosen = graph.op(graph.reshape(expert_id, (1, 1)), indices, op.EQUAL)
    chosen_weights = n.add_select(
        chosen, weights, graph.const(np.zeros((1, 1), np.float32))
    ).get_output(0)
    coefficient = n.add_reduce(chosen_weights, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    weighted = graph.op(graph.cast(result, trt.float32), coefficient, op.PROD)
    # Selection is independent of the learned per-expert scale's sign.
    active = n.add_reduce(
        graph.cast(chosen, trt.int32), trt.ReduceOperation.MAX, 1 << 1, True
    ).get_output(0)
    active = graph.cast(active, trt.bool)
    weighted = n.add_select(active, weighted, graph.const(np.zeros((1, 1), np.float32))).get_output(
        0
    )
    # The reference index_add casts each weighted expert result to compute
    # dtype, then accumulates experts in ascending source order.
    total = loop.add_recurrence(graph.scale(normalized, 0.0))
    total.set_input(1, graph.op(total.get_output(0), graph.cast(weighted, graph.dtype), op.SUM))
    result = loop.add_loop_output(total.get_output(0), trt.LoopOutput.LAST_VALUE, 0).get_output(0)
    return graph.norm(result, prefix + ".post_feedforward_layernorm_2")
