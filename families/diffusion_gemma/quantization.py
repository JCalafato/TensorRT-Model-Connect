# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared ModelOpt NVFP4 math without a device-specific quantized kernel.

Keep source FP4 weights, FP8 block scales and global weight/input scales. The
fallback expresses rounding and decoding with ordinary TensorRT operations;
it neither re-calibrates nor replaces W4A4 with a weight-only computation.
"""

import ml_dtypes
import numpy as np
import tensorrt as trt


def _constant(graph, value, rank):
    return graph.const(np.full((1,) * rank, value, np.float32))


def _round_even(graph, value):
    """Round nonnegative finite values to integer, with ties to even."""
    n, op = graph.net, trt.ElementWiseOperation
    rank = len(value.shape)
    base = n.add_unary(value, trt.UnaryOperation.FLOOR).get_output(0)
    half = _constant(graph, 0.5, rank)
    fraction = graph.op(value, base, op.SUB)
    pairs = n.add_unary(graph.scale(base, 0.5), trt.UnaryOperation.FLOOR).get_output(0)
    odd = graph.op(
        graph.op(base, graph.scale(pairs, 2), op.SUB), _constant(graph, 0, rank), op.GREATER
    )
    increment = graph.op(
        graph.op(fraction, half, op.GREATER),
        graph.op(graph.op(fraction, half, op.EQUAL), odd, op.AND),
        op.OR,
    )
    return graph.op(base, graph.cast(increment, trt.float32), op.SUM)


def round_e4m3_scale(graph, value):
    """Positive E4M3FN round-to-nearest-even with finite saturation at448."""
    n, op = graph.net, trt.ElementWiseOperation
    rank = len(value.shape)
    value = graph.op(value, _constant(graph, 448, rank), op.MIN)
    safe = graph.op(value, _constant(graph, 2**-6, rank), op.MAX)
    exponent = n.add_unary(
        graph.scale(n.add_unary(safe, trt.UnaryOperation.LOG).get_output(0), np.log2(np.e)),
        trt.UnaryOperation.FLOOR,
    ).get_output(0)
    step = graph.op(
        _constant(graph, 2, rank), graph.op(exponent, _constant(graph, 3, rank), op.SUB), op.POW
    )
    units = _round_even(graph, graph.op(value, step, op.DIV))
    return graph.op(graph.op(units, step, op.PROD), _constant(graph, 448, rank), op.MIN)


def round_e2m1(graph, value):
    """Signed E2M1 nearest-even rounding, including unequal exponent spacing."""
    n, op = graph.net, trt.ElementWiseOperation
    rank = len(value.shape)
    absolute = n.add_unary(value, trt.UnaryOperation.ABS).get_output(0)
    code = graph.cast(graph.scale(absolute, 0.0), trt.int32)
    for index, midpoint in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        threshold = _constant(graph, midpoint, rank)
        higher = graph.op(absolute, threshold, op.GREATER)
        if (index + 1) % 2 == 0:
            higher = graph.op(higher, graph.op(absolute, threshold, op.EQUAL), op.OR)
        code = graph.op(code, graph.cast(higher, trt.int32), op.SUM)
    table = graph.const(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32))
    rounded = n.add_gather(table, code, 0).get_output(0)
    negative = graph.op(value, _constant(graph, 0, rank), op.LESS)
    return n.add_select(negative, graph.scale(rounded, -1.0), rounded).get_output(0)


def activation_qdq(graph, value, global_scale):
    """Dynamic block16 activation QDQ with the checkpoint's calibrated scale."""
    n, op = graph.net, trt.ElementWiseOperation
    columns = value.shape[-1]
    if columns <= 0 or columns % 16:
        raise ValueError("DiffusionGemma NVFP4 activation width must be a multiple of16")
    grouped = graph.reshape(graph.cast(value, trt.float32), (-1, columns // 16, 16))
    absolute = n.add_unary(grouped, trt.UnaryOperation.ABS).get_output(0)
    maximum = n.add_reduce(absolute, trt.ReduceOperation.MAX, 1 << 2, True).get_output(0)
    alpha = graph.reshape(global_scale, (1, 1, 1))
    block = round_e4m3_scale(graph, graph.op(graph.scale(maximum, 1 / 6), alpha, op.DIV))
    scale = graph.op(block, alpha, op.PROD)
    nonzero = graph.op(scale, _constant(graph, 0, 3), op.GREATER)
    divisor = n.add_select(nonzero, scale, _constant(graph, 1, 3)).get_output(0)
    normalized = graph.op(grouped, divisor, op.DIV)
    decoded = graph.op(round_e2m1(graph, normalized), scale, op.PROD)
    return graph.cast(graph.reshape(decoded, (-1, columns)), value.dtype)


def validate_prepared_source(config, weights):
    quant = config.raw.get("quantization_config") or {}
    expected = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    if (
        quant.get("quant_method") != "modelopt"
        or quant.get("quant_algo") != "NVFP4"
        or quant.get("config_groups")
        != {"group_0": {"weights": expected, "input_activations": expected, "targets": ["Linear"]}}
    ):
        raise ValueError(
            "DiffusionGemma prepared source requires ModelOpt NVFP4 W4A4 block16 metadata"
        )
    for name in weights.tensor_map:
        if name.endswith((".q_scale", ".k_scale", ".v_scale")):
            raise NotImplementedError(
                "DiffusionGemma calibrated attention/cache quantization is not mapped"
            )
        if (
            name.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))
            and ".experts." not in name
        ):
            raise NotImplementedError(
                "DiffusionGemma prepared quantization must be confined to experts"
            )


def _unpack_codes(graph, storage, count, bits):
    op = trt.ElementWiseOperation
    per_word = 32 // bits
    radix = 1 << bits
    divisors = graph.const(
        np.array([radix**i for i in range(per_word)], np.int32).reshape(1, per_word)
    )
    quotient = graph.op(graph.reshape(storage, (-1, 1)), divisors, op.FLOOR_DIV)
    base = graph.const(np.array([[radix]], np.int32))
    high = graph.op(quotient, base, op.FLOOR_DIV)
    codes = graph.op(quotient, graph.op(high, base, op.PROD), op.SUB)
    return graph.reshape(codes, (count,))


class PreparedExperts:
    """Iterate packed source experts before decoding, bounding expanded storage."""

    def __init__(self, graph, prefix, loop):
        self.graph, self.prefix, self.loop = graph, prefix, loop
        self.projections = {}

    def _scalar(self, name):
        value = self.graph.weights.floating(name).reshape(-1)
        if value.size != 1 or not np.isfinite(value[0]) or value[0] <= 0:
            raise ValueError(f"DiffusionGemma NVFP4 scalar must be finite and positive: {name}")
        return value[0]

    def projection(self, name):
        if name in self.projections:
            return self.projections[name]
        g, n, raw, op = self.graph, self.graph.net, self.graph.raw, trt.ElementWiseOperation
        count, hidden, width = raw["num_experts"], raw["hidden_size"], raw["moe_intermediate_size"]
        rows, columns = (hidden, width) if name == "down_proj" else (width, hidden)
        if columns % 16:
            raise ValueError("DiffusionGemma packed expert columns must be divisible by16")
        packed = np.empty((count, rows, columns // 2), np.uint8)
        scales = np.empty((count, rows * columns // 16), np.uint8)
        weight_scales = np.empty((count, 1), np.float32)
        inputs = np.empty((count, 1), np.float32)
        for index in range(count):
            stem = f"{self.prefix}.experts.{index}.{name}"
            weight = g.weights.raw(stem + ".weight")
            block = g.weights.raw(stem + ".weight_scale")
            if (
                weight.dtype != np.uint8
                or weight.shape != (rows, columns // 2)
                or block.dtype != np.dtype(ml_dtypes.float8_e4m3fn)
                or block.shape != (rows, columns // 16)
            ):
                raise ValueError(f"DiffusionGemma NVFP4 packed geometry/dtype differs: {stem}")
            global_scale = self._scalar(stem + ".weight_scale_2")
            combined = block.astype(np.float32) * global_scale
            if not np.isfinite(combined).all() or (combined < 0).any():
                raise ValueError(
                    f"DiffusionGemma NVFP4 block scales must be finite and nonnegative: {stem}"
                )
            packed[index] = weight
            scales[index] = block.view(np.uint8).reshape(-1)
            weight_scales[index, 0] = global_scale
            inputs[index, 0] = self._scalar(stem + ".input_scale")
        # Store eight source FP4 values in each INT32 constant. Decode after
        # loop selection, never materialize all E experts as dense constants.
        bank = g.const(packed.reshape(count, -1).view("<i4").astype(np.int32, copy=False))
        current = self.loop.add_iterator(bank, 0, False).get_output(0)
        scale_bank = g.const(scales.view("<i4").astype(np.int32, copy=False))
        scale_bytes = self.loop.add_iterator(scale_bank, 0, False).get_output(0)
        alpha = self.loop.add_iterator(g.const(weight_scales), 0, False).get_output(0)
        input_scale = self.loop.add_iterator(g.const(inputs), 0, False).get_output(0)
        codes = _unpack_codes(g, current, rows * columns, 4)
        table = g.const(
            np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        )
        decoded = n.add_gather(table, codes, 0).get_output(0)
        # The scalar type is only used to form a small constant LUT. No FP8
        # tensor operation or device-specific cast is required by this graph.
        scale_table = (
            np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        )
        scale_table = np.nan_to_num(
            scale_table, nan=0.0
        )  # Invalid source scales were rejected above.
        block_codes = _unpack_codes(g, scale_bytes, rows * columns // 16, 8)
        block = n.add_gather(g.const(scale_table), block_codes, 0).get_output(0)
        block = g.op(block, alpha, op.PROD)
        decoded = g.op(
            g.reshape(decoded, (rows, columns // 16, 16)),
            g.reshape(block, (rows, columns // 16, 1)),
            op.PROD,
        )
        result = g.cast(g.reshape(decoded, (rows, columns)), g.dtype), input_scale
        self.projections[name] = result
        return result

    def linear(self, value, projection):
        weight, scale = self.projection(projection)
        return self.graph.matmul(activation_qdq(self.graph, value, scale), weight, transpose=True)
