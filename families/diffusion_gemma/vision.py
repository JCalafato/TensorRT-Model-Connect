# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma patch encoder and projector, expressed with native TensorRT math."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph import Graph


def spatial_rope_tables(raw: dict, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """Each spatial axis rotates its own half-head; row zero represents -1."""
    dim = int(raw["head_dim"])
    params = raw["rope_parameters"]
    if dim % 4 or params.get("rope_type", "default") != "default":
        raise NotImplementedError("DiffusionGemma vision requires independent default spatial RoPE")
    inv = float(params["rope_theta"]) ** (-np.arange(0, dim // 2, 2, dtype=np.float32) / (dim // 2))
    angles = np.arange(-1, capacity, dtype=np.float32)[:, None] * inv
    angles = np.concatenate((angles, angles), axis=-1)
    return np.cos(angles), np.sin(angles)


class VisionGraph:
    """One image per invocation; padded patches have positions (-1, -1).

    Keep a fixed output capacity and expose its validity mask. The owning runtime
    removes padded rows, avoiding data-dependent output allocation in TensorRT.
    """

    def __init__(self, network, raw, weights, precision, soft_tokens):
        self.net, self.raw, self.weights = network, raw, weights
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.soft_tokens = int(soft_tokens)
        self.kernel = int(raw["pooling_kernel_size"])
        self.patches = self.soft_tokens * self.kernel**2
        self.buffers = []
        for name in (
            "hidden_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "intermediate_size",
            "patch_size",
            "position_embedding_size",
        ):
            if type(raw.get(name)) is not int or raw[name] <= 0:
                raise ValueError(f"DiffusionGemma vision {name} must be positive")
        if raw["hidden_activation"] != "gelu_pytorch_tanh" or raw.get("attention_bias", False):
            raise ValueError("DiffusionGemma vision requires unbiased attention and tanh GEGLU")
        if raw["num_attention_heads"] % raw["num_key_value_heads"]:
            raise ValueError("DiffusionGemma vision GQA geometry is inconsistent")
        if self.kernel < 2 or self.soft_tokens < 1:
            raise ValueError("DiffusionGemma vision requires a positive pooled patch budget")

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.net.add_cast(value, dtype).get_output(0)

    def const(self, value, dtype=None):
        value = np.ascontiguousarray(value)
        self.buffers.append(value)
        tensor = self.net.add_constant(value.shape, trt.Weights(value)).get_output(0)
        return self.cast(tensor, dtype) if dtype is not None else tensor

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, lhs, rhs, operation):
        return self.net.add_elementwise(lhs, rhs, operation).get_output(0)

    def weight(self, name, dtype=None):
        value = self.weights.floating(name)
        # Reference module parameters are converted to the working dtype
        # before normalization/standardization upcast them for FP32 arithmetic.
        value = self.const(value.astype(self.storage), self.dtype)
        return value if dtype is None else self.cast(value, dtype)

    def scalar(self, value, rank, dtype):
        return self.const(np.full((1,) * rank, value, np.float32), dtype)

    def scale(self, value, factor):
        return self.op(
            value, self.scalar(factor, len(value.shape), value.dtype), trt.ElementWiseOperation.PROD
        )

    def linear(self, value, name):
        weight = self.weight(name + ".weight")
        return self.matmul(value, self.cast(weight, value.dtype), transpose=True)

    # Mathematical primitives are owned by this same family; no other owner is
    # imported or linked. Keep the reference's rounded linear/GELU boundaries.
    matmul = Graph.matmul
    gelu = Graph.gelu

    def rows_to_heads(self, value, heads, dimension):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = (self.patches, heads, dimension)
        layer.second_transpose = (1, 0, 2)
        return self.reshape(layer.get_output(0), (1, heads, self.patches, dimension))

    def heads_to_rows(self, value, width):
        layer = self.net.add_shuffle(value)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (self.patches, width)
        return layer.get_output(0)

    def clip(self, value, name, direction):
        if not self.raw.get("use_clipped_linears", False):
            return value
        for suffix, operation, neutral in (
            ("min", trt.ElementWiseOperation.MAX, -np.inf),
            ("max", trt.ElementWiseOperation.MIN, np.inf),
        ):
            bound = self.weights.floating(f"{name}.{direction}_{suffix}")
            if not np.all(bound == neutral):
                value = self.op(
                    value,
                    self.const(np.asarray(bound).reshape((1,) * len(value.shape)), value.dtype),
                    operation,
                )
        return value

    def clipped_linear(self, value, name):
        value = self.clip(value, name, "input")
        return self.clip(self.linear(value, name + ".linear"), name, "output")

    def norm(self, value, name=None):
        dtype, rank = value.dtype, len(value.shape)
        value = self.cast(value, trt.float32)
        square = self.op(value, value, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(
            square, trt.ReduceOperation.AVG, 1 << (rank - 1), True
        ).get_output(0)
        mean = self.op(
            mean,
            self.scalar(self.raw["rms_norm_eps"], rank, trt.float32),
            trt.ElementWiseOperation.SUM,
        )
        inv = self.op(mean, self.scalar(-0.5, rank, trt.float32), trt.ElementWiseOperation.POW)
        output = self.op(value, inv, trt.ElementWiseOperation.PROD)
        if name is not None:
            weight = self.reshape(
                self.weight(name + ".weight", trt.float32), (1,) * (rank - 1) + (value.shape[-1],)
            )
            output = self.op(output, weight, trt.ElementWiseOperation.PROD)
        return self.cast(output, dtype)

    def rotate(self, value, heads, cos, sin):
        dim = int(self.raw["head_dim"])
        value = self.reshape(value, (self.patches, heads, 2, dim // 2))
        permutation = np.roll(np.arange(dim // 2, dtype=np.int32), dim // 4)
        rotated = self.net.add_gather(value, self.const(permutation), 3).get_output(0)
        signs = np.concatenate((-np.ones(dim // 4), np.ones(dim // 4))).astype(np.float32)
        rotated = self.op(
            rotated,
            self.const(signs.reshape(1, 1, 1, -1), value.dtype),
            trt.ElementWiseOperation.PROD,
        )

        def product(left, right):
            return self.cast(
                self.op(
                    self.cast(left, trt.float32),
                    self.cast(right, trt.float32),
                    trt.ElementWiseOperation.PROD,
                ),
                self.dtype,
            )

        result = self.cast(
            self.op(
                self.cast(product(value, cos), trt.float32),
                self.cast(product(rotated, sin), trt.float32),
                trt.ElementWiseOperation.SUM,
            ),
            self.dtype,
        )
        return self.reshape(result, (self.patches, heads * dim))

    def attention(self, value, name, cos, sin, mask):
        heads, kv_heads, dim = (
            int(self.raw[k]) for k in ("num_attention_heads", "num_key_value_heads", "head_dim")
        )
        projected = []
        for kind, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
            item = self.clipped_linear(value, f"{name}.{kind}_proj")
            item = self.norm(
                self.reshape(item, (self.patches, count, dim)),
                f"{name}.{kind}_norm" if kind != "v" else None,
            )
            item = self.reshape(item, (self.patches, count * dim))
            if kind != "v":
                item = self.rotate(item, count, cos, sin)
            item = self.rows_to_heads(self.cast(item, trt.float32), count, dim)
            projected.append(self.reshape(item, (kv_heads, count // kv_heads, self.patches, dim)))
        q, k, v = projected
        scores = self.net.add_matrix_multiply(
            q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.op(scores, mask, trt.ElementWiseOperation.SUM)  # score scale is one
        probabilities = self.net.add_softmax(scores)
        probabilities.axes = 1 << 3
        output = self.net.add_matrix_multiply(
            probabilities.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        output = self.reshape(output, (1, heads, self.patches, dim))
        output = self.heads_to_rows(output, heads * dim)
        return self.clipped_linear(self.cast(output, self.dtype), f"{name}.o_proj")

    def build(self):
        n, raw, p = self.net, self.raw, self.patches
        hidden, dim = int(raw["hidden_size"]), int(raw["head_dim"])
        pixels = n.add_input("pixel_values", trt.float32, (p, 3 * int(raw["patch_size"]) ** 2))
        positions = n.add_input("pixel_position_ids", trt.int32, (p, 2))
        zero = self.const(np.zeros((1, 1), np.int32))
        valid_axes = self.op(
            positions, self.const(np.full((1, 1), -1, np.int32)), trt.ElementWiseOperation.GREATER
        )
        valid = n.add_reduce(
            self.cast(valid_axes, trt.int32), trt.ReduceOperation.MAX, 2, True
        ).get_output(0)
        valid = self.cast(valid, trt.bool)
        clamped = self.op(positions, zero, trt.ElementWiseOperation.MAX)
        prefix = "model.encoder.vision_tower"
        normalized = self.scale(
            self.op(pixels, self.scalar(-0.5, 2, trt.float32), trt.ElementWiseOperation.SUM), 2
        )
        state = self.linear(
            self.cast(normalized, self.dtype), prefix + ".patch_embedder.input_proj"
        )
        table = self.weights.floating(prefix + ".patch_embedder.position_embedding_table")
        embedded = []
        axes = []
        for axis in (0, 1):
            ids = n.add_gather(clamped, self.const(np.array([axis], np.int32)), 1).get_output(0)
            axes.append(ids)
            embedded.append(
                n.add_gather(
                    self.const(table[axis].astype(self.storage), self.dtype),
                    self.reshape(ids, (p,)),
                    0,
                ).get_output(0)
            )
        positional = self.op(*embedded, trt.ElementWiseOperation.SUM)
        positional = n.add_select(valid, positional, self.scalar(0, 2, self.dtype)).get_output(0)
        state = self.op(state, positional, trt.ElementWiseOperation.SUM)
        mask = n.add_select(
            self.reshape(valid, (1, 1, 1, p)),
            self.scalar(0, 4, trt.float32),
            self.scalar(-np.inf, 4, trt.float32),
        ).get_output(0)
        shifted = self.op(
            positions, self.const(np.ones((1, 1), np.int32)), trt.ElementWiseOperation.SUM
        )
        tables = spatial_rope_tables(raw, int(raw["position_embedding_size"]))
        cos, sin = [
            self.reshape(
                n.add_gather(self.const(t, self.dtype), shifted, 0).get_output(0),
                (p, 1, 2, dim // 2),
            )
            for t in tables
        ]
        for index in range(int(raw["num_hidden_layers"])):
            name = f"{prefix}.encoder.layers.{index}"
            attended = self.attention(
                self.norm(state, name + ".input_layernorm"), name + ".self_attn", cos, sin, mask
            )
            state = self.op(
                state,
                self.norm(attended, name + ".post_attention_layernorm"),
                trt.ElementWiseOperation.SUM,
            )
            value = self.norm(state, name + ".pre_feedforward_layernorm")
            gate = self.clipped_linear(value, name + ".mlp.gate_proj")
            gate = self.gelu(gate)
            up = self.clipped_linear(value, name + ".mlp.up_proj")
            value = self.clipped_linear(
                self.op(gate, up, trt.ElementWiseOperation.PROD), name + ".mlp.down_proj"
            )
            state = self.op(
                state,
                self.norm(value, name + ".post_feedforward_layernorm"),
                trt.ElementWiseOperation.SUM,
            )
        state = n.add_select(valid, state, self.scalar(0, 2, self.dtype)).get_output(0)
        x, y = axes
        maximum = n.add_reduce(x, trt.ReduceOperation.MAX, 1, True).get_output(0)
        width = self.op(
            maximum, self.const(np.ones((1, 1), np.int32)), trt.ElementWiseOperation.SUM
        )
        divisor = self.const(np.full((1, 1), self.kernel, np.int32))
        column = self.op(x, divisor, trt.ElementWiseOperation.FLOOR_DIV)
        row = self.op(y, divisor, trt.ElementWiseOperation.FLOOR_DIV)
        width = self.op(width, divisor, trt.ElementWiseOperation.FLOOR_DIV)
        groups = self.op(
            column, self.op(row, width, trt.ElementWiseOperation.PROD), trt.ElementWiseOperation.SUM
        )
        target = self.const(np.arange(self.soft_tokens, dtype=np.int32).reshape(1, -1))
        member = self.op(groups, target, trt.ElementWiseOperation.EQUAL)
        weights = self.scale(self.cast(member, trt.float32), 1 / self.kernel**2)
        pooled = n.add_matrix_multiply(
            weights,
            trt.MatrixOperation.TRANSPOSE,
            self.cast(state, trt.float32),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        pooled = self.scale(self.cast(self.cast(pooled, self.dtype), trt.float32), hidden**0.5)
        if raw.get("standardize", False):
            bias = self.reshape(self.weight(prefix + ".std_bias", trt.float32), (1, hidden))
            scale = self.reshape(self.weight(prefix + ".std_scale", trt.float32), (1, hidden))
            pooled = self.op(
                self.op(pooled, bias, trt.ElementWiseOperation.SUB),
                scale,
                trt.ElementWiseOperation.PROD,
            )
        features = self.linear(
            self.norm(self.cast(pooled, self.dtype)),
            "model.encoder.embed_vision.embedding_projection",
        )
        features = self.cast(features, trt.float32)
        features.name = "image_features"
        n.mark_output(features)
        valid_output = n.add_reduce(
            self.cast(member, trt.int32), trt.ReduceOperation.MAX, 1, False
        ).get_output(0)
        valid_output.name = "image_feature_mask"
        n.mark_output(valid_output)
