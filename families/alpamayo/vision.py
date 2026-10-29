# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo's native visual encoder and its family-owned DeepStack mergers."""

import numpy as np
import tensorrt as trt

from .graph import Graph


class VisionGraph(Graph):
    def __init__(self, network, weights, config, precision):
        super().__init__(network, weights, precision)
        self.config = config

    def rounded_op(self, left, right, operation):
        result = self.op(self.cast(left, trt.float32), self.cast(right, trt.float32), operation)
        return self.cast(result, left.dtype)

    def transpose(self, value, order):
        layer = self.net.add_shuffle(value)
        layer.second_transpose = order
        return layer.get_output(0)

    def patch_embedding(self, pixels):
        config = self.config
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        shape = (config.hidden_size, config.in_channels, *kernel)
        name = "vision.patch_embed.proj"
        if tuple(self.weights.info(name + ".weight")[1]) != shape:
            raise ValueError("Unexpected Alpamayo visual patch projection shape")
        if tuple(self.weights.info(name + ".bias")[1]) != (config.hidden_size,):
            raise ValueError("Unexpected Alpamayo visual patch projection bias")
        # Preserve the requested operand precision, but accumulate patch
        # reductions in FP32 just like the dense projections. Round the
        # convolution output before applying the separately rounded bias.
        weight = self.weights.array(name + ".weight", self.storage).astype(np.float32, copy=False)
        bias = self.weights.array(name + ".bias", self.storage)
        self.buffers.extend([weight, bias])
        pixels = self.cast(self.cast(pixels, self.dtype), trt.float32)
        pixels = self.reshape(pixels, (-1, config.in_channels, *kernel))
        layer = self.net.add_convolution_nd(
            pixels, config.hidden_size, kernel, trt.Weights(weight), trt.Weights()
        )
        layer.stride_nd = kernel
        # The provider rounds the convolution before adding its FP16 bias.
        # Supplying the bias to TensorRT's convolution instead permits a
        # single fused affine rounding, which changes the provider arithmetic.
        projected = self.reshape(
            self.cast(layer.get_output(0), self.dtype), (-1, config.hidden_size)
        )
        bias = self.const(bias.reshape(1, config.hidden_size))
        return self.rounded_op(projected, bias, trt.ElementWiseOperation.SUM)

    def positions(self, indices, coefficients):
        config = self.config
        table = self.weight(
            "vision.pos_embed.weight", (config.num_position_embeddings, config.hidden_size)
        )
        values = self.net.add_gather(table, indices, 0).get_output(0)
        coefficients = self.reshape(self.cast(coefficients, self.dtype), (0, 0, 1))
        values = self.rounded_op(values, coefficients, trt.ElementWiseOperation.PROD)
        terms = [
            self.net.add_gather(values, self.const(np.array([i], np.int32)), 0).get_output(0)
            for i in range(4)
        ]
        result = terms[0]
        for term in terms[1:]:
            result = self.rounded_op(result, term, trt.ElementWiseOperation.SUM)
        return self.reshape(result, (-1, config.hidden_size))

    def norm(self, value, name):
        # Vision LayerNorm preserves its input dtype; action projection has a
        # different residual dtype and does not share this orchestration.
        return self.cast(self.layer_norm(value, name, 1e-6), value.dtype)

    def gelu(self, value, *, approximate):
        kind = trt.ActivationType.GELU_TANH if approximate else trt.ActivationType.GELU_ERF
        result = self.net.add_activation(self.cast(value, trt.float32), kind).get_output(0)
        return self.cast(result, value.dtype)

    def rotary(self, value, cosine, sine):
        width = self.config.hidden_size // self.config.num_heads
        value = self.cast(value, trt.float32)
        first = self.net.add_gather(
            value, self.const(np.arange(width // 2, dtype=np.int32)), 2
        ).get_output(0)
        second = self.net.add_gather(
            value, self.const(np.arange(width // 2, width, dtype=np.int32)), 2
        ).get_output(0)
        rotated = self.concat(
            [self.net.add_unary(second, trt.UnaryOperation.NEG).get_output(0), first], 2
        )
        # The upstream vision rotary operation computes both products and the
        # sum in FP32, then casts once, unlike the language rotary operation.
        result = self.op(
            self.op(value, cosine, trt.ElementWiseOperation.PROD),
            self.op(rotated, sine, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM,
        )
        return self.cast(result, self.dtype)

    def matmul(self, left, right, *, transpose=False):
        left = self.cast(self.cast(left, self.dtype), trt.float32)
        right = self.cast(self.cast(right, self.dtype), trt.float32)
        result = self.net.add_matrix_multiply(
            left,
            trt.MatrixOperation.NONE,
            right,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)
        return self.cast(result, self.dtype)

    def attention(self, hidden, mask, cosine, sine, stem):
        config = self.config
        width = config.hidden_size // config.num_heads
        qkv = self.linear(hidden, stem + ".qkv", 3 * config.hidden_size)
        qkv = self.reshape(qkv, (0, 3, config.num_heads, width))
        pieces = [
            self.net.add_gather(qkv, self.const(np.array([i], np.int32)), 1).get_output(0)
            for i in range(3)
        ]
        query, key, value = [self.reshape(item, (0, config.num_heads, width)) for item in pieces]
        query = self.transpose(self.rotary(query, cosine, sine), (1, 0, 2))
        key = self.transpose(self.rotary(key, cosine, sine), (1, 0, 2))
        value = self.transpose(value, (1, 0, 2))
        scores = self.matmul(query, key, transpose=True)
        scores = self.op(
            self.cast(scores, trt.float32),
            self.scalar(width**-0.5, 3),
            trt.ElementWiseOperation.PROD,
        )
        scores = self.cast(self.cast(scores, self.dtype), trt.float32)
        scores = self.op(scores, mask, trt.ElementWiseOperation.SUM)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << 2
        attended = self.matmul(softmax.get_output(0), value)
        attended = self.reshape(self.transpose(attended, (1, 0, 2)), (-1, config.hidden_size))
        return self.linear(attended, stem + ".proj", config.hidden_size)

    def merger(self, hidden, stem, *, postshuffle):
        width = self.config.hidden_size * self.config.spatial_merge_size**2
        if postshuffle:
            hidden = self.reshape(hidden, (-1, width))
        hidden = self.norm(hidden, stem + ".norm")
        hidden = self.reshape(hidden, (-1, width))
        hidden = self.gelu(self.linear(hidden, stem + ".linear_fc1", width), approximate=False)
        return self.linear(hidden, stem + ".linear_fc2", self.config.out_hidden_size)

    def forward(self, pixels, position_indices, position_weights, mask, cosine, sine):
        """Encode packed patch groups without attention across images/frames.

        Patch order is temporal, merged-block row/column, intra-block row/column.
        Position indices/weights are four-corner learned-grid interpolation in
        that same order. The additive mask isolates each temporal grid's patches.
        """
        hidden = self.rounded_op(
            self.patch_embedding(pixels),
            self.positions(position_indices, position_weights),
            trt.ElementWiseOperation.SUM,
        )
        deepstack = []
        for index in range(self.config.depth):
            stem = f"vision.blocks.{index}"
            attended = self.attention(
                self.norm(hidden, stem + ".norm1"), mask, cosine, sine, stem + ".attn"
            )
            hidden = self.rounded_op(hidden, attended, trt.ElementWiseOperation.SUM)
            mlp = self.linear(
                self.norm(hidden, stem + ".norm2"),
                stem + ".mlp.linear_fc1",
                self.config.intermediate_size,
            )
            mlp = self.linear(
                self.gelu(mlp, approximate=True), stem + ".mlp.linear_fc2", self.config.hidden_size
            )
            hidden = self.rounded_op(hidden, mlp, trt.ElementWiseOperation.SUM)
            if index in self.config.deepstack_visual_indexes:
                ordinal = self.config.deepstack_visual_indexes.index(index)
                deepstack.append(
                    self.merger(hidden, f"vision.deepstack_merger_list.{ordinal}", postshuffle=True)
                )
        return self.merger(hidden, "vision.merger", postshuffle=False), deepstack
