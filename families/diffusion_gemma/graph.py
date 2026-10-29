# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma mathematical primitives, using only TensorRT network APIs.

Caches use absolute positions in fixed-capacity storage. Encoder attention is
causal (with same-image bidirectionality in both layer types); a decoder canvas is
fully bidirectional and sees one common window of the committed prefix.
"""

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, config, weights, precision):
        self.net, self.config, self.weights = network, config, weights
        self.raw = config.text
        self.dtype = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}[precision]
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.buffers = []
        self.constants = {}

    def cast(self, tensor, dtype):
        return tensor if tensor.dtype == dtype else self.net.add_cast(tensor, dtype).get_output(0)

    def const(self, array, dtype=None):
        array = np.ascontiguousarray(array)
        self.buffers.append(array)
        tensor = self.net.add_constant(array.shape, trt.Weights(array)).get_output(0)
        return tensor if dtype is None else self.cast(tensor, dtype)

    def weight(self, name, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        key = (name, dtype)
        if key not in self.constants:
            array = self.weights.floating(name).astype(self.storage)
            # Module conversion rounds parameter storage before normalization
            # upcasts it for FP32 arithmetic, including BF16 parameters.
            value = self.const(array, self.dtype)
            self.constants[key] = self.cast(value, dtype)
        return self.constants[key]

    def reshape(self, tensor, shape):
        layer = self.net.add_shuffle(tensor)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, left, right, operation):
        return self.net.add_elementwise(left, right, operation).get_output(0)

    def scale(self, value, factor):
        factor = self.const(np.full((1,) * len(value.shape), factor, np.float32), value.dtype)
        return self.op(value, factor, trt.ElementWiseOperation.PROD)

    def concat(self, tensors, axis):
        layer = self.net.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    def matmul(self, left, right, *, transpose=False):
        # Reference linear/attention operations accumulate rounded FP16/BF16
        # operands in FP32. A compute-dtype TRT GEMM lost a ULP already at the
        # first Q/K projection; explicitly represent the accumulator boundary.
        result = self.net.add_matrix_multiply(
            self.cast(left, trt.float32),
            trt.MatrixOperation.NONE,
            self.cast(right, trt.float32),
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)
        return self.cast(result, left.dtype)

    def linear(self, value, stem):
        if self.weights.has(stem + ".weight_scale_2"):
            raise NotImplementedError(
                "DiffusionGemma prepared projections require the expert graph"
            )
        weight = self.cast(self.weight(stem + ".weight"), value.dtype)
        return self.matmul(value, weight, transpose=True)

    def norm(self, value, name=None):
        fp32 = self.cast(value, trt.float32)
        rank = len(value.shape)
        squared = self.op(fp32, fp32, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(
            squared, trt.ReduceOperation.AVG, 1 << (rank - 1), True
        ).get_output(0)
        epsilon = self.const(np.full((1,) * rank, self.raw["rms_norm_eps"], np.float32))
        exponent = self.const(np.full((1,) * rank, -0.5, np.float32))
        inverse = self.op(
            self.op(mean, epsilon, trt.ElementWiseOperation.SUM),
            exponent,
            trt.ElementWiseOperation.POW,
        )
        result = self.op(fp32, inverse, trt.ElementWiseOperation.PROD)
        if name is not None:
            weight = self.reshape(
                self.weight(name + ".weight", trt.float32), (1,) * (rank - 1) + (value.shape[-1],)
            )
            result = self.op(result, weight, trt.ElementWiseOperation.PROD)
        return self.cast(result, value.dtype)

    def gelu(self, value):
        # PyTorch tanh GELU; retain its output compute dtype before the gated product.
        fp32 = self.cast(value, trt.float32)
        square = self.op(fp32, fp32, trt.ElementWiseOperation.PROD)
        cube = self.op(square, fp32, trt.ElementWiseOperation.PROD)
        inner = self.scale(
            self.op(fp32, self.scale(cube, 0.044715), trt.ElementWiseOperation.SUM),
            (2 / np.pi) ** 0.5,
        )
        tanh = self.net.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        one = self.const(np.ones((1,) * len(value.shape), np.float32))
        result = self.scale(
            self.op(
                fp32,
                self.op(one, tanh, trt.ElementWiseOperation.SUM),
                trt.ElementWiseOperation.PROD,
            ),
            0.5,
        )
        return self.cast(result, value.dtype)

    def mlp(self, value, stem):
        gate = self.gelu(self.linear(value, stem + ".gate_proj"))
        up = self.linear(value, stem + ".up_proj")
        return self.linear(self.op(gate, up, trt.ElementWiseOperation.PROD), stem + ".down_proj")

    def token_embeddings(self, tokens):
        table = self.weight("model.decoder.embed_tokens.weight")
        embedded = self.net.add_gather(table, tokens, 0).get_output(0)
        return self.scale(embedded, self.raw["hidden_size"] ** 0.5)

    def self_condition(self, embeddings, previous_soft_embeddings):
        prefix = "model.decoder.self_conditioning"
        conditioned = self.mlp(self.norm(previous_soft_embeddings, prefix + ".pre_norm"), prefix)
        return self.norm(self.op(embeddings, conditioned, trt.ElementWiseOperation.SUM))

    def next_soft_embeddings(self, logits, temperature):
        # HF retains processed logits in embedding dtype, then starts the next
        # step with an FP32 softmax. Do not erase that intermediate rounding.
        scaled = self.op(self.cast(logits, trt.float32), temperature, trt.ElementWiseOperation.DIV)
        rounded = self.cast(self.cast(scaled, self.dtype), trt.float32)
        softmax = self.net.add_softmax(rounded)
        softmax.axes = 1 << 1
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        table = self.weight("model.decoder.embed_tokens.weight")
        soft = self.matmul(probabilities, table)
        # Scale AFTER multiplying by the unscaled shared embedding, as HF does.
        return self.scale(soft, self.raw["hidden_size"] ** 0.5)

    def phase_scale(self, value, layer, decoder):
        enc = self.weight(self.weights.layer_scale_name(layer, decoder=False))
        dec = self.weight(self.weights.layer_scale_name(layer, decoder=True))
        scalar = self.net.add_select(decoder, dec, enc).get_output(0)
        return self.op(value, self.reshape(scalar, (1, 1)), trt.ElementWiseOperation.PROD)

    def attention_masks(self, positions, prefix_length, decoder, capacity, media_blocks=None):
        n, op = self.net, trt.ElementWiseOperation
        cache_pos = self.const(np.arange(capacity, dtype=np.int32))
        all_pos = self.concat([cache_pos, positions], 0)
        key = self.reshape(all_pos, (1, -1))
        query = self.reshape(positions, (-1, 1))
        # Invalid static cache slots must never become visible in either phase.
        cached_valid = self.op(cache_pos, prefix_length, op.LESS)
        current_valid = self.op(positions, positions, op.EQUAL)
        valid = self.reshape(self.concat([cached_valid, current_valid], 0), (1, -1))
        future = self.op(key, query, op.GREATER)
        causal = n.add_unary(future, trt.UnaryOperation.NOT).get_output(0)
        phase = self.reshape(decoder, (1, 1))
        full = self.op(valid, self.op(causal, phase, op.OR), op.AND)
        window = self.const(np.array([[self.raw["sliding_window"]]], np.int32))
        encoder_window = self.op(key, self.op(query, window, op.SUB), op.GREATER)
        # All canvas queries see the same prefix window. Static-cache reference
        # includes the token at prefix_length-window, unlike the dynamic cache.
        lower = self.op(self.reshape(prefix_length, (1, 1)), window, op.SUB)
        too_old = self.op(key, lower, op.LESS)
        decoder_window = n.add_unary(too_old, trt.UnaryOperation.NOT).get_output(0)
        local_allowed = self.op(encoder_window, causal, op.AND)
        if media_blocks is not None and self.raw.get("use_bidirectional_attention") == "vision":
            prior_blocks = self.const(np.zeros(capacity, np.int32))
            blocks = self.reshape(self.concat([prior_blocks, media_blocks], 0), (1, -1))
            same = self.op(blocks, self.reshape(media_blocks, (-1, 1)), op.EQUAL)
            image = self.op(
                self.reshape(media_blocks, (-1, 1)),
                self.const(np.zeros((1, 1), np.int32)),
                op.GREATER,
            )
            # The reference overlays the whole image block on BOTH causal
            # masks, after constructing the sliding-window restriction.
            # Consequently an image may attend beyond the ordinary window.
            image_block = self.op(same, image, op.AND)
            full = self.op(valid, self.op(full, image_block, op.OR), op.AND)
            local_allowed = self.op(local_allowed, image_block, op.OR)
        local = n.add_select(phase, decoder_window, local_allowed).get_output(0)
        local = self.op(local, valid, op.AND)
        result = {}
        zero = self.const(np.zeros((1, 1), np.float32))
        blocked = self.const(np.full((1, 1), -np.inf, np.float32))
        for kind, allowed in (("full_attention", full), ("sliding_attention", local)):
            mask = n.add_select(allowed, zero, blocked).get_output(0)
            shape = self.concat(
                [self.const(np.array([1, 1], np.int64)), n.add_shape(mask).get_output(0)], 0
            )
            layer = n.add_shuffle(mask)
            layer.set_input(1, shape)
            result[kind] = layer.get_output(0)
        return result
