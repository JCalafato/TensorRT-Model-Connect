# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-causal action expert attending to an immutable VLM prefix cache."""

import numpy as np
import tensorrt as trt

from .graph import Graph


class ExpertGraph(Graph):
    def __init__(self, network, weights, config, precision):
        super().__init__(network, weights, precision)
        self.config = config

    def transpose(self, value, order):
        layer = self.net.add_shuffle(value)
        layer.second_transpose = order
        return layer.get_output(0)

    def matmul(self, left, right, transpose=False):
        left = self.cast(self.cast(left, self.dtype), trt.float32)
        right = self.cast(self.cast(right, self.dtype), trt.float32)
        result = self.net.add_matrix_multiply(
            left,
            trt.MatrixOperation.NONE,
            right,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)
        return self.cast(result, self.dtype)

    def rotary(self, value, cosine, sine):
        # The provider generates rotary values in the incoming residual dtype.
        value = self.cast(value, cosine.dtype)
        half = self.config.head_dim // 2
        first = self.net.add_gather(
            value, self.const(np.arange(half, dtype=np.int32)), 3
        ).get_output(0)
        second = self.net.add_gather(
            value, self.const(np.arange(half, 2 * half, dtype=np.int32)), 3
        ).get_output(0)
        rotated = self.concat(
            [self.net.add_unary(second, trt.UnaryOperation.NEG).get_output(0), first], 3
        )
        return self.op(
            self.op(value, cosine, trt.ElementWiseOperation.PROD),
            self.op(rotated, sine, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM,
        )

    def attention(self, hidden, prefix_key, prefix_value, mask, cosine, sine, stem):
        config = self.config
        heads, kv_heads, width = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        query = self.linear(hidden, stem + ".q_proj", heads * width, bias=False)
        key = self.linear(hidden, stem + ".k_proj", kv_heads * width, bias=False)
        value = self.linear(hidden, stem + ".v_proj", kv_heads * width, bias=False)
        query = self.reshape(query, (0, 0, heads, width))
        key = self.reshape(key, (0, 0, kv_heads, width))
        query = self.rms_norm(query, stem + ".q_norm", config.rms_norm_eps)
        key = self.rms_norm(key, stem + ".k_norm", config.rms_norm_eps)
        query = self.rotary(self.transpose(query, (0, 2, 1, 3)), cosine, sine)
        key = self.rotary(self.transpose(key, (0, 2, 1, 3)), cosine, sine)
        value = self.transpose(self.reshape(value, (0, 0, kv_heads, width)), (0, 2, 1, 3))
        key = self.concat([self.cast(prefix_key, key.dtype), key], 2)
        value = self.concat([prefix_value, value], 2)
        repeat = self.const(np.repeat(np.arange(kv_heads, dtype=np.int32), heads // kv_heads))
        key = self.net.add_gather(key, repeat, 1).get_output(0)
        value = self.net.add_gather(value, repeat, 1).get_output(0)
        scores = self.matmul(query, key, transpose=True)
        scores = self.op(
            self.cast(scores, trt.float32),
            self.scalar(width**-0.5, 4),
            trt.ElementWiseOperation.PROD,
        )
        scores = self.cast(self.cast(scores, self.dtype), trt.float32)
        scores = self.op(scores, mask, trt.ElementWiseOperation.SUM)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << 3
        attended = self.matmul(softmax.get_output(0), value)
        attended = self.reshape(self.transpose(attended, (0, 2, 1, 3)), (0, 0, heads * width))
        return self.linear(attended, stem + ".o_proj", config.hidden_size, bias=False)

    def forward(self, hidden, prefix_keys, prefix_values, mask, cosine, sine):
        """Return action hidden states; never append action tokens to the VLM cache.

        Mask shape is (B, 1, T, prefix_length + T). The caller masks only
        post-EOS prefix padding; all action queries attend to all action keys.
        Rotary tables have shape (B, 1, T, head_dim).
        """
        config = self.config
        if (
            len(prefix_keys) != config.num_hidden_layers
            or len(prefix_values) != config.num_hidden_layers
        ):
            raise ValueError("Every expert layer requires its matching VLM prefix KV pair")
        for index in range(config.num_hidden_layers):
            stem = f"expert.layers.{index}"
            normalized = self.rms_norm(hidden, stem + ".input_layernorm", config.rms_norm_eps)
            attended = self.attention(
                normalized,
                prefix_keys[index],
                prefix_values[index],
                mask,
                cosine,
                sine,
                stem + ".self_attn",
            )
            hidden = self.op(
                hidden, self.cast(attended, hidden.dtype), trt.ElementWiseOperation.SUM
            )
            normalized = self.rms_norm(
                hidden, stem + ".post_attention_layernorm", config.rms_norm_eps
            )
            gate = self.silu(
                self.linear(
                    normalized, stem + ".mlp.gate_proj", config.intermediate_size, bias=False
                )
            )
            up = self.linear(
                normalized, stem + ".mlp.up_proj", config.intermediate_size, bias=False
            )
            fused = self.op(gate, up, trt.ElementWiseOperation.PROD)
            down = self.linear(fused, stem + ".mlp.down_proj", config.hidden_size, bias=False)
            hidden = self.op(hidden, self.cast(down, hidden.dtype), trt.ElementWiseOperation.SUM)
        return self.rms_norm(hidden, "expert.norm", config.rms_norm_eps)
