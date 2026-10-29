# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo VLM decoder, including visual injection and reusable prefix KV."""

import numpy as np
import tensorrt as trt

from .graph import Graph


class TextGraph(Graph):
    def __init__(self, network, weights, config, vocab_size, precision):
        super().__init__(network, weights, precision)
        self.config, self.vocab_size = config, vocab_size

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

        # The provider rounds each product before their sum. Express these
        # boundaries explicitly rather than allowing a fused multiply-add.
        def product(left, right):
            return self.cast(
                self.op(
                    self.cast(left, trt.float32),
                    self.cast(right, trt.float32),
                    trt.ElementWiseOperation.PROD,
                ),
                value.dtype,
            )

        first = product(value, cosine)
        second = product(rotated, sine)
        return self.cast(
            self.op(
                self.cast(first, trt.float32),
                self.cast(second, trt.float32),
                trt.ElementWiseOperation.SUM,
            ),
            value.dtype,
        )

    def attention(self, hidden, past_key, past_value, mask, cosine, sine, stem):
        config = self.config
        heads, kv_heads, width = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        query = self.linear(hidden, stem + ".q_proj", heads * width, bias=False)
        key = self.linear(hidden, stem + ".k_proj", kv_heads * width, bias=False)
        value = self.linear(hidden, stem + ".v_proj", kv_heads * width, bias=False)
        query = self.rms_norm(
            self.reshape(query, (0, 0, heads, width)), stem + ".q_norm", config.rms_norm_eps
        )
        key = self.rms_norm(
            self.reshape(key, (0, 0, kv_heads, width)), stem + ".k_norm", config.rms_norm_eps
        )
        query = self.rotary(self.transpose(query, (0, 2, 1, 3)), cosine, sine)
        key = self.rotary(self.transpose(key, (0, 2, 1, 3)), cosine, sine)
        value = self.transpose(self.reshape(value, (0, 0, kv_heads, width)), (0, 2, 1, 3))
        # Preserve compact KV heads and append only actual VLM tokens. This
        # cache subsequently conditions the separate non-causal action expert.
        key = self.concat([past_key, key], 2)
        value = self.concat([past_value, value], 2)
        repeat = self.const(np.repeat(np.arange(kv_heads, dtype=np.int32), heads // kv_heads))
        repeated_key = self.net.add_gather(key, repeat, 1).get_output(0)
        repeated_value = self.net.add_gather(value, repeat, 1).get_output(0)
        scores = self.matmul(query, repeated_key, transpose=True)
        scores = self.op(
            self.cast(scores, trt.float32),
            self.scalar(width**-0.5, 4),
            trt.ElementWiseOperation.PROD,
        )
        scores = self.cast(self.cast(scores, self.dtype), trt.float32)
        scores = self.op(scores, mask, trt.ElementWiseOperation.SUM)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << 3
        attended = self.matmul(softmax.get_output(0), repeated_value)
        attended = self.reshape(self.transpose(attended, (0, 2, 1, 3)), (0, 0, heads * width))
        return self.linear(attended, stem + ".o_proj", config.hidden_size, bias=False), key, value

    def forward(
        self,
        tokens,
        visual,
        visual_mask,
        deepstack,
        past_keys,
        past_values,
        mask,
        cosine,
        sine,
        *,
        last_logits_only=False,
    ):
        """Return logits and growing KV for prefill or incremental decoding.

        Visual features are aligned to the token grid; visual_mask selects the
        embedding replacements. Each DeepStack input is zero at nonvisual
        positions and is added after its corresponding early decoder layer.
        The caller owns the causal/padding mask and interleaved 3-axis rotary
        tables. Prefill uses genuinely empty prefix tensors, not dummy tokens.
        """
        config = self.config
        if (
            len(past_keys) != config.num_hidden_layers
            or len(past_values) != config.num_hidden_layers
        ):
            raise ValueError("Every VLM layer requires its matching prefix KV pair")
        if len(deepstack) > config.num_hidden_layers:
            raise ValueError("DeepStack injection exceeds the VLM layer count")
        embedding = self.weight("text.embed_tokens.weight", (self.vocab_size, config.hidden_size))
        hidden = self.net.add_gather(embedding, tokens, 0).get_output(0)
        hidden = self.net.add_select(visual_mask, visual, hidden).get_output(0)
        keys, values = [], []
        for index in range(config.num_hidden_layers):
            stem = f"text.layers.{index}"
            normalized = self.rms_norm(hidden, stem + ".input_layernorm", config.rms_norm_eps)
            attended, key, value = self.attention(
                normalized,
                past_keys[index],
                past_values[index],
                mask,
                cosine,
                sine,
                stem + ".self_attn",
            )
            keys.append(key)
            values.append(value)
            hidden = self.op(hidden, attended, trt.ElementWiseOperation.SUM)
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
            down = self.linear(
                self.op(gate, up, trt.ElementWiseOperation.PROD),
                stem + ".mlp.down_proj",
                config.hidden_size,
                bias=False,
            )
            hidden = self.op(hidden, down, trt.ElementWiseOperation.SUM)
            if index < len(deepstack):
                hidden = self.op(hidden, deepstack[index], trt.ElementWiseOperation.SUM)
        hidden = self.rms_norm(hidden, "text.norm", config.rms_norm_eps)
        if last_logits_only:
            shape = self.net.add_shape(hidden).get_output(0)
            length = self.net.add_gather(shape, self.const(np.array([1], np.int32)), 0).get_output(
                0
            )
            last = self.op(
                length,
                self.cast(self.const(np.array([1], np.int32)), length.dtype),
                trt.ElementWiseOperation.SUB,
            )
            hidden = self.net.add_gather(hidden, last, 1).get_output(0)
        logits = self.linear(hidden, "text.lm_head", self.vocab_size, bias=False)
        return self.cast(logits, trt.float32), keys, values
