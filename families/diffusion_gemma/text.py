# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared-weight encoder/denoiser graph with explicit phase and cache ownership."""

import numpy as np
import tensorrt as trt

from .attention import add_attention, rotary_tables
from .experts import add_experts
from .graph import Graph


class TextGraph(Graph):
    def build(self, capacity, *, conditioned=False):
        n, raw, op = self.net, self.raw, trt.ElementWiseOperation
        hidden = raw["hidden_size"]
        tokens = n.add_input("token_id", trt.int32, (-1,))
        positions = n.add_input("position_id", trt.int32, (-1,))
        prefix = n.add_input("cache_length", trt.int32, (1,))
        phase = n.add_input("decoder_phase", trt.int32, (1,))
        decoder = self.op(phase, self.const(np.zeros(1, np.int32)), op.GREATER)
        previous = n.add_input("previous_soft_embeddings", self.dtype, (-1, hidden))
        temperature = n.add_input("temperature", trt.float32, (1, 1))
        indices = n.add_input("logits_indices", trt.int32, (-1,))
        # The multimodal encoder replaces image IDs with PAD even when no
        # pixels are supplied (e.g. a generated image ID is committed). The
        # denoiser retains ordinary vocabulary semantics for the same ID.
        embedding_tokens = tokens
        if "image_token_id" in self.config.raw:
            image = self.op(
                tokens,
                self.const(np.array([self.config.raw["image_token_id"]], np.int32)),
                op.EQUAL,
            )
            encoder = n.add_unary(decoder, trt.UnaryOperation.NOT).get_output(0)
            replace = self.op(image, encoder, op.AND)
            padding = self.const(np.array([raw.get("pad_token_id", 0)], np.int32))
            embedding_tokens = n.add_select(replace, padding, tokens).get_output(0)
        embeddings = self.token_embeddings(embedding_tokens)
        blocks = None
        if conditioned:
            blocks = n.add_input("media_blocks", trt.int32, (-1,))
            media = n.add_input("media_features", trt.float32, (-1, hidden))
            media = self.cast(media, self.dtype)
            valid = self.op(blocks, self.const(np.zeros(1, np.int32)), op.GREATER)
            embeddings = n.add_select(self.reshape(valid, (-1, 1)), media, embeddings).get_output(0)
        conditioned_state = self.self_condition(embeddings, previous)
        state = n.add_select(
            self.reshape(decoder, (1, 1)), conditioned_state, embeddings
        ).get_output(0)
        masks = self.attention_masks(positions, prefix, decoder, capacity, blocks)
        tables = {}
        for index, kind in enumerate(raw["layer_types"]):
            if kind not in tables:
                dim, _ = self.config.attention_geometry(index)
                tables[kind] = tuple(
                    self.const(value.astype(self.storage), self.dtype)
                    for value in rotary_tables(raw, kind, dim, capacity)
                )
            stem = f"model.decoder.layers.{index}"
            normalized = self.norm(state, stem + ".input_layernorm")
            attention = add_attention(
                self, normalized, positions, masks[kind], index, tables[kind], capacity
            )
            state = self.op(state, self.norm(attention, stem + ".post_attention_layernorm"), op.SUM)
            dense = self.mlp(self.norm(state, stem + ".pre_feedforward_layernorm"), stem + ".mlp")
            dense = self.norm(dense, stem + ".post_feedforward_layernorm_1")
            routed = add_experts(self, state, index)
            combined = self.norm(
                self.op(dense, routed, op.SUM), stem + ".post_feedforward_layernorm"
            )
            state = self.phase_scale(self.op(state, combined, op.SUM), index, decoder)
        # Encoder cache filling does not need a prompt-length vocabulary tensor.
        # The runtime selects one discarded row in that phase, all canvas rows
        # when denoising, and commits present_K/V ONLY after encoder execution.
        state = n.add_gather(self.norm(state, "model.decoder.norm"), indices, 0).get_output(0)
        logits = self.matmul(
            state, self.weight("model.decoder.embed_tokens.weight"), transpose=True
        )
        logits = self.cast(logits, trt.float32)
        cap = raw.get("final_logit_softcapping")
        if cap is not None:
            logits = self.scale(
                n.add_activation(self.scale(logits, 1 / cap), trt.ActivationType.TANH).get_output(
                    0
                ),
                cap,
            )
        logits.name = "logits"
        n.mark_output(logits)
        soft = self.next_soft_embeddings(logits, temperature)
        soft.name = "next_soft_embeddings"
        n.mark_output(soft)
