# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze independent VLM prefill, DeepStack, and two cached decoding steps."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "4.57.1":
        raise ValueError("Alpamayo reference requires transformers==4.57.1")
    torch.manual_seed(733)
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = dict(
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        vocab_size=37,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        hidden_act="silu",
        attention_bias=False,
        rope_scaling=dict(rope_type="default", mrope_section=[2, 2, 2], mrope_interleaved=True),
    )
    config = Qwen3VLTextConfig(**raw)
    config._attn_implementation = "eager"
    model = Qwen3VLTextModel(config).eval()
    head = torch.nn.Linear(48, 37, bias=False).eval()
    with torch.no_grad():
        for parameter in [*model.parameters(), *head.parameters()]:
            if parameter.ndim == 2:
                parameter.normal_(0, 0.08)
    arrays = {"weight__text." + name: value.numpy() for name, value in model.state_dict().items()}
    arrays["weight__text.lm_head.weight"] = head.weight.detach().numpy()
    start = 0
    for step, length in enumerate((5, 1, 2)):
        arrays[f"tokens_{step}"] = torch.randint(0, 37, (2, length)).numpy().astype(np.int32)
        visual_mask = np.zeros((2, length, 1), bool)
        if step == 0:
            visual_mask[0, [1, 3], 0] = True
            visual_mask[1, [2, 4], 0] = True
        arrays[f"visual_mask_{step}"] = visual_mask
        arrays[f"visual_{step}"] = torch.randn(2, length, 48).numpy() * visual_mask
        for layer in range(3):
            arrays[f"deepstack_{layer}_{step}"] = (
                torch.randn(2, length, 48).numpy() * visual_mask * 0.1
            )
        positions = np.broadcast_to(np.arange(start, start + length), (3, 2, length)).copy()
        # Distinct T/H/W coordinates exercise the actual interleaved rotary map.
        positions[1] += 2
        positions[2] += 5
        positions[:, 1] += 7
        arrays[f"positions_{step}"] = positions
        causal = np.arange(start + length)[None, :] > np.arange(start, start + length)[:, None]
        arrays[f"mask_{step}"] = np.broadcast_to(
            np.where(causal, np.finfo(np.float32).min, 0).astype(np.float32),
            (2, 1, length, start + length),
        ).copy()
        start += length
    for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        current = copy.deepcopy(model).cuda().to(dtype)
        current.rotary_emb.inv_freq = model.rotary_emb.inv_freq.cuda()
        current_head = copy.deepcopy(head).cuda().to(dtype)
        for count in (1, 2):
            cache = None
            for step in range(3):

                def tensor(name, dtype=None):
                    result = torch.from_numpy(arrays[f"{name}_{step}"][:count]).cuda()
                    return result.to(dtype) if dtype else result

                with torch.inference_mode():
                    tokens = tensor("tokens", torch.int64)
                    visual_mask = tensor("visual_mask").squeeze(-1)
                    hidden = current.embed_tokens(tokens)
                    hidden[visual_mask] = tensor("visual", dtype)[visual_mask]
                    positions = torch.from_numpy(arrays[f"positions_{step}"][:, :count]).cuda()
                    cosine, sine = current.rotary_emb(hidden, positions)
                    arrays[f"{precision}_{count}_{step}_cosine"] = cosine[:, None].cpu().numpy()
                    arrays[f"{precision}_{count}_{step}_sine"] = sine[:, None].cpu().numpy()
                    deepstack = [
                        tensor(f"deepstack_{layer}", dtype)[visual_mask] for layer in range(3)
                    ]
                    result = current(
                        inputs_embeds=hidden,
                        position_ids=positions,
                        past_key_values=cache,
                        use_cache=True,
                        visual_pos_masks=visual_mask,
                        deepstack_visual_embeds=deepstack if step == 0 else None,
                    )
                    cache = result.past_key_values
                    arrays[f"{precision}_{count}_{step}_logits"] = (
                        current_head(result.last_hidden_state).float().cpu().numpy()
                    )
                    for layer in range(3):
                        for kind, value in (
                            ("key", cache.layers[layer].keys),
                            ("value", cache.layers[layer].values),
                        ):
                            arrays[f"{precision}_{count}_{step}_{kind}_{layer}"] = (
                                value.cpu().numpy()
                            )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "text.npz"
    np.savez_compressed(path, **arrays)
    provenance = dict(
        package="transformers==4.57.1",
        torch=torch.__version__,
        seed=733,
        config=raw,
        repository="https://github.com/huggingface/transformers",
        revision="v4.57.1",
        source="src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
        fixture_file=path.name,
        arrays={
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        tf32=False,
        scope="Synthetic VLM: visual prefill, three DeepStack levels, distinct T/H/W positions, and two cached decode steps. Not full-model qualification.",
    )
    (args.output / "text.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
