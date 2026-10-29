# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate an independent tiny expert with immutable, partially masked prefix KV."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "4.57.1":
        raise ValueError("Alpamayo reference requires transformers==4.57.1")
    torch.manual_seed(732)
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = dict(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=16,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        hidden_act="silu",
        attention_bias=False,
        rope_scaling=dict(rope_type="default", mrope_section=[2, 1, 1], mrope_interleaved=True),
    )
    config = Qwen3VLTextConfig(**raw)
    config._attn_implementation = "eager"
    model = Qwen3VLTextModel(config).eval()
    del model.embed_tokens
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.normal_(0, 0.12)
    arrays = {"weight__expert." + name: value.numpy() for name, value in model.state_dict().items()}
    arrays.update({f"hidden_{step}": torch.randn(2, 4, 32).numpy() for step in range(2)})
    for layer in range(2):
        for kind in ("key", "value"):
            arrays[f"prefix_{kind}_{layer}"] = torch.randn(2, 2, 3, 8).numpy()
    positions = torch.arange(4)[None, :] + torch.tensor([[10], [8]])
    mask = torch.zeros(2, 1, 4, 7)
    mask[1, :, :, 1:3] = torch.finfo(torch.float32).min
    arrays["mask"] = mask.numpy()
    for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        current = copy.deepcopy(model).cuda().to(dtype)
        for count in (1, 2):
            cache = DynamicCache()
            for layer in range(2):
                key = torch.from_numpy(arrays[f"prefix_key_{layer}"][:count]).cuda().to(dtype)
                value = torch.from_numpy(arrays[f"prefix_value_{layer}"][:count]).cuda().to(dtype)
                cache.update(key, value, layer)
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"),
            ):
                for step in range(2):
                    hidden = torch.from_numpy(arrays[f"hidden_{step}"][:count]).cuda()
                    pos = positions[:count].cuda()[None].expand(3, -1, -1)
                    cosine, sine = current.rotary_emb(hidden, pos)
                    arrays[f"cosine_{count}"] = cosine[:, None].cpu().numpy()
                    arrays[f"sine_{count}"] = sine[:, None].cpu().numpy()
                    result = current(
                        inputs_embeds=hidden,
                        position_ids=pos,
                        attention_mask=mask[:count].cuda(),
                        past_key_values=cache,
                        use_cache=True,
                        is_causal=False,
                    )
                    arrays[f"{precision}_{count}_{step}"] = (
                        result.last_hidden_state.float().cpu().numpy()
                    )
                    assert cache.get_seq_length() == 7
                    cache.crop(3)
                    assert cache.get_seq_length() == 3
                    for layer in range(2):
                        np.testing.assert_array_equal(
                            cache.layers[layer].keys.float().cpu().numpy(),
                            arrays[f"prefix_key_{layer}"][:count]
                            .astype(np.float32 if precision == "fp32" else np.float16)
                            .astype(np.float32),
                        )
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "expert.npz"
    np.savez_compressed(path, **arrays)
    provenance = dict(
        package="transformers==4.57.1",
        torch=torch.__version__,
        seed=732,
        repository="https://github.com/huggingface/transformers",
        revision="v4.57.1",
        source="src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
        fixture_file=path.name,
        arrays={
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        config=raw,
        attention="eager, explicit non-causal additive mask",
        tf32=False,
        scope="Synthetic expert with two action steps and cropped immutable prefix; not full-model validation",
    )
    (args.output / "expert.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
