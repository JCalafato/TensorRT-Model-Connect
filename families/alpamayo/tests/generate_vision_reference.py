# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze independent visual and DeepStack outputs from the pinned provider."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def interpolation_inputs(grids, side, merge):
    """Express the provider's learned-grid interpolation in packed patch order."""
    indices, weights = [], []
    for t, h, w in grids:
        y, x = torch.linspace(0, side - 1, h), torch.linspace(0, side - 1, w)
        y0, x0 = y.int(), x.int()
        y1, x1 = (y0 + 1).clamp(max=side - 1), (x0 + 1).clamp(max=side - 1)
        dy, dx = y - y0, x - x0
        ids = torch.stack(
            [
                (yy[:, None] * side + xx[None, :])
                for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
            ]
        )
        coefficients = torch.stack(
            [
                yy[:, None] * xx[None, :]
                for yy, xx in ((1 - dy, 1 - dx), (1 - dy, dx), (dy, 1 - dx), (dy, dx))
            ]
        )

        def order(value):
            return (
                value[:, None]
                .expand(4, t, h, w)
                .reshape(4, t, h // merge, merge, w // merge, merge)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(4, -1)
            )

        indices.append(order(ids))
        weights.append(order(coefficients))
    return torch.cat(indices, 1).numpy().astype(np.int32), torch.cat(weights, 1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "4.57.1":
        raise ValueError("Alpamayo reference requires transformers==4.57.1")
    torch.manual_seed(734)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = dict(
        depth=3,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=32,
        deepstack_visual_indexes=[0, 1, 2],
        hidden_act="gelu_pytorch_tanh",
    )
    config = Qwen3VLVisionConfig(**raw)
    config._attn_implementation = "eager"
    model = Qwen3VLVisionModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, 0.08)
    arrays = {"weight__vision." + name: value.numpy() for name, value in model.state_dict().items()}
    modes = {"single": [[1, 4, 6]], "mixed": [[1, 2, 4], [2, 6, 2]]}
    for mode, grids in modes.items():
        count = sum(t * h * w for t, h, w in grids)
        arrays[f"{mode}_pixels"] = torch.randn(count, 24).numpy() * 0.4
        indices, weights = interpolation_inputs(grids, 4, 2)
        arrays[f"{mode}_position_indices"], arrays[f"{mode}_position_weights"] = indices, weights
        mask = np.full((1, count, count), np.finfo(np.float32).min, np.float32)
        offset = 0
        for t, h, w in grids:
            for _ in range(t):
                length = h * w
                mask[:, offset : offset + length, offset : offset + length] = 0
                offset += length
        arrays[f"{mode}_mask"] = mask
    for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        current = copy.deepcopy(model).cuda().to(dtype)
        # HF loading converts trained parameters without rounding the rotary
        # module's nonpersistent FP32 frequency table.
        current.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.cuda()
        for mode, grids in modes.items():
            grid = torch.tensor(grids, device="cuda")
            pixels = torch.from_numpy(arrays[f"{mode}_pixels"]).cuda()
            with torch.inference_mode():
                indexes = torch.from_numpy(arrays[f"{mode}_position_indices"]).cuda().long()
                weights = torch.from_numpy(arrays[f"{mode}_position_weights"]).cuda().to(dtype)
                terms = current.pos_embed(indexes) * weights[:, :, None]
                position = terms[0] + terms[1] + terms[2] + terms[3]
                torch.testing.assert_close(
                    position, current.fast_pos_embed_interpolate(grid), rtol=0, atol=0
                )
                phases = current.rot_pos_emb(grid)
                phases = torch.cat((phases, phases), -1)
                for name, value in (("cosine", phases.cos()), ("sine", phases.sin())):
                    value = value[:, None].cpu().numpy()
                    key = f"{mode}_{name}"
                    if key in arrays:
                        np.testing.assert_array_equal(arrays[key], value)
                    arrays[key] = value
                features, deepstack = current(pixels, grid_thw=grid)
                arrays[f"{precision}_{mode}_features"] = features.float().cpu().numpy()
                for index, value in enumerate(deepstack):
                    arrays[f"{precision}_{mode}_deepstack_{index}"] = value.float().cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "vision.npz"
    np.savez_compressed(path, **arrays)
    provenance = dict(
        package="transformers==4.57.1",
        torch=torch.__version__,
        seed=734,
        config=raw,
        grids=modes,
        repository="https://github.com/huggingface/transformers",
        revision="v4.57.1",
        source="src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
        fixture_file=path.name,
        arrays={
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        tf32=False,
        scope="Synthetic packed patches, two spatial layouts, three frame chunks, and all DeepStack outputs. Not raw-image preprocessing or full-model qualification.",
    )
    (args.output / "vision.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
