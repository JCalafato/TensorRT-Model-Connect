# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a tiny fixture from the unmodified, pinned NVlabs action module.

Run explicitly with --source pointing to src/alpamayo_r1/models/action_in_proj.py
from NVlabs/alpamayo revision 11a0e01c13a5622377c45ee37d653351453ec43b.
The fixture generator has no imports from the Model Connect implementation.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


REVISION = "11a0e01c13a5622377c45ee37d653351453ec43b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.source.resolve().parent),
            "show",
            REVISION + ":src/alpamayo_r1/models/action_in_proj.py",
        ]
    )
    if args.source.read_bytes() != expected:
        raise ValueError("Projection source differs from the pinned repository revision")
    spec = importlib.util.spec_from_file_location("alpamayo_provider_projection", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    projection = module.PerWaypointActionInProjV2(
        [4, 2], 16, num_enc_layers=2, hidden_size=12, num_fourier_feats=20
    )
    output = torch.nn.Linear(16, 2)
    # Nontrivial learned norms catch missing affine operations and incorrect ordering.
    with torch.no_grad():
        for name, parameter in projection.named_parameters():
            if parameter.ndim == 1 and name.endswith("weight"):
                parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel()))
        projection.norm.bias.copy_(torch.linspace(-0.2, 0.2, 16))
    arrays = {}
    for stem, model in (("action_in_proj", projection), ("action_out_proj", output)):
        for name, value in model.state_dict().items():
            arrays["weight__" + stem + "." + name] = value.numpy()
    arrays["actions"] = torch.randn(2, 4, 2).numpy()
    arrays["timesteps"] = np.array([0.0, 0.73], np.float32).reshape(2, 1, 1)
    arrays["hidden"] = torch.randn(2, 4, 16).numpy()
    for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        current = copy.deepcopy(projection).cuda().to(dtype)
        head = copy.deepcopy(output).cuda().to(dtype)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"),
        ):
            for count in (1, 2):
                actions = torch.from_numpy(arrays["actions"][:count]).cuda()
                timesteps = torch.from_numpy(arrays["timesteps"][:count]).cuda()
                hidden = torch.from_numpy(arrays["hidden"][:count]).cuda().to(dtype)
                arrays[f"{precision}_{count}_input"] = (
                    current(actions, timesteps).float().cpu().numpy()
                )
                arrays[f"{precision}_{count}_output"] = head(hidden).float().cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.output / "projection.npz"
    np.savez_compressed(fixture, **arrays)
    provenance = {
        "repository": "https://github.com/NVlabs/alpamayo",
        "revision": REVISION,
        "source": "src/alpamayo_r1/models/action_in_proj.py",
        "fixture_file": fixture.name,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "torch": torch.__version__,
        "seed": 731,
        "device": torch.cuda.get_device_name(),
        "tf32": False,
        "precision": ["fp32", "fp16 CUDA autocast"],
        "scope": "Synthetic action projections only; not trajectory/model qualification",
    }
    (args.output / "projection.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
