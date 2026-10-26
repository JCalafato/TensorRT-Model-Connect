# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the reasoner's flat text and visual keys without loading policy tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class CheckpointWeights:
    """Lazy access through the checkpoint's reasoning safetensors index.

    Text tensors reside inside the diffusion transformer shards. Iterating every
    tensor in those shards would also load unrelated generation-policy weights.
    The root index explicitly names only reasoning tensors and their shard files.
    """

    def __init__(self, model_dir: Path):
        self.root = Path(model_dir).resolve()
        document = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.mapping = document.get("weight_map")
        if not isinstance(self.mapping, dict) or not self.mapping:
            raise ValueError("Cosmos3-Edge requires a nonempty reasoning weight_map")
        for key, shard in self.mapping.items():
            if not isinstance(key, str) or not isinstance(shard, str):
                raise ValueError("Cosmos3-Edge reasoning weight_map entries must be strings")
            relative = Path(shard)
            path = (self.root / relative).resolve()
            # Standard HF snapshots store shards as links into their own blobs
            # directory. Accept that cache layout, not arbitrary external links.
            cache_blob = (self.root.parent.name == "snapshots" and
                          path.is_relative_to((self.root.parent.parent / "blobs").resolve()))
            if (relative.is_absolute() or relative.as_posix() != shard
                    or any(part in {".", ".."} for part in relative.parts)
                    or "\\" in shard or "\0" in shard or not path.is_file()
                    or not (path.is_relative_to(self.root) or cache_blob)):
                raise ValueError(f"Cosmos3-Edge missing or non-contained checkpoint shard: {shard}")

    def __getitem__(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        # Convert BF16 directly to FP32, never via FP16 (which has a smaller
        # exponent range). The graph owns the requested computation dtype.
        with safe_open(self.root / self.mapping[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key).float().numpy()
