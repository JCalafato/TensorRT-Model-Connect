# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy checkpoint access and family-owned Alpamayo tensor namespaces."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def _canonical(source: str) -> str:
    for prefix, canonical in (
        ("vlm.model.language_model.", "text."),
        ("vlm.model.visual.", "vision."),
        ("vlm.lm_head.", "text.lm_head."),
    ):
        if source.startswith(prefix):
            return canonical + source[len(prefix) :]
    if source.startswith(("expert.", "action_in_proj.", "action_out_proj.", "action_space.")):
        return source
    raise ValueError(f"Unrecognized Alpamayo tensor namespace: {source}")


class Weights:
    """Open each shard once and convert only tensors requested by a component.

    Arrays preserve provider layouts; the owning graph selects transposes.
    Physical normalization buffers are read from the checkpoint, not replaced
    with rounded constants from an evaluation script.
    """

    def __init__(self, model_dir: str | Path):
        self.root = Path(model_dir)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self._sources: dict[str, tuple[str, str]] = {}
        for source, shard in index["weight_map"].items():
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError(f"Invalid checkpoint shard name: {shard}")
            name = _canonical(source)
            if name in self._sources:
                raise ValueError(f"Duplicate canonical tensor: {name}")
            self._sources[name] = (source, shard)
        if not self._sources:
            raise ValueError("Empty Alpamayo checkpoint index")
        self._stack = ExitStack()
        self._handles: dict = {}
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self._closed = False

    def __enter__(self) -> Weights:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()
        self._arrays.clear()
        self._closed = True

    def release(self) -> None:
        """Release converted arrays after the owning engine has been serialized."""
        self._arrays.clear()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources))

    def source_name(self, name: str) -> str:
        return self._sources[name][0]

    def _handle(self, name: str):
        if self._closed:
            raise RuntimeError("Checkpoint reader is closed")
        source, shard = self._sources[name]
        if shard not in self._handles:
            self._handles[shard] = self._stack.enter_context(
                safe_open(self.root / shard, framework="pt", device="cpu")
            )
        return source, self._handles[shard]

    def info(self, name: str) -> tuple[str, tuple[int, ...]]:
        source, handle = self._handle(name)
        view = handle.get_slice(source)
        return view.get_dtype(), tuple(view.get_shape())

    def array(self, name: str, dtype=np.float32) -> np.ndarray:
        import torch

        target = np.dtype(dtype)
        if target not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("Native Alpamayo weights require float16 or float32 arrays")
        source, handle = self._handle(name)
        key = (name, target.str)
        if key not in self._arrays:
            tensor = handle.get_tensor(source)
            if not tensor.is_floating_point():
                raise ValueError(f"Expected a floating-point Alpamayo tensor: {source}")
            torch_dtype = torch.float16 if target == np.dtype(np.float16) else torch.float32
            self._arrays[key] = tensor.to(dtype=torch_dtype, copy=True).contiguous().numpy()
        return self._arrays[key]
