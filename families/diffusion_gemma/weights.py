# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact provider tensor access, including byte-preserved prepared FP4/FP8."""

import json
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open


class Checkpoint:
    """Own validated readers without expanding the expert bank in host memory."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.readers = {}
        self.headers = {}
        self.offsets = {}
        single = self.directory / "model.safetensors"
        index = self.directory / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text()).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("DiffusionGemma safetensors index has no weight_map")
        elif single.is_file():
            with safe_open(single, framework="numpy") as reader:
                mapping = dict.fromkeys(reader.keys(), single.name)
        else:
            raise FileNotFoundError("DiffusionGemma checkpoint has no safetensors weights")
        try:
            for shard in sorted(set(mapping.values())):
                path = (self.directory / shard).resolve()
                if not path.is_relative_to(self.directory):
                    raise ValueError("DiffusionGemma shard must stay within its checkpoint")
                reader = safe_open(path, framework="numpy")
                self.readers[shard] = reader
                with path.open("rb") as source:
                    length = int.from_bytes(source.read(8), "little")
                    self.headers[shard] = json.loads(source.read(length))
                self.offsets[shard] = 8 + length
            for name, shard in mapping.items():
                if name not in self.headers[shard] or name == "__metadata__":
                    raise ValueError(f"DiffusionGemma indexed tensor is missing: {name}")
        except BaseException:
            self.close()
            raise
        self.tensor_map = mapping

    def close(self):
        for reader in self.readers.values():
            reader.__exit__(None, None, None)
        self.readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def has(self, name):
        return name in self.tensor_map

    def raw(self, name):
        shard = self.tensor_map[name]
        metadata = self.headers[shard][name]
        # safe_open validates shape, offsets and bounds before this view. NumPy
        # safetensors does not expose these dtypes on every supported version.
        special = {"F8_E4M3": ml_dtypes.float8_e4m3fn, "BF16": ml_dtypes.bfloat16}
        if metadata["dtype"] in special:
            return np.memmap(
                self.directory / shard,
                mode="r",
                dtype=special[metadata["dtype"]],
                offset=self.offsets[shard] + metadata["data_offsets"][0],
                shape=tuple(metadata["shape"]),
            )
        return self.readers[shard].get_tensor(name)

    def floating(self, name):
        value = self.raw(name)
        if value.dtype == np.uint8:
            raise ValueError(f"Packed DiffusionGemma tensor requires explicit decoding: {name}")
        return np.asarray(value, dtype=np.float32)

    def layer_scale_name(self, layer, *, decoder):
        # The provider shares text weights but not the two phase scalars.
        prefix = "model.decoder" if decoder else "model.encoder.language_model"
        return f"{prefix}.layers.{layer}.layer_scalar"
