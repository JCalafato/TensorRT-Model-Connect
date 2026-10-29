# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native visual-prefill and incremental language decoding with actual growing KV."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.alpamayo.config import DecoderConfig
from families.alpamayo.tests.test_action import (
    FixtureWeights,
    environment,
    execute,
    validate_fixture,
)


DATA = Path(__file__).parent / "data"


def test_text_fixture_provenance():
    provenance = json.loads((DATA / "text.json").read_text())
    assert provenance["package"] == "transformers==4.57.1"
    assert provenance["revision"] == "v4.57.1"
    validate_fixture("text.npz", provenance)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("last_logits_only", [False, True])
def test_visual_prefill_and_cached_decoding(precision, last_logits_only):
    from families.alpamayo.text_engine import build_text_engine

    provenance = json.loads((DATA / "text.json").read_text())
    config = SimpleNamespace(
        text=DecoderConfig.parse(provenance["config"]),
        vocab_size=provenance["config"]["vocab_size"],
        vision=SimpleNamespace(deepstack_visual_indexes=(0, 1, 2)),
    )
    trt, logger, _, _ = environment()
    with np.load(DATA / "text.npz") as fixture:
        plan = build_text_engine(
            config,
            FixtureWeights(fixture),
            precision,
            max_batch=2,
            input_capacity=5,
            prefix_capacity=6,
            workspace_bytes=1 << 28,
            last_logits_only=last_logits_only,
        )
        dtype = np.float32 if precision == "fp32" else np.float16
        tolerance = (
            dict(rtol=2e-5, atol=2e-6) if precision == "fp32" else dict(rtol=2e-3, atol=2e-4)
        )
        for count in (1, 2):
            cache = {
                f"past_{kind}_{i}": np.empty((count, 2, 0, 12), dtype)
                for kind in ("key", "value")
                for i in range(3)
            }
            for step in range(3):
                tag = f"{precision}_{count}_{step}"
                data = {
                    name: fixture[f"{name}_{step}"][:count]
                    for name in ("tokens", "visual_mask", "mask")
                }
                data["visual_mask"] = data["visual_mask"].astype(np.int32)
                data.update(
                    {
                        name: fixture[f"{name}_{step}"][:count].astype(dtype)
                        for name in ("visual", "deepstack_0", "deepstack_1", "deepstack_2")
                    }
                )
                data.update({name: fixture[f"{tag}_{name}"] for name in ("cosine", "sine")})
                actual = execute(trt, logger, plan, {**data, **cache})
                expected = fixture[f"{tag}_logits"]
                if last_logits_only:
                    expected = expected[:, -1:]
                np.testing.assert_allclose(actual["logits"], expected, **tolerance)
                for kind in ("key", "value"):
                    for i in range(3):
                        name = f"{kind}_{i}"
                        np.testing.assert_allclose(
                            actual[name], fixture[f"{tag}_{name}"], **tolerance
                        )
                        cache["past_" + name] = actual[name]
