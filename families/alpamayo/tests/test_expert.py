# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-causal expert math, padding isolation, and reusable immutable prefix KV."""

import json
from pathlib import Path

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


def test_expert_fixture_provenance():
    provenance = json.loads((DATA / "expert.json").read_text())
    assert provenance["package"] == "transformers==4.57.1"
    assert provenance["revision"] == "v4.57.1"
    validate_fixture("expert.npz", provenance)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_expert_reuses_prefix_for_two_noncausal_steps(precision):
    from families.alpamayo.expert import ExpertGraph

    provenance = json.loads((DATA / "expert.json").read_text())
    config = DecoderConfig.parse(provenance["config"])
    trt, logger, builder, network = environment()
    with np.load(DATA / "expert.npz") as fixture:
        graph = ExpertGraph(network, FixtureWeights(fixture), config, precision)
        inputs = {}

        def add(name, dtype, shape):
            inputs[name] = network.add_input(name, dtype, shape)
            return inputs[name]

        hidden = add("hidden", trt.float32, (-1, 4, 32))
        mask = add("mask", trt.float32, (-1, 1, 4, 7))
        cosine = add("cosine", trt.float32, (-1, 1, 4, 8))
        sine = add("sine", trt.float32, (-1, 1, 4, 8))
        keys, values = [], []
        for index in range(2):
            keys.append(add(f"prefix_key_{index}", graph.dtype, (-1, 2, 3, 8)))
            values.append(add(f"prefix_value_{index}", graph.dtype, (-1, 2, 3, 8)))
        output = graph.forward(hidden, keys, values, mask, cosine, sine)
        output.name = "expert_hidden"
        network.mark_output(output)
        profile = builder.create_optimization_profile()
        for name, tensor in inputs.items():
            profile.set_shape(
                name, (1, *tensor.shape[1:]), (2, *tensor.shape[1:]), (2, *tensor.shape[1:])
            )
        options = builder.create_builder_config()
        options.clear_flag(trt.BuilderFlag.TF32)
        options.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, options)
        assert plan is not None
        tolerance = (
            dict(rtol=2e-5, atol=2e-6) if precision == "fp32" else dict(rtol=2e-3, atol=2e-4)
        )
        for count in (1, 2):
            for step in range(2):
                data = dict(
                    hidden=fixture[f"hidden_{step}"][:count],
                    mask=fixture["mask"][:count],
                    cosine=fixture[f"cosine_{count}"],
                    sine=fixture[f"sine_{count}"],
                )
                for index in range(2):
                    for kind in ("key", "value"):
                        name = f"prefix_{kind}_{index}"
                        data[name] = fixture[name][:count].astype(
                            np.float32 if precision == "fp32" else np.float16
                        )
                actual = execute(trt, logger, plan, data)
                np.testing.assert_allclose(
                    actual["expert_hidden"], fixture[f"{precision}_{count}_{step}"], **tolerance
                )
