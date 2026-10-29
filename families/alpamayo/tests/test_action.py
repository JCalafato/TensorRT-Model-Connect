# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action projection GPU math against the pinned independent provider fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from families.alpamayo.config import ActionConfig


DATA = Path(__file__).parent / "data"


class FixtureWeights:
    def __init__(self, arrays):
        self.arrays = arrays

    def info(self, name):
        return "F32", self.arrays["weight__" + name].shape

    def array(self, name, dtype):
        return self.arrays["weight__" + name].astype(dtype)


def environment():
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Action projection execution requires a CUDA GPU")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return trt, logger, builder, network


def execute(trt, logger, plan, inputs):
    import torch

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    buffers = {}
    for name, value in inputs.items():
        buffers[name] = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        # TensorRT requires a valid address even for an empty prefill cache.
        if value.size == 0:
            buffers[name] = torch.empty(1, dtype=buffers[name].dtype, device="cuda")
        assert context.set_input_shape(name, value.shape)
        assert context.set_tensor_address(name, buffers[name].data_ptr())
    assert context.infer_shapes() == []
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            dtype = {trt.float32: torch.float32, trt.float16: torch.float16}[
                engine.get_tensor_dtype(name)
            ]
            buffers[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            assert context.set_tensor_address(name, buffers[name].data_ptr())
            outputs.append(name)
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: buffers[name].cpu().numpy() for name in outputs}


def build(weights, config, expert_width, precision, max_batch=2):
    from families.alpamayo.action import ActionGraph

    trt, logger, builder, network = environment()
    graph = ActionGraph(network, weights, config, expert_width, precision)
    waypoints = config.num_waypoints
    actions = network.add_input("actions", trt.float32, (-1, waypoints, 2))
    time = network.add_input("timesteps", trt.float32, (-1, 1, 1))
    hidden = network.add_input("hidden", graph.dtype, (-1, waypoints, expert_width))
    for name, value in (
        ("projected", graph.input_projection(actions, time)),
        ("velocity", graph.output_projection(hidden)),
    ):
        value = graph.cast(value, trt.float32)
        value.name = name
        network.mark_output(value)
    profile = builder.create_optimization_profile()
    for name, tail in {
        "actions": (waypoints, 2),
        "timesteps": (1, 1),
        "hidden": (waypoints, expert_width),
    }.items():
        profile.set_shape(name, (1, *tail), (max_batch, *tail), (max_batch, *tail))
    options = builder.create_builder_config()
    options.clear_flag(trt.BuilderFlag.TF32)
    options.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, options)
    assert plan is not None
    return trt, logger, plan


def test_projection_fixture_provenance():
    provenance = json.loads((DATA / "projection.json").read_text())
    assert provenance["revision"] == "11a0e01c13a5622377c45ee37d653351453ec43b"
    validate_fixture("projection.npz", provenance)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_action_projections_match_provider(precision):
    with np.load(DATA / "projection.npz") as fixture:
        config = ActionConfig(4, 0.1, 48, 8, 12, 2, 20, 100.0)
        trt, logger, plan = build(FixtureWeights(fixture), config, 16, precision)
        dtype = np.float32 if precision == "fp32" else np.float16
        tolerance = (
            {"rtol": 2e-5, "atol": 2e-6} if precision == "fp32" else {"rtol": 2e-3, "atol": 2e-4}
        )
        for count in (1, 2):
            inputs = {
                name: fixture[name][:count].astype(dtype if name == "hidden" else np.float32)
                for name in ("actions", "timesteps", "hidden")
            }
            result = execute(trt, logger, plan, inputs)
            np.testing.assert_allclose(
                result["projected"], fixture[f"{precision}_{count}_input"], **tolerance
            )
            np.testing.assert_allclose(
                result["velocity"], fixture[f"{precision}_{count}_output"], **tolerance
            )


def validate_fixture(name, provenance):
    """Check the pinned reference structure, independently of native outputs."""
    assert provenance["fixture_file"] == name
    with np.load(DATA / name, allow_pickle=False) as reference:
        assert set(reference.files) == set(provenance["arrays"])
        for key, layout in provenance["arrays"].items():
            assert list(reference[key].shape) == layout["shape"]
            assert str(reference[key].dtype) == layout["dtype"]
            assert np.isfinite(reference[key]).all()
