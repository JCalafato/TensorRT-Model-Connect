# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual portable TRT QDQ versus independent ml_dtypes and Torch arithmetic."""

import ml_dtypes
import numpy as np
import pytest

from .test_graph import _config, _environment, _execute, _plan

pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _source_qdq(values, alpha):
    shape = values.shape
    grouped = values.astype(np.float32).reshape(shape[0], shape[1] // 16, 16)
    maximum = np.max(np.abs(grouped), axis=-1, keepdims=True)
    block = np.clip(maximum / np.float32(6) / np.float32(alpha), 0, 448)
    block = block.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scale = block * np.float32(alpha)
    normalized = grouped / np.where(scale > 0, scale, np.float32(1))
    rounded = np.clip(normalized, -6, 6).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    return (rounded * scale).reshape(shape).astype(values.dtype)


def test_portable_nvfp4_rounding_matches_independent_scalar_types():
    from families.diffusion_gemma.graph import Graph
    from families.diffusion_gemma.quantization import round_e4m3_scale, round_e2m1

    trt, logger, builder, network = _environment()
    graph = Graph(network, _config(), None, "fp32")
    value = network.add_input("value", trt.float32, (-1,))
    for name, function in (("fp8", round_e4m3_scale), ("fp4", round_e2m1)):
        output = function(graph, value)
        output.name = name
        network.mark_output(output)
    scales = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    middles = (scales[:-1] + scales[1:]) * np.float32(0.5)
    fp4_midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    midpoints = np.r_[middles, fp4_midpoints]
    values = np.r_[
        scales,
        midpoints,
        np.nextafter(midpoints, np.float32(0)),
        np.nextafter(midpoints, np.float32(np.inf)),
        np.float32(10000),
    ].astype(np.float32)
    plan = _plan(trt, builder, network, {"value": values.shape})
    actual = _execute(trt, logger, plan, {"value": values})
    expected8 = np.minimum(values, 448).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expected4 = np.minimum(values, 6).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    np.testing.assert_array_equal(actual["fp8"], expected8)
    np.testing.assert_array_equal(actual["fp4"], expected4)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_source_calibrated_activation_qdq(precision):
    from families.diffusion_gemma.graph import Graph
    from families.diffusion_gemma.quantization import activation_qdq

    trt, logger, builder, network = _environment()
    graph = Graph(network, _config(), None, precision)
    value = network.add_input("value", graph.dtype, (-1, 32))
    scale = network.add_input("scale", trt.float32, (1,))
    output = graph.cast(activation_qdq(graph, value, scale), trt.float32)
    output.name = "decoded"
    network.mark_output(output)
    plan = _plan(trt, builder, network, {"value": (8, 32)})
    dtype = np.float32 if precision == "fp32" else np.float16
    values = np.random.default_rng(17).normal(size=(8, 32)).astype(dtype)
    values[0] = 0
    values[1] *= dtype(2**-14)
    for scale in (0.00390625, 0.071, 2.0):
        expected = _source_qdq(values, scale).astype(np.float32)
        actual = _execute(
            trt, logger, plan, {"value": values, "scale": np.array([scale], np.float32)}
        )
        np.testing.assert_allclose(actual["decoded"], expected, rtol=2e-6, atol=2e-7)
        # This proves the operation is not a silent weight-only fallback.
        assert not np.array_equal(expected[2:], values[2:])


def _packed_fixture(directory):
    import torch
    from safetensors.torch import save_file

    rng = np.random.default_rng(81)
    tensors = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            rows, columns = (16, 32) if projection == "down_proj" else (32, 16)
            stem = f"model.decoder.layers.0.experts.{expert}.{projection}"
            tensors[stem + ".weight"] = torch.from_numpy(
                rng.integers(0, 256, size=(rows, columns // 2), dtype=np.uint8)
            )
            block = rng.uniform(0.1, 1.5, size=(rows, columns // 16)).astype(
                ml_dtypes.float8_e4m3fn
            )
            tensors[stem + ".weight_scale"] = torch.from_numpy(block.astype(np.float32)).to(
                torch.float8_e4m3fn
            )
            tensors[stem + ".weight_scale_2"] = torch.tensor([0.03 + 0.005 * expert])
            tensors[stem + ".input_scale"] = torch.tensor(
                [0.021 + 0.013 * expert + (0.017 if projection == "up_proj" else 0)]
            )
    save_file(tensors, directory / "model.safetensors")


def test_packed_experts_preserve_all_source_scales(tmp_path):
    import torch
    from families.diffusion_gemma.graph import Graph
    from families.diffusion_gemma.quantization import PreparedExperts, validate_prepared_source
    from families.diffusion_gemma.weights import Checkpoint

    _packed_fixture(tmp_path)
    config = _config()
    config.text.update(num_experts=2, top_k_experts=1, moe_intermediate_size=32)
    trt, logger, builder, network = _environment()
    values = np.random.default_rng(9).normal(size=(3, 16)).astype(np.float16)
    down_values = np.random.default_rng(10).normal(size=(3, 32)).astype(np.float16)
    with Checkpoint(tmp_path) as source:
        validate_prepared_source(config, source)
        graph = Graph(network, config, source, "fp16")
        first = network.add_input("first", graph.dtype, (-1, 16))
        down = network.add_input("down", graph.dtype, (-1, 32))
        selected = network.add_input("selected", trt.int32, (1,))
        loop = network.add_loop()
        count = graph.reshape(graph.const(np.array([2], np.int32)), ())
        loop.add_trip_limit(count, trt.TripLimit.COUNT)
        expert_id = loop.add_iterator(
            graph.const(np.arange(2, dtype=np.int32)), 0, False
        ).get_output(0)
        chosen = graph.op(
            graph.reshape(expert_id, (1, 1)),
            graph.reshape(selected, (1, 1)),
            trt.ElementWiseOperation.EQUAL,
        )
        prepared = PreparedExperts(graph, "model.decoder.layers.0", loop)
        for name in ("gate_proj", "up_proj", "down_proj"):
            result = graph.cast(
                prepared.linear(down if name == "down_proj" else first, name), trt.float32
            )
            # Match the production accumulation loop. TRT11.1's concatenate
            # loop output triggers a compiler SSA error in this diagnostic;
            # select each expert in separate executions without losing any
            # per-expert/per-projection reference assertions below.
            initial = network.add_gather(
                graph.scale(graph.cast(first, trt.float32), 0.0),
                graph.const(np.zeros(result.shape[-1], np.int32)),
                1,
            ).get_output(0)
            accumulation = loop.add_recurrence(initial)
            contribution = network.add_select(
                chosen, result, graph.const(np.zeros((1, 1), np.float32))
            ).get_output(0)
            accumulation.set_input(
                1, graph.op(accumulation.get_output(0), contribution, trt.ElementWiseOperation.SUM)
            )
            output = loop.add_loop_output(accumulation.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
            output.get_output(0).name = name
            network.mark_output(output.get_output(0))
        plan = _plan(trt, builder, network, {"first": (3, 16), "down": (3, 32)})
        expected = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            outputs = []
            for expert in range(2):
                stem = f"model.decoder.layers.0.experts.{expert}.{name}"
                packed = source.raw(stem + ".weight")
                codes = np.stack([packed & 15, packed >> 4], axis=-1).reshape(packed.shape[0], -1)
                fp4 = codes.view(ml_dtypes.float4_e2m1fn).astype(np.float32)
                block = source.raw(stem + ".weight_scale").astype(np.float32)
                global_scale = source.floating(stem + ".weight_scale_2")[0]
                weights = (
                    (fp4.reshape(block.shape + (16,)) * (block * global_scale)[..., None])
                    .reshape(fp4.shape)
                    .astype(np.float16)
                )
                inputs = _source_qdq(
                    down_values if name == "down_proj" else values,
                    source.floating(stem + ".input_scale")[0],
                )
                outputs.append(
                    (torch.from_numpy(inputs).float() @ torch.from_numpy(weights).float().T)
                    .half()
                    .float()
                    .numpy()
                )
            expected[name] = np.stack(outputs)
    for expert in range(2):
        actual = _execute(
            trt,
            logger,
            plan,
            {"first": values, "down": down_values, "selected": np.array([expert], np.int32)},
        )
        for name, result in expected.items():
            np.testing.assert_allclose(actual[name], result[expert], rtol=2e-3, atol=2e-4)
