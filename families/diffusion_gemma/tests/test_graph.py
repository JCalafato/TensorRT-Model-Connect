# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real GPU components versus independent HF snapshots and exact mask contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.trt]
DATA = Path(__file__).parent / "data"


def _environment():
    import torch
    import tensorrt as trt

    if not torch.cuda.is_available():
        pytest.skip("DiffusionGemma mathematical comparison requires a CUDA GPU")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return trt, logger, builder, network


def _plan(trt, builder, network, shapes):
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    for name, shape in shapes.items():
        profile.set_shape(name, *(shape if isinstance(shape[0], tuple) else (shape, shape, shape)))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    return plan


def _execute(trt, logger, plan, inputs):
    import torch

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    buffers = {}
    for name, value in inputs.items():
        buffers[name] = torch.from_numpy(np.ascontiguousarray(value)).cuda()
        assert context.set_input_shape(name, value.shape)
        assert context.set_tensor_address(name, buffers[name].data_ptr())
    assert context.infer_shapes() == []
    outputs = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            dtype = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
            }[engine.get_tensor_dtype(name)]
            buffers[name] = torch.empty(
                tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            assert context.set_tensor_address(name, buffers[name].data_ptr())
            outputs.append(name)
    stream = torch.cuda.current_stream()
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return {name: buffers[name].cpu().numpy() for name in outputs}


def _config():
    from families.diffusion_gemma.config import ModelConfig

    config = ModelConfig.from_directory(DATA)
    config.text.update(hidden_size=16, intermediate_size=24, vocab_size=32, sliding_window=3)
    return config


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_conditioning_matches_independent_reference(tmp_path, precision):
    from safetensors.numpy import save_file
    from families.diffusion_gemma.graph import Graph
    from families.diffusion_gemma.weights import Checkpoint

    fixture = np.load(DATA / "conditioning-reference.npz")
    save_file(
        {
            key.removeprefix("weight__"): fixture[key]
            for key in fixture.files
            if key.startswith("weight__")
        },
        tmp_path / "model.safetensors",
    )
    trt, logger, builder, network = _environment()
    with Checkpoint(tmp_path) as weights:
        graph = Graph(network, _config(), weights, precision)
        tokens = network.add_input("tokens", trt.int32, (-1,))
        previous = network.add_input("previous", graph.dtype, (-1, 16))
        logits = network.add_input("logits", trt.float32, (-1, 32))
        temperature = network.add_input("temperature", trt.float32, (1, 1))
        values = {
            "conditioned": graph.self_condition(graph.token_embeddings(tokens), previous),
            "next_soft": graph.next_soft_embeddings(logits, temperature),
        }
        for name, value in values.items():
            value = graph.cast(value, trt.float32)
            value.name = name
            network.mark_output(value)
        plan = _plan(
            trt, builder, network, {"tokens": (3,), "previous": (3, 16), "logits": (3, 32)}
        )
    dtype = np.float32 if precision == "fp32" else np.float16
    for initial in (False, True):
        inputs = {
            "tokens": fixture["tokens"],
            "previous": (np.zeros_like(fixture["soft"]) if initial else fixture["soft"]).astype(
                dtype
            ),
            "logits": fixture["logits"],
            "temperature": np.array([[0.6]], np.float32),
        }
        result = _execute(trt, logger, plan, inputs)
        tolerance = (
            {"rtol": 2e-5, "atol": 2e-6} if precision == "fp32" else {"rtol": 2e-3, "atol": 2e-4}
        )
        expected = fixture[precision + ("_initial" if initial else "_conditioned")]
        np.testing.assert_allclose(result["conditioned"], expected, **tolerance)
        np.testing.assert_allclose(
            result["next_soft"], fixture[precision + "_next_soft"], **tolerance
        )


def test_encoder_and_canvas_masks_never_expose_empty_cache():
    from families.diffusion_gemma.graph import Graph

    trt, logger, builder, network = _environment()
    graph = Graph(network, _config(), None, "fp32")
    positions = network.add_input("positions", trt.int32, (-1,))
    prefix = network.add_input("prefix", trt.int32, (1,))
    decoder = network.add_input("decoder", trt.bool, (1,))
    blocks = network.add_input("blocks", trt.int32, (-1,))
    masks = graph.attention_masks(positions, prefix, decoder, 8, blocks)
    for name, value in masks.items():
        value.name = name
        network.mark_output(value)
    plan = _plan(trt, builder, network, {"positions": (3,), "blocks": (3,)})
    for prefix_length in (0, 2, 6):
        positions = np.arange(prefix_length, prefix_length + 3, dtype=np.int32)
        for is_decoder in (False, True):
            for image in (False, True):
                blocks = np.array([0, 1, 1] if image else [0, 0, 0], np.int32)
                result = _execute(
                    trt,
                    logger,
                    plan,
                    {
                        "positions": positions,
                        "prefix": np.array([prefix_length], np.int32),
                        "decoder": np.array([is_decoder]),
                        "blocks": blocks,
                    },
                )
                for kind in ("full_attention", "sliding_attention"):
                    expected = np.full((3, 11), -np.inf, np.float32)
                    for row, query in enumerate(positions):
                        for column, key in enumerate([*range(8), *positions]):
                            valid = column >= 8 or column < prefix_length
                            allowed = is_decoder or key <= query
                            if kind == "sliding_attention":
                                if is_decoder:
                                    allowed = key >= prefix_length - 3
                                else:
                                    allowed = key > query - 3 and allowed
                            if not is_decoder:
                                # HF overlays the same-image block after its
                                # causal/window mask, for both attention types.
                                same_image = (
                                    column >= 8
                                    and blocks[row] > 0
                                    and blocks[row] == blocks[column - 8]
                                )
                                allowed = allowed or same_image
                            if valid and allowed:
                                expected[row, column] = 0
                    np.testing.assert_array_equal(result[kind][0, 0], expected)


def test_reference_fixture_provenance():
    provenance = json.loads((DATA / "conditioning-reference.json").read_text())
    assert provenance["package"] == "transformers==5.11.0"
    assert provenance["source_revision"] == "v5.11.0"
    assert provenance["fixture_file"] == "conditioning-reference.npz"
    with np.load(DATA / provenance["fixture_file"], allow_pickle=False) as reference:
        assert set(reference.files) == set(provenance["arrays"])
        for name, layout in provenance["arrays"].items():
            assert list(reference[name].shape) == layout["shape"]
            assert str(reference[name].dtype) == layout["dtype"]
            assert np.isfinite(reference[name]).all()


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_text_encoder_and_two_denoising_steps_match_hf(precision):
    from families.diffusion_gemma.config import ModelConfig
    from families.diffusion_gemma.text import TextGraph
    from families.diffusion_gemma.weights import Checkpoint

    directory = DATA / "tiny-text"
    config = ModelConfig.from_directory(directory)
    reference = np.load(directory / "reference.npz")
    trt, logger, builder, network = _environment()
    with Checkpoint(directory) as weights:
        graph = TextGraph(network, config, weights, precision)
        graph.build(16)
        shapes = {
            name: ((1,), (3,), (6,)) for name in ("token_id", "position_id", "logits_indices")
        }
        shapes["previous_soft_embeddings"] = ((1, 16), (3, 16), (6, 16))
        plan = _plan(trt, builder, network, shapes)
    dtype = np.float32 if precision == "fp32" else np.float16
    inputs = {
        "token_id": reference["prompt"],
        "position_id": np.arange(6, dtype=np.int32),
        "logits_indices": np.arange(6, dtype=np.int32),
        "cache_length": np.array([0], np.int32),
        "decoder_phase": np.array([0], np.int32),
        "temperature": np.array([[0.6]], np.float32),
        "previous_soft_embeddings": np.zeros((6, 16), dtype),
    }
    for index in range(2):
        dim, heads = config.attention_geometry(index)
        for kind in ("k", "v"):
            inputs[f"cache_{kind}_{index}"] = np.zeros((16, dim * heads), dtype)
    tolerance = {"rtol": 2e-3, "atol": 2e-4}
    prefill = _execute(trt, logger, plan, inputs)
    np.testing.assert_allclose(prefill["logits"], reference[precision + "_prefill"], **tolerance)
    for index in range(2):
        dim, heads = config.attention_geometry(index)
        for kind, label in (("k", "keys"), ("v", "values")):
            present = prefill[f"present_{kind}_{index}"]
            shaped = present.reshape(6, heads, dim).transpose(1, 0, 2)[None]
            expected = reference[f"{precision}_cache_{label}_{index}"]
            if index == 0:
                np.testing.assert_allclose(shaped[:, :, -3:], expected, **tolerance)
            else:
                np.testing.assert_allclose(shaped, expected[:, :, :6], **tolerance)
            inputs[f"cache_{kind}_{index}"][:6] = present
    inputs.update(
        cache_length=np.array([6], np.int32),
        decoder_phase=np.array([1], np.int32),
        position_id=np.arange(6, 9, dtype=np.int32),
        logits_indices=np.arange(3, dtype=np.int32),
        previous_soft_embeddings=np.zeros((3, 16), dtype),
    )
    for step in range(2):
        inputs["token_id"] = reference["canvas"] if step == 0 else reference["canvas"][::-1].copy()
        result = _execute(trt, logger, plan, inputs)
        np.testing.assert_allclose(
            result["logits"], reference[f"{precision}_denoise_{step}"], **tolerance
        )
        np.testing.assert_allclose(
            result["next_soft_embeddings"], reference[f"{precision}_soft_{step}"], **tolerance
        )
        # Denoising never commits its present K/V to the encoder's cache.
        inputs["previous_soft_embeddings"] = result["next_soft_embeddings"]
