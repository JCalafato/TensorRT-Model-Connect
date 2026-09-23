# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_qwen38_marker_default_task_is_checkpoint_owned() -> None:
    family, support = resolve_family(
        ModelMetadata(
            {
                "model_type": "qwen3_5",
                "text_config": {"output_gate_type": "sigmoid"},
            },
            {},
        )
    )
    assert family == "qwen3_8"
    assert support.default_task == "text_generation"
    assert support.build_cli_module == "edge_llm.cli"


def _edge_cli_source(tmp_path):
    import json

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "text_config": {"output_gate_type": "sigmoid"},
    }))
    draft = tmp_path / "draft"
    draft.mkdir()
    return source, draft


def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch):
    from tensorrt_model_connect import build_cli
    from families.qwen3_8.edge_llm import dispatch
    from families.qwen3_8.edge_llm.config import Qwen38BuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, Qwen38BuildRequest)
        assert request.execution is execution
        assert execution.variant == "dspark"
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == [("draft", draft)]
        seen.append(request)
        writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    assert build_cli.main([
        "build", str(source), "--precision", "fp16", "-o", str(output),
        "--execution-variant", "dspark", "--companion", f"draft={draft}",
    ]) == 0
    assert len(seen) == 1
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "dspark", "--companion", "missing_separator"],
    ["--execution-variant", "dspark", "--companion", "=path"],
    ["--execution-variant", "dspark", "--companion", "draft="],
    ["--execution-variant", "dspark", "--companion", "draft=https://example.com/model"],
])
def test_bad_edge_cli_inputs_fail_before_backend(tmp_path, monkeypatch, options):
    import importlib
    from tensorrt_model_connect import build_cli

    core = importlib.import_module("tensorrt_model_connect.build")
    source, _ = _edge_cli_source(tmp_path)
    monkeypatch.setattr(core, "_select_backend", lambda *_: pytest.fail("backend touched"))
    monkeypatch.setattr(core, "BundleWriter", lambda *_: pytest.fail("writer created"))
    with pytest.raises(ValueError):
        build_cli.main(["build", str(source), "-o", str(tmp_path / "out"), *options])


def test_edge_cli_help_is_family_owned(tmp_path, capsys):
    from tensorrt_model_connect import build_cli

    source, _ = _edge_cli_source(tmp_path)
    with pytest.raises(SystemExit) as caught:
        build_cli.main(["build", str(source), "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--execution-variant {dspark}" in help_text
    assert "--companion" in help_text


def test_edge_request_preserves_fields_and_family_owner(tmp_path):
    import argparse
    from families.qwen3_8.edge_llm import cli
    from dataclasses import fields, replace
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_8.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_8", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    execution = BuildExecutionInputs("dspark", (NamedCheckpoint("draft", draft),))
    args = argparse.Namespace(command="build", execution_variant=None, companion=[])
    assert cli.prepare_build_request(request, args) is request
    extended = with_execution(request, execution)
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(request, field.name)
    with pytest.raises(ValueError, match="requires the qwen3_8 family"):
        with_execution(replace(request, family="other"), execution)
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("dspark", execution.checkpoints * 2)


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_cli_failure_preserves_existing_bundle(tmp_path, monkeypatch, failure):
    from tensorrt_model_connect import build_cli
    from families.qwen3_8.edge_llm import dispatch

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    output.write_bytes(b"previous publication")

    def fail(request, writer, execution):
        writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(dispatch, "build_paired", fail)
    with pytest.raises(type(failure)) as caught:
        build_cli.main([
            "build", str(source), "-o", str(output), "--execution-variant", "dspark",
            "--companion", f"draft={draft}",
        ])
    assert caught.value is failure
    assert output.read_bytes() == b"previous publication"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft", "pair.bundle", "target"]


def test_edge_pair_requires_draft_and_rechecks_local_inputs(tmp_path):
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_8.edge_llm.config import BuildExecutionInputs, NamedCheckpoint
    from families.qwen3_8.edge_llm.dispatch import build_paired

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_8", "text_generation", "fp16")
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(request, None, BuildExecutionInputs("dspark"))
    execution = BuildExecutionInputs("dspark", (NamedCheckpoint("draft", draft),))
    draft.rmdir()
    with pytest.raises(ValueError, match="existing local directory"):
        build_paired(request, None, execution)


@pytest.mark.parametrize("model_type", ["qwen38", "qwen3.8", "qwen3_8"])
def test_qwen38_aliases_register_the_same_family_cli(model_type):
    from families.qwen3_8.support import describe

    support = describe(ModelMetadata({"model_type": model_type}, {}))
    assert support.build_cli_module == "edge_llm.cli"


def _dspark_pair_request(tmp_path):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_8.edge_llm.config import BuildExecutionInputs, NamedCheckpoint

    source, draft = _edge_cli_source(tmp_path)
    layers = {"layer0": {"quant_algo": "FP8"}, "layer1": {"quant_algo": "NVFP4"}}
    base = {
        "model_type": "qwen3_5", "output_gate_type": "sigmoid",
        "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 8,
        "max_position_embeddings": 4096, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "quantized_layers": layers, "kv_cache_scheme": {"type": "float"},
        },
    }
    (source / "config.json").write_text(json.dumps(base))
    (source / "hf_quant_config.json").write_text(json.dumps({"quantization": {
        "quant_algo": "MIXED_PRECISION", "quantized_layers": layers,
        "kv_cache_quant_algo": "FP8",
    }}))
    (draft / "config.json").write_text(json.dumps({
        "architectures": ["DSparkDraftModel"], "hidden_size": 16, "vocab_size": 32,
        "num_target_layers": 8, "max_position_embeddings": 4096,
        "dspark_config": {"block_size": 7, "target_layer_ids": [1], "mask_token_id": 3},
    }))
    return (
        BuildRequest(source, tmp_path / "out", "qwen3_8", "text_generation", "fp16"),
        BuildExecutionInputs("dspark", (NamedCheckpoint("draft", draft),)),
    )


@pytest.mark.parametrize("overrides, message", [
    ({"precision": "bf16"}, "--precision fp16"),
    ({"backend": "trt_rtx"}, "requires backend=trt"),
    ({"max_batch_size": 2}, "batch/TP/CP=1"),
    ({"tensor_parallel_size": 2}, "batch/TP/CP=1"),
    ({"dynamic_kv_cache": True}, "no dynamic KV"),
    ({"quantization": "fp8"}, "quantization unset or nvfp4"),
    ({"max_sequence_length": 8}, "above 8 and at most 1024"),
    ({"max_sequence_length": 1025}, "above 8 and at most 1024"),
    ({"max_sequence_length": 2048}, "above 8 and at most 1024"),
])
def test_dspark_request_errors_precede_adapter_work(tmp_path, monkeypatch, overrides, message):
    from dataclasses import replace
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path)
    monkeypatch.setattr(dispatch, "build", lambda *_args, **_kw: pytest.fail("adapter work started"))
    with pytest.raises(ValueError, match=message) as caught:
        dispatch.build_paired(replace(request, **overrides), None, execution)
    assert caught.value.__cause__ is None
    assert not list(tmp_path.glob(".out.edge-*"))


@pytest.mark.parametrize("limit", [9, 1024])
def test_dspark_capacity_boundaries_keep_the_requested_pair(tmp_path, monkeypatch, limit):
    from dataclasses import replace
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path)
    request = replace(request, max_sequence_length=limit)
    seen = []

    def build(original, writer, native, *, draft_dir):
        assert original is request and draft_dir == execution.checkpoints[0].model_dir
        with pytest.raises(NotImplementedError, match="Native Qwen3.8"):
            native(original, writer)
        seen.append(original)

    monkeypatch.setattr(dispatch, "build", build)
    dispatch.build_paired(request, None, execution)
    assert seen == [request]


def test_dspark_checkpoint_error_is_distinct_from_request_error(tmp_path, monkeypatch):
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path)
    (request.model_dir / "hf_quant_config.json").unlink()
    monkeypatch.setattr(dispatch, "build", lambda *_args, **_kw: pytest.fail("adapter work started"))
    with pytest.raises(ValueError, match="matching mixed-NVFP4 base"):
        dispatch.build_paired(request, None, execution)
