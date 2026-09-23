# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from families.nemotron_h.model import _runtime_config, _stop_token_ids


def test_qualification_environment_keeps_extension_builds_in_family_environment() -> None:
    profile = yaml.safe_load(
        (Path(__file__).parent / "benchmark/nemotron-h-nano-9b.yaml").read_text(encoding="utf-8")
    )

    assert profile["reference_environment"]["build_isolation"] is False


def test_runtime_keeps_a_scalar_eos_and_every_declared_stop_token(tmp_path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 11, 12]}), encoding="utf-8"
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        bos_token_id=1,
        eos_token_id=12,
        pad_token_id=0,
    )
    family_model = SimpleNamespace(get_bundle_config_overrides=lambda _config: {})

    runtime = _runtime_config(tmp_path, config, family_model)

    assert runtime["eos_token_id"] == 2
    assert runtime["stop_token_ids"] == [2, 11, 12]


@pytest.mark.parametrize("value", [[], [2, 2], [True], [32], "2"])
def test_stop_token_contract_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="eos_token_id"):
        _stop_token_ids(value, 32)


def test_runtime_keeps_checkpoint_prompt_and_builder_policies() -> None:
    family = Path(__file__).resolve().parents[1]
    plugin = (family / "runtime/plugin.cpp").read_text(encoding="utf-8")
    model = (family / "model.py").read_text(encoding="utf-8")
    assert 'require_text_section(bundle, "tokenizer_config.json")' in plugin
    assert 'config.find("chat_template")' in plugin
    assert "chat_template.jinja" not in plugin
    assert '"chat_template.jinja"' not in model

    for path in (family / "model.py", family / "tp_builder.py"):
        source = path.read_text(encoding="utf-8")
        assert "builder_optimization_level = 3" in source
        assert "builder_optimization_level = 1" not in source

    # Edge has native scalar prefill even when the optimized head80 SSD is absent.
    from families.nemotron_h.edge_llm.dispatch import EDGE_DISPATCH, platform_matches

    for sm in (80, 120):
        assert platform_matches({"mamba_head_dim": 80}, {"sm": sm})
        assert ("linux", "x86_64", sm, "fp16") in EDGE_DISPATCH
    assert ("linux", "x86_64", 80, "nvfp4") not in EDGE_DISPATCH


def _edge_cli_source(tmp_path):
    import json

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    draft = tmp_path / "draft"
    draft.mkdir()
    return source, draft


def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch):
    from tensorrt_model_connect import build_cli
    from families.nemotron_h.edge_llm import dispatch
    from families.nemotron_h.edge_llm.config import NemotronHBuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, NemotronHBuildRequest)
        assert request.execution is execution
        assert execution.variant == "dflash"
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == [("draft", draft)]
        seen.append(request)
        writer.set_header(family="nemotron_h", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    assert build_cli.main([
        "build", str(source), "--precision", "fp16", "-o", str(output),
        "--execution-variant", "dflash", "--companion", f"draft={draft}",
    ]) == 0
    assert len(seen) == 1
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "dflash", "--companion", "missing_separator"],
    ["--execution-variant", "dflash", "--companion", "=path"],
    ["--execution-variant", "dflash", "--companion", "draft="],
    ["--execution-variant", "dflash", "--companion", "draft=https://example.com/model"],
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
    assert "--execution-variant {dflash}" in help_text
    assert "--companion" in help_text


def test_edge_request_preserves_fields_and_family_owner(tmp_path):
    import argparse
    from families.nemotron_h.edge_llm import cli
    from dataclasses import fields, replace
    from tensorrt_model_connect.build import BuildRequest
    from families.nemotron_h.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "nemotron_h", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    execution = BuildExecutionInputs("dflash", (NamedCheckpoint("draft", draft),))
    args = argparse.Namespace(command="build", execution_variant=None, companion=[])
    assert cli.prepare_build_request(request, args) is request
    extended = with_execution(request, execution)
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(request, field.name)
    with pytest.raises(ValueError, match="requires the nemotron_h family"):
        with_execution(replace(request, family="other"), execution)
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("dflash", execution.checkpoints * 2)


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_cli_failure_preserves_existing_bundle(tmp_path, monkeypatch, failure):
    from tensorrt_model_connect import build_cli
    from families.nemotron_h.edge_llm import dispatch

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    output.write_bytes(b"previous publication")

    def fail(request, writer, execution):
        writer.set_header(family="nemotron_h", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(dispatch, "build_paired", fail)
    with pytest.raises(type(failure)) as caught:
        build_cli.main([
            "build", str(source), "-o", str(output), "--execution-variant", "dflash",
            "--companion", f"draft={draft}",
        ])
    assert caught.value is failure
    assert output.read_bytes() == b"previous publication"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft", "pair.bundle", "target"]


def test_edge_pair_requires_draft_and_rechecks_local_inputs(tmp_path):
    from tensorrt_model_connect.build import BuildRequest
    from families.nemotron_h.edge_llm.config import BuildExecutionInputs, NamedCheckpoint
    from families.nemotron_h.edge_llm.dispatch import build_paired

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "nemotron_h", "text_generation", "fp16")
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(request, None, BuildExecutionInputs("dflash"))
    execution = BuildExecutionInputs("dflash", (NamedCheckpoint("draft", draft),))
    draft.rmdir()
    with pytest.raises(ValueError, match="existing local directory"):
        build_paired(request, None, execution)
