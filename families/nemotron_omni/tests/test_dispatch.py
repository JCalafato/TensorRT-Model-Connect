# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Known command failures select full native tasks, never quality-based gates."""
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from families.nemotron_omni.edge_llm import dispatch
from families.nemotron_omni.support import SUPPORTED_TASKS


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_tasks_select_only_available_edge_commands(task, tmp_path, monkeypatch):
    calls = []
    request = SimpleNamespace(task=task, precision="fp16", output_path=tmp_path / "model.bundle")
    writer = object()
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(dispatch.builder, "package_present", lambda: True)
    monkeypatch.setattr(dispatch.builder, "local_target", lambda: {})
    monkeypatch.setattr(dispatch.builder, "prepare", lambda *args: ({}, {}))
    monkeypatch.setattr(dispatch.builder, "publish", lambda *args: calls.append("edge"))
    dispatch.build(request, writer, lambda *args: calls.append("native"), 4096)
    edge = {"text_generation", "text_continuation", "vision_language_generation", "images_text_to_text"}
    assert calls == ["edge" if task in edge else "native"]
    assert not list(tmp_path.iterdir())


def test_builder_error_preserved_and_native_called_once(tmp_path, monkeypatch, caplog):
    request = SimpleNamespace(task="text_generation", precision="fp16", output_path=tmp_path / "model.bundle")
    calls = []
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(dispatch.builder, "package_present", lambda: True)
    def fail():
        raise RuntimeError("upstream builder fixture failure")
    monkeypatch.setattr(dispatch.builder, "local_target", fail)
    dispatch.build(request, object(), lambda *args: calls.append("native"), 4096)
    assert calls == ["native"]
    assert "Retrying native once" in caplog.text
    logs = list(tmp_path.glob("*.log"))
    assert len(logs) == 1 and "upstream builder fixture failure" in logs[0].read_text()


def test_edge_checkpoint_metadata_and_hf_cache_links(tmp_path, monkeypatch):
    """A normal linked HF snapshot must not fall back or publish private notes."""
    builder = dispatch.builder
    source, staging = tmp_path / "snapshot", tmp_path / "staging"
    source.mkdir()
    staging.mkdir()
    blob = tmp_path / "cached-blob"
    blob.write_bytes(b"prepared weights fixture")
    shard = source / "model-00001.safetensors"
    shard.symlink_to(blob)
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    (source / "config.json").write_text("{}")
    (source / "private-notes.json").write_text("{}")
    request = SimpleNamespace(model_dir=source, verbose=False)
    monkeypatch.setattr(builder, "installed_package", lambda _: {"plugin": "fixture.so"})
    monkeypatch.setattr(builder, "builder_python", lambda *_: "fixture-python")

    def build_fixture(command, **kwargs):
        engine = Path(command[command.index("--engine-dir") + 1])
        for name in ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "visual/visual.engine", "visual/config.json"):
            path = engine / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"builder output fixture")

    monkeypatch.setattr(builder.subprocess, "run", build_fixture)
    files, _ = builder.prepare(request, {}, staging, tmp_path / "build.log", 4096)
    assert files["edge_llm/checkpoint/" + shard.name].read_bytes() == blob.read_bytes()
    assert "edge_llm/checkpoint/config.json" in files
    assert "edge_llm/checkpoint/private-notes.json" not in files
    index.write_text(json.dumps({"weight_map": {"weight": "../cached-blob"}}))
    with pytest.raises(ValueError, match="shard path is not normalized"):
        builder.prepare(request, {}, staging, tmp_path / "build.log", 4096)
