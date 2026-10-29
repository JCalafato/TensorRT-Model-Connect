# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter mechanics only; mocked commands are not model qualification."""

from contextlib import contextmanager
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file

from families.alpamayo.edge_llm import builder, dispatch


class Writer:
    def __init__(self):
        self.header = None
        self.sections = {}

    def set_header(self, **kwargs):
        self.header = kwargs

    def add_json(self, name, value):
        self.sections[name] = value

    @contextmanager
    def open_section(self, name):
        buffer = io.BytesIO()
        yield buffer
        self.sections[name] = buffer.getvalue()


@pytest.fixture
def request_(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "package_present", lambda: True)
    monkeypatch.setattr(builder, "local_target", lambda: {"sm": 89})
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    return SimpleNamespace(
        model_dir=tmp_path,
        output_path=tmp_path / "driving.bundle",
        task="structured_decision",
        backend="trt",
        precision="fp16",
        max_batch_size=6,
        max_sequence_length=4096,
    )


def test_complete_components_exact_mask_and_no_checkpoint_copy(request_, monkeypatch, tmp_path):
    source = request_.model_dir
    raw = {
        "model_type": "alpamayo_r1",
        "traj_token_start_idx": 10,
        "traj_vocab_size": 4,
        "vocab_size": 17,
    }
    (source / "config.json").write_text(json.dumps(raw))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"vlm.lm_head.weight": "part.safetensors"}})
    )
    shard = source / "part.safetensors"
    shard.write_bytes(b"not copied")
    backbone = source / "base_vlm"
    backbone.mkdir()
    staging = source / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        builder, "installed_package", lambda t: {"python": "/sdk/python", "plugin": "/sdk/plugin"}
    )
    monkeypatch.setattr(builder, "builder_python", lambda p, t: p["python"])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        model = Path(command[command.index("--model-dir") + 1])
        assert (model / "part.safetensors").is_symlink()
        assert json.loads((model / "config.json").read_text())["vlm_name_or_path"] == str(backbone)
        engine = Path(command[command.index("--engine-dir") + 1])
        component = command[command.index("--components") + 1]
        if component == "llm":
            mapping = Path(command[command.index("--reduced-vocab-dir") + 1])
            np.testing.assert_array_equal(
                load_file(mapping / "vocab_map.safetensors")["vocab_map"], [*range(10), 14, 15, 16]
            )
            names = [
                "llm.engine",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab_map.safetensors",
            ]
            engine.mkdir(parents=True)
            (engine / "config.json").write_text(
                json.dumps({"vocab_size": 17, "reduced_vocab_size": 13})
            )
        else:
            assert component == "visual,action" and "--reduced-vocab-dir" not in command
            names = [
                "visual/visual.engine",
                "visual/config.json",
                "action/action.engine",
                "action/config.json",
            ]
        for name in names:
            path = engine / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"engine artifact")

    monkeypatch.setattr(builder.subprocess, "run", run)
    files, marker = builder.prepare(
        request_,
        {"sm": 89},
        staging,
        source / "build.log",
        {"prompt_capacity": 3968, "execution": "native"},
        backbone,
    )
    assert len(calls) == 2
    assert files["edge_llm/checkpoint/part.safetensors"] == shard
    assert json.loads((source / "config.json").read_text()) == raw
    assert marker["excluded_output_token_interval"] == [10, 14]
    assert marker["runtime"]["execution"] == "edge_llm"
    assert "content.placeholder" in files["edge_llm/engine/chat_template.jinja"].read_text()
    assert "<|cot_start|>" in files["edge_llm/engine/chat_template.jinja"].read_text()
    writer = Writer()
    builder.publish(request_, writer, files, marker)
    assert writer.header["family"] == "alpamayo"
    assert writer.sections["runtime.json"]["execution"] == "edge_llm"


def test_failed_edge_build_warns_and_calls_native_once(request_, monkeypatch, caplog):
    def fail(*args):
        raise RuntimeError("official builder error")

    monkeypatch.setattr(builder, "prepare", fail)
    writer, calls = Writer(), []
    dispatch.build(request_, writer, lambda r, w: calls.append((r, w)), {}, request_.model_dir)
    assert calls == [(request_, writer)] and writer.header is None
    assert "unchanged request" in caplog.text
    assert "official builder error" in next(request_.model_dir.glob(".*.log")).read_text()


@pytest.mark.parametrize("reason", ["absent", "platform", "precision", "native"])
def test_nonmatches_do_not_attempt_edge(request_, monkeypatch, reason):
    if reason == "absent":
        monkeypatch.setattr(builder, "package_present", lambda: False)
    elif reason == "platform":
        monkeypatch.setattr(dispatch.sys, "platform", "win32")
    elif reason == "precision":
        request_.precision = "fp32"
    else:
        request_.execution_variant = "native"
        monkeypatch.setattr(
            builder, "package_present", lambda: pytest.fail("native must not inspect Edge")
        )
    monkeypatch.setattr(builder, "prepare", lambda *a: pytest.fail("unexpected Edge command"))
    calls = []
    dispatch.build(request_, Writer(), lambda r, w: calls.append(r), {}, request_.model_dir)
    assert calls == [request_]


def test_publication_failure_is_not_retried(request_, monkeypatch):
    monkeypatch.setattr(builder, "prepare", lambda *a: ({}, {}))

    def fail(*args):
        raise OSError("bundle write failed")

    monkeypatch.setattr(builder, "publish", fail)
    with pytest.raises(OSError, match="bundle write failed"):
        dispatch.build(
            request_, Writer(), lambda *a: pytest.fail("unsafe retry"), {}, request_.model_dir
        )


def test_cancellation_is_not_retried(request_, monkeypatch):
    def fail(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(builder, "prepare", fail)
    with pytest.raises(KeyboardInterrupt):
        dispatch.build(
            request_, Writer(), lambda *a: pytest.fail("cancelled retry"), {}, request_.model_dir
        )
