# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installation origin must not enable a known command-failing Edge route."""

import json

import pytest

from families.diffusion_gemma.edge_llm import dispatch


@pytest.mark.parametrize("release", [None, "0.11.0", "0.10.0"])
def test_native_request_is_preserved(monkeypatch, caplog, release):
    monkeypatch.setattr(dispatch, "installed_release", lambda: release)
    request, writer = object(), object()
    calls = []

    def native(r, w):
        calls.append((r, w))
        return "native"

    assert dispatch.build(request, writer, native) == "native"
    assert calls == [(request, writer)]
    if release == "0.11.0":
        assert "token_phase" in caplog.text
        assert "no Edge engine is advertised" in caplog.text


def test_wheel_discovery_needs_no_native_runtime_import(monkeypatch):
    seen = []
    monkeypatch.setattr(dispatch.metadata, "version", lambda name: seen.append(name) or "0.11.0")
    assert dispatch.installed_release() == "0.11.0"
    assert seen == ["tensorrt-edgellm"]


def test_source_installation_and_invalid_manifest(monkeypatch, tmp_path, caplog):
    def absent(name):
        raise dispatch.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(dispatch.metadata, "version", absent)
    monkeypatch.setattr(dispatch, "cmake_prefixes", lambda: [tmp_path])
    assert dispatch.installed_release() is None
    manifest = tmp_path / "share/trtmc/edge-llm.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "version": "0.11.0", "revision": dispatch.EDGE_REVISION})
    )
    assert dispatch.installed_release() == "0.11.0"
    manifest.write_text("not JSON")
    calls = []
    dispatch.build(None, None, lambda *args: calls.append(args))
    assert calls == [(None, None)]
    assert "discovery failed" in caplog.text


def test_native_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(dispatch, "installed_release", lambda: "0.11.0")

    def failed(*args):
        raise RuntimeError("native build failure")

    with pytest.raises(RuntimeError, match="native build failure"):
        dispatch.build(None, None, failed)
