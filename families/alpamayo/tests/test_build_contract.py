# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission for complete native driving bundles, not numerical qualification."""

import pytest

from families.alpamayo.cli import BuildRequest


@pytest.mark.parametrize("batch", [1, 6])
def test_documented_native_build_contract(tmp_path, batch):
    request = BuildRequest(tmp_path, max_batch_size=batch)
    assert request.task == "structured_decision"
    assert request.max_sequence_length > request.max_new_tokens


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_batch_size": 2},
        {"max_batch_size": True},
        {"precision": "fp32"},
        {"execution_variant": "unrecognized"},
        {"task": "text_generation"},
        {"backend": "edgellm"},
        {"max_new_tokens": 0},
        {"max_sequence_length": 128},
        {"workspace_bytes": -1},
        {"workspace_bytes": 1.5},
    ],
)
def test_reject_invalid_driving_build_contract(tmp_path, arguments):
    with pytest.raises(ValueError):
        BuildRequest(tmp_path, **arguments)


def test_family_build_command_is_registered():
    from tensorrt_model_connect.family_cli import load_family_cli

    declaration = load_family_cli("alpamayo")
    (command,) = declaration["commands"]
    assert command["executor"] == "python"
    assert command["handler"] == "cli:build"
    arguments = {item["name"]: item for item in command["arguments"]}
    assert arguments["max_batch_size"]["choices"] == [1, 6]
    assert arguments["max_sequence_length"]["default"] == 4096
    assert arguments["execution_variant"]["choices"] == ["auto", "native"]


def test_existing_generic_request_maps_to_same_family_defaults(tmp_path):
    from tensorrt_model_connect.build import BuildRequest as GenericBuildRequest
    from families.alpamayo.cli import normalize_build_request

    request = GenericBuildRequest(
        tmp_path,
        tmp_path / "model.bundle",
        "alpamayo",
        "structured_decision",
        "fp16",
        max_batch_size=6,
    )
    assert normalize_build_request(request) == BuildRequest(
        tmp_path, max_batch_size=6, output_path=tmp_path / "model.bundle"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"image_height": 720},
        {"video_num_frames": 4},
        {"quantization": "fp8"},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
        {"fp32_layers": (0,)},
        {"mtp_seq_len": 2},
    ],
)
def test_generic_unsupported_controls_are_not_silently_ignored(tmp_path, arguments):
    from tensorrt_model_connect.build import BuildRequest as GenericBuildRequest
    from families.alpamayo.cli import normalize_build_request

    request = GenericBuildRequest(
        tmp_path, tmp_path / "model.bundle", "alpamayo", "structured_decision", "fp16", **arguments
    )
    with pytest.raises(ValueError):
        normalize_build_request(request)


def test_local_backbone_assets_are_used_without_download(tmp_path, monkeypatch):
    import huggingface_hub
    from families.alpamayo.assets import BACKBONE_FILES, resolve_backbone

    def forbidden(**kwargs):
        raise AssertionError("local assets must not trigger network access")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    local = tmp_path / "base_vlm"
    local.mkdir()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        resolve_backbone(tmp_path)
    for name in BACKBONE_FILES:
        (local / name).write_text("{}")
    assert resolve_backbone(tmp_path) == local


def test_missing_backbone_fetches_only_pinned_metadata(tmp_path, monkeypatch):
    import huggingface_hub
    from families.alpamayo.assets import (
        BACKBONE_FILES,
        BACKBONE_REPOSITORY,
        BACKBONE_REVISION,
        resolve_backbone,
    )

    cached = tmp_path / "cached"
    cached.mkdir()
    for name in BACKBONE_FILES:
        (cached / name).write_text("{}")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert resolve_backbone(tmp_path) == cached
    assert calls == [
        {
            "repo_id": BACKBONE_REPOSITORY,
            "revision": BACKBONE_REVISION,
            "allow_patterns": list(BACKBONE_FILES),
        }
    ]
