# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve only the tokenizer/processor metadata missing from Alpamayo weights."""

from pathlib import Path

BACKBONE_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
BACKBONE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BACKBONE_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
)


def resolve_backbone(model_dir: str | Path) -> Path:
    """Prefer explicit local assets; otherwise fetch the pinned metadata only.

    This does not download backbone weights or register a standalone Qwen
    family. Hugging Face's existing cache/offline settings remain in effect.
    Explicit but incomplete local assets fail rather than mixing revisions.
    """
    path = Path(model_dir) / "base_vlm"
    if not path.exists():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=BACKBONE_REPOSITORY,
                revision=BACKBONE_REVISION,
                allow_patterns=list(BACKBONE_FILES),
            )
        )
    missing = [name for name in BACKBONE_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete Alpamayo backbone metadata: " + ", ".join(missing))
    return path
