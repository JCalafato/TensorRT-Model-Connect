# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the trained driving vocabulary for the family-native tokenizer."""

import json
from pathlib import Path


_SPECIAL_NAMES = (
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
)
_TRAJECTORY_NAMES = (
    "history",
    "future",
    "history_start",
    "future_start",
    "history_end",
    "future_end",
)


def prepare_tokenizer(checkpoint: Path, *, backbone_dir: Path | None = None) -> bytes:
    """Return tokenizer JSON with the provider-ordered trajectory extension.

    This runs during bundle construction, never as a native-runtime callback.
    Only local checkpoint assets are read; no provider code is executed.
    """
    from tokenizers import Tokenizer

    checkpoint = Path(checkpoint)
    raw = json.loads((checkpoint / "config.json").read_text())
    count = raw["traj_vocab_size"]
    if type(count) is not int or count <= 0 or count > raw["vocab_size"]:
        raise ValueError("Invalid Alpamayo trajectory vocabulary size")
    # Load the serialized backend directly: AutoTokenizer may rewrite ByteLevel
    # options across Transformers releases, even when checkpoint assets agree.
    backbone_dir = Path(backbone_dir) if backbone_dir is not None else checkpoint / "base_vlm"
    tokenizer = Tokenizer.from_file(str(backbone_dir / "tokenizer.json"))
    tokenizer.add_tokens([f"<i{i}>" for i in range(count)])
    names = (
        _SPECIAL_NAMES
        if raw["add_special_tokens"]
        else tuple("traj_" + name for name in _TRAJECTORY_NAMES)
    )
    tokenizer.add_special_tokens([f"<|{name}|>" for name in names])
    actual = {name: tokenizer.token_to_id(f"<|traj_{name}|>") for name in _TRAJECTORY_NAMES}
    if (
        tokenizer.get_vocab_size(with_added_tokens=True) != raw["vocab_size"]
        or actual != raw["traj_token_ids"]
        or any(
            tokenizer.token_to_id(f"<i{i}>") != raw["traj_token_start_idx"] + i
            for i in range(count)
        )
    ):
        raise ValueError("Alpamayo tokenizer extension disagrees with trained vocabulary IDs")
    return tokenizer.to_str().encode("utf-8")
