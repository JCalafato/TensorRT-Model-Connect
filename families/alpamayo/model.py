# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete family-owned driving construction with optional Edge offload."""

import json
import math
from pathlib import Path

from .config import ModelConfig


def runtime_metadata(request, config, weights, *, backbone_dir=None):
    root = Path(request.model_dir)
    raw = json.loads((root / "config.json").read_text())
    backbone_dir = Path(backbone_dir) if backbone_dir is not None else root / "base_vlm"
    vlm = json.loads((backbone_dir / "config.json").read_text())
    processor = json.loads((backbone_dir / "preprocessor_config.json").read_text())
    generation = json.loads((backbone_dir / "generation_config.json").read_text())
    history = raw["hist_traj_tokenizer_cfg"]
    if history != {"_target_": "alpamayo_r1.models.delta_tokenizer.DeltaTrajectoryTokenizer"}:
        raise ValueError("Alpamayo requires its trained default XYZ delta history tokenizer")
    vision = config.vision
    for key, expected in (
        ("patch_size", vision.patch_size),
        ("temporal_patch_size", vision.temporal_patch_size),
        ("merge_size", vision.spatial_merge_size),
    ):
        if processor[key] != expected:
            raise ValueError(f"Alpamayo processor {key} disagrees with the visual graph")
    unit = vision.spatial_merge_size**2
    patch_capacity = math.ceil(config.max_pixels / (vision.patch_size**2 * unit)) * unit
    action = {**raw["action_space_cfg"], "history_points": config.action.history_tokens // 3}
    action.pop("_target_")
    # These FP32 checkpoint buffers, not higher-precision JSON constants, define
    # the provider's physical action normalization.
    for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
        value = weights.array("action_space." + name)
        if value.size != 1 or not math.isfinite(float(value.item())):
            raise ValueError(f"Invalid trained action normalization buffer: {name}")
        action[name] = float(value.item())
    if action["accel_std"] <= 0 or action["curvature_std"] <= 0:
        raise ValueError("Action normalization scales must be positive")
    eos = generation["eos_token_id"]
    return {
        "version": 1,
        "execution": "native",
        "precision": request.precision,
        "max_batch_size": request.max_batch_size,
        "prompt_capacity": request.max_sequence_length - request.max_new_tokens,
        "positions": {
            **{
                name: vlm[name + "_id"]
                for name in ("image_token", "video_token", "vision_start_token", "vision_end_token")
            },
            "spatial_merge_size": vision.spatial_merge_size,
            "head_width": config.text.head_dim,
            "mrope_section": config.text.mrope_section,
            "rope_theta": config.text.rope_theta,
        },
        "vision": {
            "patch_size": vision.patch_size,
            "temporal_patch_size": vision.temporal_patch_size,
            "spatial_merge_size": vision.spatial_merge_size,
            "num_position_embeddings": vision.num_position_embeddings,
            "head_width": vision.hidden_size // vision.num_heads,
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
            "patch_capacity": patch_capacity,
            "image_mean": processor["image_mean"],
            "image_std": processor["image_std"],
            "rescale_factor": processor.get("rescale_factor", 1.0 / 255.0),
            "deepstack_count": len(vision.deepstack_visual_indexes),
        },
        "history": {
            "points": config.action.history_tokens // 3,
            "bins": 1000,
            "token_offset": config.trajectory_vocab_start + raw["traj_tokenizer_cfg"]["num_bins"],
            "placeholder_token": config.trajectory_token_ids["history"],
            "minimum": [-4, -4, -10],
            "maximum": [4, 4, 10],
        },
        "action_space": action,
        "reasoning": {
            # Driving sampling defaults, not the backbone's unrelated chat defaults.
            "temperature": 0.6,
            "top_p": 0.98,
            "top_k": 0,
            "max_new_tokens": request.max_new_tokens,
            "future_start_token": config.trajectory_token_ids["future_start"],
            "pad_token": generation["pad_token_id"],
            "eos_tokens": eos if isinstance(eos, list) else [eos],
            "suppressed_begin": config.trajectory_vocab_start,
            "suppressed_end": config.trajectory_vocab_start + config.trajectory_vocab_size,
        },
    }


def build(request, writer):
    from .action_engine import build_action_engine
    from .assets import resolve_backbone
    from .cli import normalize_build_request
    from .text_engine import build_text_engine
    from .tokenizer import prepare_tokenizer
    from .vision_engine import build_vision_engine
    from .weights import Weights
    from .edge_llm import dispatch

    request = normalize_build_request(request)
    backbone = resolve_backbone(request.model_dir)
    config = ModelConfig.parse(
        json.loads((Path(request.model_dir) / "config.json").read_text()),
        json.loads((backbone / "config.json").read_text()),
    )
    with Weights(request.model_dir) as weights:
        metadata = runtime_metadata(request, config, weights, backbone_dir=backbone)

        def native(request, writer):
            tokenizer = prepare_tokenizer(Path(request.model_dir), backbone_dir=backbone)
            writer.set_header(family="alpamayo", task=request.task, backend=request.backend)
            writer.add_json("runtime.json", metadata)
            writer.add_bytes("tokenizer.json", tokenizer)
            # Build vision first: command failures must not leave a publishable
            # partial bundle or spend time constructing unrelated later engines.
            constructors = (
                (
                    "vision.plan",
                    build_vision_engine,
                    config.vision,
                    {"patch_capacity": metadata["vision"]["patch_capacity"]},
                ),
                (
                    "text.plan",
                    build_text_engine,
                    config,
                    {
                        "max_batch": request.max_batch_size,
                        "input_capacity": metadata["prompt_capacity"],
                        "prefix_capacity": request.max_sequence_length,
                    },
                ),
                (
                    "action.plan",
                    build_action_engine,
                    config,
                    {
                        "max_batch": request.max_batch_size,
                        "prefix_capacity": request.max_sequence_length,
                    },
                ),
            )
            for name, constructor, component, capacities in constructors:
                plan = constructor(
                    component,
                    weights,
                    request.precision,
                    workspace_bytes=request.workspace_bytes,
                    **capacities,
                )
                # Stream from TensorRT host memory; do not duplicate multi-GB plans
                # as Python bytes or create a second private engine directory.
                with writer.open_section(name) as section:
                    section.write(memoryview(plan))
                del plan
                weights.release()

        dispatch.build(request, writer, native, metadata, backbone)
