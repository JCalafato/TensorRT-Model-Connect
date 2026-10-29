# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete driving Task against one pinned public ground-truth trajectory.

This is not a claim about the original 644-clip benchmark. Component numerical
checks remain independent and unchanged.
"""

from pathlib import Path
import json
import os
import subprocess

import numpy as np
import pytest

from families.alpamayo.cli import build
from tools.e2e_evidence import evidence_stage, record_evidence
from families.alpamayo.tests.driving_sample import load_sample, prepare_sample

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/alpamayo-r1-fp16-driving-b6.json").read_text())
CASE = MANIFEST["testcases"][0]["name"]


def _selection(config, name):
    return {
        part.strip()
        for value in (config.getoption(name, default=[]) or [])
        for part in value.split(",")
        if part.strip()
    }


def _selected(config):
    models, cases = _selection(config, "--e2e-model"), _selection(config, "--e2e-testcase")
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        models.update(
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not models and not cases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real Alpamayo E2E requires explicit selection")
    if models and not models.intersection({"alpamayo", CASE}):
        pytest.skip("model profile was not selected")
    if cases and CASE not in cases:
        pytest.skip("testcase was not selected")


def _checkpoint():
    explicit = os.environ.get("TRTMC_ALPAMAYO_MODEL_DIR")
    if explicit:
        path = Path(explicit)
        assert (path / "model.safetensors.index.json").is_file()
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MANIFEST["hf_id"],
            revision=MANIFEST["hf_revision"],
            allow_patterns=["*.json", "*.safetensors"],
        )
    )


def _request(directory, data):
    specification = MANIFEST["sample"]
    t0 = specification["t0_us"]
    document = {
        "schema": "alpamayo.driving.v1",
        "coordinate_frame": "current_ego",
        "num_samples": 6,
        "sampling": {"seed": 0},
        "observations": [
            {
                "id": specification["clip_id"],
                "cameras": [
                    {
                        "camera_index": camera,
                        "video_index": i,
                        "timestamps_us": data["absolute_timestamps"][i].astype(np.int64).tolist(),
                    }
                    for i, camera in enumerate(specification["camera_indices"])
                ],
                "history": {
                    "xyz": data["ego_history_xyz"][0, 0].tolist(),
                    "rotations": data["ego_history_rot"][0, 0].tolist(),
                    "timestamps_us": [t0 - (15 - i) * 100000 for i in range(16)],
                },
            }
        ],
    }
    return {
        "document": json.dumps(document),
        "video_frame_paths": [
            [str(directory / f"frame-{4 * c + i:02d}.png") for i in range(4)] for c in range(4)
        ],
    }


def trajectory_metrics(response, expected):
    assert response["schema"] == "alpamayo.driving.result.v1"
    assert response["coordinate_frame"] == "current_ego"
    samples = response["samples"]
    assert len(samples) == 6
    errors = []
    for i, sample in enumerate(samples):
        assert sample["id"] == MANIFEST["sample"]["clip_id"] and sample["sample_index"] == i
        assert 0 < len(sample["tokens"]) <= 128 and sample["reasoning"]["cot"].strip()
        trajectory = sample["trajectory"]
        xyz, rotations = np.asarray(trajectory["xyz"]), np.asarray(trajectory["rotations"])
        actions = np.asarray(sample["normalized_acceleration_curvature"])
        assert xyz.shape == (64, 3) and rotations.shape == (64, 3, 3) and actions.shape == (64, 2)
        assert all(np.isfinite(value).all() for value in (xyz, rotations, actions))
        assert trajectory["timestamps_us"] == [
            MANIFEST["sample"]["t0_us"] + (i + 1) * 100000 for i in range(64)
        ]
        errors.append(float(np.linalg.norm(xyz[:, :2] - expected[:, :2], axis=-1).mean()))
    return {
        "per_sample_ade_m": errors,
        "minade6_at_6_4s_m": min(errors),
        "scope": MANIFEST["sample"]["scope"],
    }


def test_e2e(request, tmp_path):
    _selected(request.config)
    assert MANIFEST["task"] == "structured_decision" and MANIFEST["precision"] == "fp16"
    assert MANIFEST["tensor_parallel_size"] == 1 and MANIFEST["trust_remote_code"] is False
    case = MANIFEST["testcases"][0]
    assert case["reference_backend"] == "physical_ai_av_ground_truth"
    assert set(case["modalities"]) == {"video", "history", "reasoning", "trajectory"}
    assert MANIFEST["max_batch_size"] == 6
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    suffix = ".exe" if os.name == "nt" else ""
    worker = Path(
        os.environ.get("TRTMC_BENCHMARK_WORKER", runtime / ("trtmc_benchmark_worker" + suffix))
    )
    assert worker.is_file(), "build the existing trtmc_benchmark_worker target"
    explicit = os.environ.get("TRTMC_ALPAMAYO_SAMPLE_DIR")
    directory = Path(explicit) if explicit else tmp_path / "public-sample"
    with evidence_stage("reference"):
        data = (load_sample if explicit else prepare_sample)(directory, MANIFEST["sample"])
        record_evidence(
            "reference", {"backend": "physical_ai_av_ground_truth", "sample": MANIFEST["sample"]}
        )
    bundle = tmp_path / MANIFEST["bundle"]
    with evidence_stage("build"):
        # This existing test qualifies native execution, regardless of optional SDK discovery.
        build(
            model=str(_checkpoint()),
            output=bundle,
            max_batch_size=MANIFEST["max_batch_size"],
            max_sequence_length=MANIFEST["max_sequence_length"],
            execution_variant="native",
        )
    source, destination = tmp_path / "request.json", tmp_path / "output.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "case_name": CASE,
                "bundle": str(bundle),
                "runtime_root": str(runtime),
                "expected_family": "alpamayo",
                "expected_task": "structured_decision",
                "operation": "decide",
                "request": _request(directory, data),
                "measurement": {"warmup": 0, "iterations": 1, "asset_loading_included": False},
            }
        )
    )
    command = [str(worker), "--request", str(source), "--output", str(destination)]
    environment = dict(os.environ)
    library_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    environment[library_path] = str(runtime) + os.pathsep + environment.get(library_path, "")
    with evidence_stage("native"):
        completed = subprocess.run(
            command, env=environment, text=True, capture_output=True, timeout=1800
        )
        record_evidence("commands", {"argv": command, "returncode": completed.returncode})
        record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
        assert completed.returncode == 0, completed.stdout + completed.stderr
    output = json.loads(destination.read_text())["output_summary"]["response"]
    with evidence_stage("comparison"):
        metrics = trajectory_metrics(output, data["ego_future_xyz"][0, 0])
        limit = json.loads((ROOT / "thresholds" / f"{CASE}.json").read_text())[
            "threshold_overrides"
        ]["minade6_at_6_4s_m"]
        record_evidence("metrics", {**metrics, "maximum_minade6_at_6_4s_m": limit})
        assert metrics["minade6_at_6_4s_m"] <= limit, metrics
