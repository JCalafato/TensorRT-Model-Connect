# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full prepared checkpoint and public Tasks against frozen independent known answers.

These six answers were fixed before model execution. They cover arithmetic,
world knowledge, image grounding and ordered two-image identification, not
open-ended generation or every possible request. Numerical graph comparisons
remain separate owning tests and keep their original tolerances.
"""

import json
import os
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

import pytest
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).parent
FAMILY = "diffusion_gemma"
FIXTURE = json.loads((ROOT / "data/full-requests.json").read_text())
MANIFESTS = [json.loads(path.read_text()) for path in sorted((ROOT / "manifests").glob("*.json"))]


def _selection(config, name):
    return {
        part.strip()
        for value in (config.getoption(name, default=[]) or [])
        for part in value.split(",")
        if part.strip()
    }


def _selected(config, manifest):
    models = _selection(config, "--e2e-model")
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        models.update(
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    cases = _selection(config, "--e2e-testcase")
    if not models and not cases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("full checkpoint E2E requires explicit selection")
    if models and not models.intersection({FAMILY, manifest["name"]}):
        pytest.skip("profile not selected")
    if cases and manifest["name"] not in cases:
        pytest.skip("testcase not selected")


def _checkpoint(manifest):
    explicit = os.environ.get("TRTMC_DIFFUSION_GEMMA_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "model.safetensors.index.json").is_file()
        return root
    from huggingface_hub import hf_hub_download, snapshot_download

    index = hf_hub_download(
        manifest["hf_id"], "model.safetensors.index.json", revision=manifest["hf_revision"]
    )
    shards = set(json.loads(Path(index).read_text())["weight_map"].values())
    return Path(
        snapshot_download(
            manifest["hf_id"],
            revision=manifest["hf_revision"],
            allow_patterns=["*.json", "*.jinja", *sorted(shards)],
        )
    )


def _requests(directory):
    from PIL import Image

    assets = {}
    source = os.environ.get("TRTMC_DIFFUSION_GEMMA_ASSET_DIR")
    assert FIXTURE["edge_revision"] == "95515c2f87fba8982db5a519f9022277667b3cc9"
    for name in FIXTURE["assets"]:
        if source:
            data = (Path(source) / name).read_bytes()
        else:
            url = (
                "https://raw.githubusercontent.com/NVIDIA/TensorRT-Edge-LLM/"
                + FIXTURE["edge_revision"]
                + "/examples/multimodal/pics/"
                + name
            )
            with urlopen(url, timeout=60) as response:
                data = response.read()
        path = directory / name
        path.write_bytes(data)
        image = Image.open(path).convert("RGB")
        pixels = path.with_suffix(".rgb")
        pixels.write_bytes(image.tobytes())
        assets[name] = {"path": str(pixels), "height": image.height, "width": image.width}
    return [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "images": [assets[name] for name in row["images"]],
            **FIXTURE["generation"],
        }
        for row in FIXTURE["cases"]
    ]


def _runner(runtime):
    roots = [runtime]
    configured = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if configured:
        roots.insert(0, Path(configured))
    for name in ("libtrtmc_model_diffusion_gemma.so", "trtmc_model_diffusion_gemma.dll"):
        if (runtime / name).is_file():
            roots.append((runtime / name).resolve().parent)
    suffix = ".exe" if os.name == "nt" else ""
    for root in roots:
        for configuration in ("", "Release", "Debug"):
            path = (
                root
                / "families/diffusion_gemma"
                / configuration
                / ("diffusion_gemma_inference_test" + suffix)
            )
            if path.is_file():
                return path
    raise AssertionError("build the owning diffusion_gemma_inference_test target")


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?")


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda manifest: manifest["name"])
def test_e2e(manifest, request, tmp_path):
    _selected(request.config, manifest)
    assert manifest["testcases"][0]["reference_backend"] == "fixed_ground_truth"
    assert manifest["trust_remote_code"] is False
    modalities = {
        "text" if not row["images"] else "image" if len(row["images"]) == 1 else "multiple_images"
        for row in FIXTURE["cases"]
    }
    assert modalities == set(manifest["testcases"][0]["modalities"])
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    runner = _runner(runtime)
    checkpoint = _checkpoint(manifest)
    requests = _requests(tmp_path)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=checkpoint,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                quantization=manifest["quantization"],
                max_sequence_length=manifest["max_sequence_length"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
    source, destination = tmp_path / "requests.json", tmp_path / "responses.json"
    source.write_text(json.dumps(requests))
    command = [str(runner), str(bundle), str(runtime), str(source), str(destination)]
    env = dict(os.environ)
    variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[variable] = str(runtime) + os.pathsep + env.get(variable, "")
    with evidence_stage("native"):
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=2400)
        record_evidence("commands", {"argv": command, "returncode": result.returncode})
        record_evidence("native", {"stdout": result.stdout, "stderr": result.stderr})
        assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(destination.read_text())
    assert len(actual) == len(FIXTURE["cases"])
    comparisons = []
    for result, expected in zip(actual, FIXTURE["cases"]):
        assert result["id"] == expected["id"]
        assert 0 < len(result["token_ids"]) <= FIXTURE["generation"]["max_new_tokens"]
        comparisons.append(
            {
                "id": result["id"],
                "actual": result["text"],
                "expected": expected["expected"],
                "correct": _normalize(result["text"]) == _normalize(expected["expected"]),
            }
        )
    accuracy = sum(row["correct"] for row in comparisons) / len(comparisons)
    threshold = json.loads((ROOT / "thresholds" / (manifest["name"] + ".json")).read_text())[
        "threshold_overrides"
    ]["answer_accuracy"]
    record_evidence("reference", {"backend": "fixed_ground_truth", "cases": FIXTURE["cases"]})
    record_evidence("metrics", {"answer_accuracy": accuracy, "cases": comparisons})
    assert accuracy >= threshold, comparisons
