# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin family-owned adapter to the pinned Edge direct-builder API."""

from __future__ import annotations

import json
from pathlib import Path

from .driving_template import DRIVING_CHAT_TEMPLATE
import shutil
import shlex
import subprocess
import sys

from tensorrt_model_connect.build import cmake_prefixes, detect_local_platform


EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"


def local_target() -> dict:
    """Return the executing worker identity supplied by generic build mechanics."""
    return detect_local_platform()


def package_present() -> bool:
    """Absence of the optional SDK is a non-match, not a failed Edge build."""
    for prefix in cmake_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if manifest.exists() or manifest.is_symlink():
            return True
    return False


def installed_package(target: dict) -> dict:
    """Resolve CMake installation via standard prefixes; never install anything.

    Args:
        target: Executing device and SDK identity.

    Returns:
        Validated package metadata with absolute Python and plugin paths.

    Raises:
        FileNotFoundError: No CMake installation or required artifact exists.
        ValueError: Pin, architecture, SDK or contained-path contract differs.
    """
    for prefix in cmake_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        if package.get("schema_version") != 1 or package.get("revision") != EDGE_REVISION:
            raise ValueError(f"Edge package has an unsupported revision/schema: {manifest}")
        if package.get("version") != "0.11.0" or package.get("arch") != target["arch"]:
            raise ValueError("Edge package version/architecture differs from executing worker")
        if target["sm"] not in package.get("architectures", []):
            raise ValueError("Edge package was not built for this local GPU")
        cuda_version = ".".join(str(package.get("cuda_version", "")).split(".")[:2])
        if (
            cuda_version != target["cuda_version"]
            or package.get("tensorrt_version") != target["tensorrt_version"]
        ):
            raise ValueError("Edge package CUDA/TensorRT differs from executing worker")
        for name in ("python", "plugin"):
            relative = Path(package[name])
            path = (prefix / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(prefix.resolve()):
                raise ValueError(f"Edge package {name} must be contained in its installation")
            if not path.is_file():
                raise FileNotFoundError(f"Edge package {name} is missing: {path}")
            package[name] = str(path)
        return package
    raise FileNotFoundError(
        "Edge-LLM is not installed; enable the optional Edge-LLM CMake dependency "
        "and set CMAKE_PREFIX_PATH to its install prefix"
    )


def builder_python(package: dict, target: dict) -> str:
    """Use installed 0.11.0 Python tools when compatible, otherwise the SDK Python.

    The native SDK remains required. This only discovers an interpreter; it
    never installs packages or substitutes the wheel's runtime for the C++ SDK.
    An incompatible ambient install is ignored.
    """
    interpreter = sys.executable
    try:
        if not Path(interpreter).is_absolute() or not Path(interpreter).is_file():
            raise ValueError("Edge builder Python must be an existing absolute interpreter path")
        probe = subprocess.run(
            [
                interpreter,
                "-I",
                "-c",
                "import json, tensorrt, tensorrt_edgellm; "
                "from experimental.builder.cli import main; "
                "print(json.dumps([tensorrt_edgellm.__version__, tensorrt.__version__]))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if json.loads(probe.stdout) != ["0.11.0", target["tensorrt_version"]]:
            raise ValueError(
                "Edge builder Python must match Edge 0.11.0 and the native TensorRT SDK"
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        return package["python"]
    return interpreter


def _checkpoint_files(source: Path) -> dict[str, Path]:
    """Keep indexed shards in place; only stream them when publishing a bundle."""
    index = json.loads((source / "model.safetensors.index.json").read_text())
    files = {}
    for name in sorted(set(index["weight_map"].values())):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or "\\" in name
            or "\0" in name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Alpamayo checkpoint shard name is not normalized")
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Alpamayo shard: {name}")
        files[name] = path
    if not files:
        raise ValueError("Alpamayo checkpoint has no indexed weight shards")
    for name in ("config.json", "model.safetensors.index.json"):
        files[name] = source / name
    return files


def prepare(request, target: dict, staging: Path, log_path: Path, metadata: dict, backbone: Path):
    """Call official builder tools for all three components, with exact output masking.

    The provider suppresses discrete trajectory tokens during language generation.
    The official reduced-vocabulary option expresses that hard exclusion exactly;
    it must not be approximated with a finite, capacity-limited runtime logit bias.
    Visual and action components use a separate invocation because that option is
    intentionally LLM-only. No model checkpoint or upstream source is modified.
    """
    import numpy as np
    from safetensors.numpy import save_file

    package = installed_package(target)
    source = Path(request.model_dir).resolve()
    checkpoint = _checkpoint_files(source)
    root = json.loads((source / "config.json").read_text())
    start, count, full = (
        int(root[key]) for key in ("traj_token_start_idx", "traj_vocab_size", "vocab_size")
    )
    if not 0 <= start < start + count <= full:
        raise ValueError("Alpamayo trajectory output-token interval is invalid")
    vocabulary = np.concatenate((np.arange(start), np.arange(start + count, full))).astype(np.int32)
    mapping = staging / "vocabulary"
    mapping.mkdir()
    save_file({"vocab_map": vocabulary}, mapping / "vocab_map.safetensors")

    # A lightweight overlay resolves local or downloaded backbone metadata without
    # altering config.json or making another multi-gigabyte checkpoint copy.
    overlay = staging / "builder-checkpoint"
    overlay.mkdir()
    for name, path in checkpoint.items():
        if name != "config.json":
            (overlay / name).symlink_to(path)
    (overlay / "base_vlm").symlink_to(Path(backbone).resolve(), target_is_directory=True)
    (overlay / "config.json").write_text(
        json.dumps({**root, "vlm_name_or_path": str(Path(backbone).resolve())})
    )
    engine = staging / "edge_llm/engine"
    prefix = [
        builder_python(package, target),
        "-I",
        "-c",
        "from experimental.builder.cli import main; main()",
    ]
    common = [
        "--model-dir",
        str(overlay),
        "--engine-dir",
        str(engine),
        "--plugin-path",
        package["plugin"],
        "--dense",
        "fp16",
        "--max-input-len",
        str(metadata["prompt_capacity"]),
        "--max-kv-cache-capacity",
        str(request.max_sequence_length),
        "--max-batch-size",
        str(request.max_batch_size),
        "--min-image-tokens",
        "160",
        "--max-image-tokens",
        str(request.max_batch_size * 16 * 192),
        "--max-image-tokens-per-image",
        "192",
    ]
    commands = [
        prefix + common + ["--components", "llm", "--reduced-vocab-dir", str(mapping)],
        prefix + common + ["--components", "visual,action"],
    ]
    if getattr(request, "verbose", False):
        for command in commands:
            command.append("--verbose")
    with log_path.open("a", encoding="utf-8") as log:
        for command in commands:
            log.write(shlex.join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=staging)
    # The backbone template drops structured history. This provider-equivalent
    # template preserves it and opens the trained reasoning continuation.
    (engine / "chat_template.jinja").write_bytes(DRIVING_CHAT_TEMPLATE.encode("utf-8"))
    required = (
        "llm.engine",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "visual/visual.engine",
        "visual/config.json",
        "action/action.engine",
        "action/config.json",
        "vocab_map.safetensors",
    )
    for name in required:
        path = engine / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Edge builder did not produce required Alpamayo artifact: {name}")
    config = json.loads((engine / "config.json").read_text())
    if config.get("reduced_vocab_size") != len(vocabulary):
        raise ValueError(
            "Edge output vocabulary does not implement the complete provider token mask"
        )
    files = {}
    for path in sorted(engine.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Edge output must not contain symlinks: {path}")
        if path.is_file():
            files[path.relative_to(staging).as_posix()] = path
    files.update({f"edge_llm/checkpoint/{name}": path for name, path in checkpoint.items()})
    return files, {
        "version": 1,
        "edge_revision": EDGE_REVISION,
        "target": target,
        "precision": "fp16",
        "max_sequence_length": request.max_sequence_length,
        "max_input_length": metadata["prompt_capacity"],
        "max_batch_size": request.max_batch_size,
        "excluded_output_token_interval": [start, start + count],
        "output_vocabulary_size": len(vocabulary),
        "artifacts": list(files),
        "runtime": {**metadata, "execution": "edge_llm"},
    }


def publish(request, writer, files: dict, marker: dict) -> None:
    """Publish only complete offload artifacts; never retry failed publication."""
    writer.set_header(family="alpamayo", task=request.task, backend=request.backend)
    for name, path in files.items():
        with path.open("rb") as source, writer.open_section(name) as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    writer.add_json("runtime.json", marker["runtime"])
    writer.add_json(
        "edge_llm.json", {key: value for key, value in marker.items() if key != "runtime"}
    )
