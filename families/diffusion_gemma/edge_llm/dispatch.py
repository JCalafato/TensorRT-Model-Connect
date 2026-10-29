# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Do not advertise the pinned Edge builder's known command-failing route.

The same unconditional graph-construction failure affects source and wheel
installations. A native bundle is built instead; no Edge engine is published
and no inference-time retry changes request semantics.
"""

from importlib import metadata
import json
import logging

from tensorrt_model_connect.build import cmake_prefixes

EDGE_REVISION = "95515c2f87fba8982db5a519f9022277667b3cc9"
_LOG = logging.getLogger(__name__)


def installed_release():
    """Find standard wheel metadata or a source-built CMake installation.

    Discovery never imports the optional CUDA runtime, installs packages, or
    depends on the executing GPU. Invalid optional installations do not make
    the native mathematical graph unavailable.
    """
    try:
        return metadata.version("tensorrt-edgellm")
    except metadata.PackageNotFoundError:
        pass
    for prefix in cmake_prefixes():
        manifest = prefix / "share/trtmc/edge-llm.json"
        if not manifest.is_file():
            continue
        package = json.loads(manifest.read_text(encoding="utf-8"))
        if package.get("schema_version") != 1:
            raise ValueError("Unsupported Edge installation manifest schema")
        if package.get("version") == "0.11.0" and package.get("revision") != EDGE_REVISION:
            raise ValueError("Edge 0.11.0 source installation does not match the pinned revision")
        return package.get("version", "unknown")
    return None


def build(request, writer, native):
    """Keep the full unchanged native request runnable on every TRT target.

    A real invocation of the unmodified 0.11.0 NetworkModule.build API with
    synthetic weights fails at the undefined token_phase in the shared text
    backbone, before serialization. This is not a full-checkpoint quality
    result. Do not repeat that unchanged failing command for every request.
    """
    try:
        release = installed_release()
    except (OSError, ValueError, TypeError, AttributeError) as error:
        _LOG.warning(
            "DiffusionGemma optional Edge discovery failed: %s; using native TensorRT", error
        )
    else:
        if release == "0.11.0":
            _LOG.warning(
                "DiffusionGemma Edge 0.11.0 graph construction has a known "
                "NameError: token_phase is undefined. Using the complete native "
                "TensorRT path with the unchanged request; no Edge engine is advertised."
            )
        elif release is not None:
            _LOG.warning(
                "DiffusionGemma Edge release %s is outside this integration's 0.11.0 scope; "
                "using native TensorRT",
                release,
            )
    return native(request, writer)
