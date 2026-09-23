# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama-owned CLI options for explicit paired Edge execution."""

import argparse
from pathlib import Path

from .config import BuildExecutionInputs, NamedCheckpoint, with_execution


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Register options only when the Llama family has been resolved."""
    parser.add_argument("--execution-variant", choices=("eagle3",),
                        help="Explicit Llama paired execution mode")
    parser.add_argument("--companion", action="append", default=[], metavar="ROLE=LOCAL_DIR",
                        help="Explicit local draft checkpoint; exactly one draft role is required")


def _execution_inputs(args: argparse.Namespace) -> BuildExecutionInputs | None:
    """Parse only explicit local inputs; no variant list or model acquisition."""
    if args.command != "build":
        return None
    if args.execution_variant is None:
        if args.companion:
            raise ValueError("--companion requires --execution-variant")
        return None
    checkpoints = []
    for value in args.companion:
        role, separator, directory = value.partition("=")
        if not separator or not role or not directory:
            raise ValueError("--companion must be ROLE=LOCAL_DIR")
        if "://" in directory:
            raise ValueError("--companion requires a local directory, not a URI")
        checkpoints.append(NamedCheckpoint(role, Path(directory)))
    return BuildExecutionInputs(args.execution_variant, tuple(checkpoints))


def prepare_build_request(request, args):
    """Attach a validated family-owned recipe before importing the GPU builder."""
    execution = _execution_inputs(args)
    return request if execution is None else with_execution(request, execution)
