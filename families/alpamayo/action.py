# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo per-waypoint action and flow-velocity projections."""

import math

import numpy as np
import tensorrt as trt

from .graph import Graph


class ActionGraph(Graph):
    def __init__(self, network, weights, config, expert_width, precision):
        super().__init__(network, weights, precision)
        self.config, self.expert_width = config, expert_width

    def fourier(self, value, name):
        """Provider FourierEncoderV2: FP32 phase with checkpoint frequencies."""
        rank = len(value.shape)
        half = self.config.fourier_features // 2
        frequencies = self.weight(name + ".freqs", (1, half))
        frequencies = self.reshape(frequencies, (1,) * (rank - 1) + (half,))
        phase = self.op(
            self.cast(value, trt.float32),
            self.cast(frequencies, trt.float32),
            trt.ElementWiseOperation.PROD,
        )
        phase = self.op(phase, self.scalar(2.0, rank), trt.ElementWiseOperation.PROD)
        phase = self.op(phase, self.scalar(math.pi, rank), trt.ElementWiseOperation.PROD)
        sine = self.net.add_unary(phase, trt.UnaryOperation.SIN).get_output(0)
        cosine = self.net.add_unary(phase, trt.UnaryOperation.COS).get_output(0)
        return self.op(
            self.concat([sine, cosine], rank - 1),
            self.scalar(math.sqrt(2), rank),
            trt.ElementWiseOperation.PROD,
        )

    def input_projection(self, actions, timesteps):
        """(B, waypoints, 2), (B, 1, 1) -> FP32 (B, waypoints, expert_width)."""
        if tuple(actions.shape[1:]) != (self.config.num_waypoints, 2):
            raise ValueError("Action projection requires a complete two-channel waypoint sequence")
        if len(timesteps.shape) != 3 or tuple(timesteps.shape[1:]) != (1, 1):
            raise ValueError("Flow timesteps must have shape (batch, 1, 1)")
        stem = "action_in_proj"
        features = []
        for channel in range(2):
            index = self.const(np.array([channel], np.int32))
            component = self.net.add_gather(actions, index, 2).get_output(0)
            features.append(self.fourier(component, f"{stem}.sinus.{channel}"))
        time = self.fourier(timesteps, f"{stem}.timestep_fourier_encoder")
        # Broadcast each sample's timestep across its waypoints, not across batches.
        zeros = self.const(np.zeros((1, self.config.num_waypoints, 1), np.float32))
        features.append(self.op(time, zeros, trt.ElementWiseOperation.SUM))
        hidden = self.concat(features, 2)
        hidden = self.silu(
            self.linear(hidden, f"{stem}.encoder.trunk.0", self.config.projection_hidden_size)
        )
        for index in range(self.config.projection_layers):
            layer = 3 * index + 2
            hidden = self.rms_norm(hidden, f"{stem}.encoder.trunk.{layer}", 1e-5)
            final = index + 1 == self.config.projection_layers
            width = self.expert_width if final else self.config.projection_hidden_size
            hidden = self.linear(hidden, f"{stem}.encoder.trunk.{layer + 1}", width)
            if not final:
                hidden = self.silu(hidden)
        return self.layer_norm(hidden, f"{stem}.norm", 1e-5)

    def output_projection(self, hidden):
        """Expert hidden states -> normalized acceleration/curvature flow velocity."""
        if hidden.shape[-1] != self.expert_width:
            raise ValueError("Action output projection requires the expert hidden width")
        return self.linear(hidden, "action_out_proj", 2)
