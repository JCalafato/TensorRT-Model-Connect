/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <vector>

namespace trtmc::alpamayo {

struct ActionSpaceParameters {
    std::size_t waypoints;
    std::size_t history_points;
    double dt;
    double velocity_smoothness;
    double velocity_ridge;
    float acceleration_mean;
    float acceleration_std;
    float curvature_mean;
    float curvature_std;
};

struct PhysicalTrajectory {
    std::vector<float> xyz;       // [waypoints, 3], current ego frame
    std::vector<float> rotations; // [waypoints, 3, 3], row-major
    float initial_velocity;
};

// Family-internal provider action space, not a substitute public Task contract.
// Fit the history once; all sampled action sequences reuse that initial state.
class TrajectoryDecoder {
  public:
    explicit TrajectoryDecoder(ActionSpaceParameters parameters);
    void set_history(const std::vector<float>& xyz, const std::vector<float>& rotations);
    PhysicalTrajectory decode(const std::vector<float>& normalized_actions) const;

  private:
    ActionSpaceParameters parameters_;
    float initial_velocity_{0};
    float height_{0};
    bool ready_{false};
};

} // namespace trtmc::alpamayo
