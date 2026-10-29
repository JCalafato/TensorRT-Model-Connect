/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trajectory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::alpamayo {
namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(std::string("Alpamayo trajectory: ") + message);
}

bool finite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

// The provider fits all history velocities, not just the last displacement.
// Solve the small SPD history normal equation with a Cholesky factorization.
// Double intermediates stabilize this host-side geometric fit; no GPU kernels.
std::vector<double> solve(std::vector<double> matrix, std::vector<double> rhs) {
    const auto n = rhs.size();
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            double value = matrix[row * n + col];
            for (std::size_t k = 0; k < col; ++k)
                value -= matrix[row * n + k] * matrix[col * n + k];
            if (row == col) {
                require(std::isfinite(value) && value > 0, "history fit is not positive definite");
                matrix[row * n + col] = std::sqrt(value);
            } else {
                matrix[row * n + col] = value / matrix[col * n + col];
            }
        }
    }
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < row; ++col)
            rhs[row] -= matrix[row * n + col] * rhs[col];
        rhs[row] /= matrix[row * n + row];
    }
    for (std::size_t row = n; row-- > 0;) {
        for (std::size_t col = row + 1; col < n; ++col)
            rhs[row] -= matrix[col * n + row] * rhs[col];
        rhs[row] /= matrix[row * n + row];
    }
    return rhs;
}

} // namespace

TrajectoryDecoder::TrajectoryDecoder(ActionSpaceParameters p) : parameters_(p) {
    const auto limit = std::numeric_limits<std::size_t>::max() / (9 * sizeof(float));
    require(p.waypoints > 0 && p.waypoints <= limit, "invalid waypoint count");
    require(p.history_points >= 2 && p.history_points <= limit &&
                p.history_points <=
                    std::numeric_limits<std::size_t>::max() / sizeof(double) / p.history_points,
            "invalid history count");
    require(std::isfinite(p.dt) && p.dt > 0, "invalid timestep");
    require(std::isfinite(p.velocity_smoothness) && p.velocity_smoothness >= 0 &&
                std::isfinite(p.velocity_ridge) && p.velocity_ridge >= 0,
            "invalid velocity regularization");
    require(std::isfinite(p.acceleration_mean) && std::isfinite(p.acceleration_std) &&
                p.acceleration_std > 0 && std::isfinite(p.curvature_mean) &&
                std::isfinite(p.curvature_std) && p.curvature_std > 0,
            "invalid checkpoint normalization");
}

void TrajectoryDecoder::set_history(const std::vector<float>& xyz,
                                    const std::vector<float>& rotations) {
    const auto n = parameters_.history_points;
    require(xyz.size() == n * 3 && rotations.size() == n * 9, "incomplete trajectory history");
    require(finite(xyz) && finite(rotations), "history must be finite");
    std::vector<float> angles(n);
    float cumulative = 0;
    const float first = std::atan2(rotations[3], rotations[0]);
    float previous = first;
    angles[0] = first;
    for (std::size_t i = 1; i < n; ++i) {
        const float yaw = std::atan2(rotations[9 * i + 3], rotations[9 * i]);
        const float delta = yaw - previous;
        cumulative += std::atan2(std::sin(delta), std::cos(delta));
        angles[i] = first + cumulative;
        previous = yaw;
    }
    std::vector<double> normal(n * n, 0), rhs(n, 0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const std::array<float, 2> left{std::cos(angles[i]), std::sin(angles[i])};
        const std::array<float, 2> right{std::cos(angles[i + 1]), std::sin(angles[i + 1])};
        for (std::size_t axis = 0; axis < 2; ++axis) {
            const float displacement = xyz[(i + 1) * 3 + axis] - xyz[i * 3 + axis];
            const double target = displacement * (2.0 / parameters_.dt);
            normal[i * n + i] += static_cast<double>(left[axis]) * left[axis];
            normal[(i + 1) * n + i + 1] += static_cast<double>(right[axis]) * right[axis];
            const double off = static_cast<double>(left[axis]) * right[axis];
            normal[i * n + i + 1] += off;
            normal[(i + 1) * n + i] += off;
            rhs[i] += left[axis] * target;
            rhs[i + 1] += right[axis] * target;
        }
    }
    // Third-order Tikhonov smoothing: lambda / dt^6 * D3^T D3.
    const double scale = parameters_.velocity_smoothness / std::pow(parameters_.dt, 6);
    constexpr std::array<double, 4> difference{-1, 3, -3, 1};
    for (std::size_t i = 0; i + 3 < n; ++i)
        for (std::size_t row = 0; row < 4; ++row)
            for (std::size_t col = 0; col < 4; ++col)
                normal[(i + row) * n + i + col] += scale * difference[row] * difference[col];
    for (std::size_t i = 0; i < n; ++i)
        normal[i * n + i] += parameters_.velocity_ridge;
    const float velocity = static_cast<float>(solve(std::move(normal), std::move(rhs)).back());
    require(std::isfinite(velocity), "history velocity is not finite");
    // Rejected histories leave the previous valid state intact.
    initial_velocity_ = velocity;
    height_ = xyz.back();
    ready_ = true;
}

PhysicalTrajectory TrajectoryDecoder::decode(const std::vector<float>& actions) const {
    require(ready_, "history must be set before decoding actions");
    require(actions.size() == parameters_.waypoints * 2, "incomplete action sequence");
    require(finite(actions), "actions must be finite");
    PhysicalTrajectory output{std::vector<float>(parameters_.waypoints * 3, 0),
                              std::vector<float>(parameters_.waypoints * 9, 0), initial_velocity_};
    const float dt = static_cast<float>(parameters_.dt);
    const float half_dt = static_cast<float>(0.5 * parameters_.dt);
    const float half_dt2 = static_cast<float>(0.5 * parameters_.dt * parameters_.dt);
    float acceleration_sum = 0, yaw_velocity_sum = 0, yaw_acceleration_sum = 0;
    float previous_velocity = initial_velocity_, previous_yaw = 0;
    float x_before = 0, x_after = 0, y_before = 0, y_after = 0;
    for (std::size_t i = 0; i < parameters_.waypoints; ++i) {
        const float a =
            actions[2 * i] * parameters_.acceleration_std + parameters_.acceleration_mean;
        const float k = actions[2 * i + 1] * parameters_.curvature_std + parameters_.curvature_mean;
        acceleration_sum += a * dt;
        const float velocity = initial_velocity_ + acceleration_sum;
        yaw_velocity_sum += k * previous_velocity * dt;
        yaw_acceleration_sum += k * a * half_dt2;
        const float yaw = yaw_velocity_sum + yaw_acceleration_sum;
        const float cosine = std::cos(yaw), sine = std::sin(yaw);
        x_before += previous_velocity * std::cos(previous_yaw) * half_dt;
        x_after += velocity * cosine * half_dt;
        y_before += previous_velocity * std::sin(previous_yaw) * half_dt;
        y_after += velocity * sine * half_dt;
        output.xyz[3 * i] = x_before + x_after;
        output.xyz[3 * i + 1] = y_before + y_after;
        output.xyz[3 * i + 2] = height_;
        output.rotations[9 * i] = cosine;
        output.rotations[9 * i + 1] = -sine;
        output.rotations[9 * i + 3] = sine;
        output.rotations[9 * i + 4] = cosine;
        output.rotations[9 * i + 8] = 1;
        previous_velocity = velocity;
        previous_yaw = yaw;
    }
    if (!finite(output.xyz) || !finite(output.rotations))
        throw std::runtime_error("Alpamayo trajectory: non-finite decoded pose");
    return output;
}

} // namespace trtmc::alpamayo
