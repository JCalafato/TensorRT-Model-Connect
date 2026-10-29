/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace trtmc::alpamayo {

// Broadcast the original prompt exclusions to noncausal action queries and
// exclude reasoning positions beyond each row's first future-marker offset.
// Host-only construction also validates malformed handoffs before allocation.
std::vector<float> make_action_prefix_mask(const std::vector<std::vector<float>>& prompt_mask,
                                           const std::vector<int64_t>& offsets, int64_t length,
                                           int64_t waypoints);

// One velocity evaluation in the action flow. Prefix KV, padding mask and
// rotary tables are immutable across steps and are uploaded once per context.
// This internal stage does not implement reasoning or physical pose decoding.
class ActionStep {
  public:
    explicit ActionStep(std::unique_ptr<ITrtModule> module);

    // Requires all engine inputs except actions and timesteps. Validation is
    // completed before replacing a previous context, so rejected requests do
    // not poison the next step. Host inputs may be released after this call.
    void set_context(const TensorMap& inputs);

    // Use the actual reasoning cache and first future-marker offsets. This
    // owns provider action positions and masks: early-finished rows cannot
    // attend their trailing reasoning tokens, and action tokens are noncausal.
    void set_generated_context(const TensorMap& prefix, const std::vector<int64_t>& offsets,
                               const std::vector<int64_t>& rope_deltas,
                               const std::vector<std::vector<float>>& prompt_mask,
                               float rope_theta);

    // Returns FP32 velocity in [batch, waypoints, 2] order. The engine and
    // immutable context remain available for subsequent denoising steps.
    std::vector<float> evaluate(const Tensor& actions, const Tensor& timesteps);

    // Provider Euler flow over FP32 normalized actions. The caller supplies
    // initial Gaussian noise; this stage does not define a public seed policy.
    // Reuses the same immutable prefix context for every velocity evaluation.
    std::vector<float> integrate(const Tensor& initial_noise, int steps = 10);

  private:
    void validate(const std::string& name, const Tensor& tensor,
                  const std::vector<int64_t>& expected) const;

    // Destroy the module before the buffers to which it is externally bound.
    std::unordered_map<std::string, DeviceTensor> context_;
    std::unique_ptr<ITrtModule> module_;
    int64_t batch_{0};
    int64_t waypoints_{0};
    int64_t heads_{0};
    int64_t head_dim_{0};
    int64_t layers_{0};
    bool ready_{false};
};

} // namespace trtmc::alpamayo
