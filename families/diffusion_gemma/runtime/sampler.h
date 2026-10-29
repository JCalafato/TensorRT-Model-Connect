/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace trtmc::diffusion_gemma {

struct SamplingSettings {
    std::int32_t vocabulary_size{0};
    std::int32_t canvas_length{256};
    std::int32_t max_denoising_steps{48};
    std::int32_t stability_threshold{1};
    float t_max{0.8F};
    float t_min{0.4F};
    float entropy_bound{0.1F};
    float confidence_threshold{0.005F};
};

struct DenoisingStep {
    std::vector<std::int32_t> next_canvas;
    std::vector<std::int32_t> argmax_canvas;
    std::vector<float> entropy;
    std::vector<bool> accepted;
    bool stable_and_confident{false};
};

// Whole-canvas diffusion sampler. Seed reproducibility is native-runtime-local;
// equal seeds do not imply equal Torch/C++ categorical RNG streams.
class Sampler {
  public:
    explicit Sampler(SamplingSettings settings, std::uint64_t seed);
    std::vector<std::int32_t> initialize_canvas();
    float temperature(std::int32_t remaining_steps) const;
    DenoisingStep step(const float* logits, std::size_t count, std::int32_t remaining_steps);

  private:
    std::vector<std::int32_t> random_canvas();
    bool update_stopping(const DenoisingStep& step);
    SamplingSettings settings_;
    std::mt19937_64 random_;
    std::deque<std::vector<std::int32_t>> history_;
};

} // namespace trtmc::diffusion_gemma
