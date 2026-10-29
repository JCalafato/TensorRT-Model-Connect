/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::diffusion_gemma {

Sampler::Sampler(SamplingSettings settings, std::uint64_t seed)
    : settings_(settings), random_(seed) {
    if (settings_.vocabulary_size <= 0 || settings_.canvas_length <= 0 ||
        settings_.max_denoising_steps <= 0 || settings_.stability_threshold < 0 ||
        !std::isfinite(settings_.t_max) || !std::isfinite(settings_.t_min) ||
        settings_.t_min <= 0 || settings_.t_max < settings_.t_min ||
        !std::isfinite(settings_.entropy_bound) || settings_.entropy_bound < 0 ||
        !std::isfinite(settings_.confidence_threshold) || settings_.confidence_threshold < 0) {
        throw std::invalid_argument("DiffusionGemma diffusion sampling settings are invalid");
    }
    if (static_cast<std::size_t>(settings_.canvas_length) >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(settings_.vocabulary_size)) {
        throw std::invalid_argument("DiffusionGemma canvas geometry overflows");
    }
}

std::vector<std::int32_t> Sampler::random_canvas() {
    std::uniform_int_distribution<std::int32_t> uniform(0, settings_.vocabulary_size - 1);
    std::vector<std::int32_t> result(static_cast<std::size_t>(settings_.canvas_length));
    for (auto& token : result) {
        token = uniform(random_);
    }
    return result;
}

std::vector<std::int32_t> Sampler::initialize_canvas() {
    history_.clear();
    return random_canvas();
}

float Sampler::temperature(std::int32_t remaining_steps) const {
    if (remaining_steps <= 0 || remaining_steps > settings_.max_denoising_steps) {
        throw std::invalid_argument("DiffusionGemma remaining denoising steps are out of range");
    }
    // The reference iterates N..1, not N-1..0. The last step need not reach t_min.
    return settings_.t_min + (settings_.t_max - settings_.t_min) *
                                 (static_cast<float>(remaining_steps) /
                                  static_cast<float>(settings_.max_denoising_steps));
}

bool Sampler::update_stopping(const DenoisingStep& step) {
    const auto required = static_cast<std::size_t>(settings_.stability_threshold);
    bool stable = history_.size() == required;
    for (const auto& previous : history_) {
        stable = stable && previous == step.argmax_canvas;
    }
    if (required > 0) {
        history_.push_back(step.argmax_canvas);
        if (history_.size() > required) {
            history_.pop_front();
        }
    }
    double entropy = std::accumulate(step.entropy.begin(), step.entropy.end(), 0.0);
    entropy /= static_cast<double>(settings_.canvas_length);
    return stable && entropy < static_cast<double>(settings_.confidence_threshold);
}

DenoisingStep Sampler::step(const float* logits, std::size_t count, std::int32_t remaining_steps) {
    const auto rows = static_cast<std::size_t>(settings_.canvas_length);
    const auto vocabulary = static_cast<std::size_t>(settings_.vocabulary_size);
    const float scale = temperature(remaining_steps);
    if (logits == nullptr || count != rows * vocabulary) {
        throw std::invalid_argument("DiffusionGemma logits do not match the entire canvas");
    }
    DenoisingStep result;
    result.argmax_canvas.resize(rows);
    result.entropy.resize(rows);
    result.accepted.resize(rows);
    std::vector<std::int32_t> sampled(rows);
    // Only one vocabulary row of host workspace, not an additional full-canvas
    // probability matrix. The engine separately computes next soft embeddings.
    std::vector<float> weights(vocabulary);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* source = logits + row * vocabulary;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < vocabulary; ++token) {
            float value = source[token] / scale;
            if (std::isnan(value) || value == std::numeric_limits<float>::infinity()) {
                throw std::invalid_argument(
                    "DiffusionGemma logits contain NaN or positive infinity");
            }
            weights[token] = value;
            if (value > maximum) {
                maximum = value;
                result.argmax_canvas[row] = static_cast<std::int32_t>(token);
            }
        }
        if (!std::isfinite(maximum)) {
            throw std::invalid_argument("DiffusionGemma logits row has no finite candidate");
        }
        double total = 0;
        for (auto& value : weights) {
            value = std::exp(value - maximum);
            total += value;
        }
        const double draw = std::generate_canonical<double, 53>(random_) * total;
        double cumulative = 0;
        double entropy = 0;
        bool selected = false;
        for (std::size_t token = 0; token < vocabulary; ++token) {
            cumulative += weights[token];
            if (!selected && draw < cumulative) {
                sampled[row] = static_cast<std::int32_t>(token);
                selected = true;
            }
            double probability = weights[token] / total;
            if (probability > 0) {
                entropy -= probability * std::log(probability);
            }
        }
        if (!selected) {
            sampled[row] = result.argmax_canvas[row];
        }
        result.entropy[row] = static_cast<float>(entropy);
    }
    std::vector<std::size_t> order(rows);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&result](std::size_t lhs, std::size_t rhs) {
        return result.entropy[lhs] < result.entropy[rhs];
    });
    float cumulative = 0;
    for (std::size_t index : order) {
        cumulative += result.entropy[index];
        result.accepted[index] = cumulative - result.entropy[index] <= settings_.entropy_bound;
    }
    // Acceptance is recomputed each step; it is not a sticky converged prefix.
    // Draw noise for every position, then retain sampled IDs only where accepted.
    result.next_canvas = random_canvas();
    for (std::size_t index = 0; index < rows; ++index) {
        if (result.accepted[index]) {
            result.next_canvas[index] = sampled[index];
        }
    }
    result.stable_and_confident = update_stopping(result);
    // The runtime commits argmax_canvas, never the sampled/renoised next_canvas.
    return result;
}

} // namespace trtmc::diffusion_gemma
