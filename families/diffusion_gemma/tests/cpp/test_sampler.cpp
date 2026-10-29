/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampler.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using trtmc::diffusion_gemma::Sampler;
using trtmc::diffusion_gemma::SamplingSettings;

namespace {
void require(bool condition, const char* description) {
    if (!condition) {
        throw std::runtime_error(description);
    }
}

template <typename Function>
void require_invalid(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid sampler input was accepted");
}

SamplingSettings settings() {
    SamplingSettings result;
    result.vocabulary_size = 3;
    result.canvas_length = 4;
    result.max_denoising_steps = 4;
    result.stability_threshold = 2;
    result.t_max = 1.F;
    result.t_min = .5F;
    result.entropy_bound = .005F;
    return result;
}

void test_reference_entropy_and_acceptance() {
    Sampler sampler(settings(), 17);
    const std::vector<float> logits{0, 0, 0, 10, 0, -10, 2, 1, 0, -2, 2, 0};
    auto result = sampler.step(logits.data(), logits.size(), 4);
    // Actual HF5.11 EntropyBoundSampler/Categorical fixture, not a copy of
    // the production C++ formula. Provenance: ../data/sampler-reference.json.
    const std::vector<float> reference{1.0986123085F, .000499796239F, .832395494F, .441057503F};
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require(std::abs(result.entropy[i] - reference[i]) < 1.e-6F,
                "HF entropy comparison failed");
    }
    require(result.accepted == std::vector<bool>({false, true, false, true}),
            "HF entropy selection differs");
    require(result.argmax_canvas == std::vector<std::int32_t>({0, 0, 0, 1}),
            "final argmax canvas differs");
    require(!result.stable_and_confident, "uncertain first canvas stopped early");
    require(sampler.temperature(4) == 1.F && sampler.temperature(1) == .625F,
            "N..1 temperature schedule differs");
}

void test_stability_and_reset() {
    Sampler sampler(settings(), 42);
    std::vector<float> logits{20, 0, -20, 20, 0, -20, 20, 0, -20, 20, 0, -20};
    require(!sampler.step(logits.data(), logits.size(), 4).stable_and_confident,
            "history initialized as stable");
    require(!sampler.step(logits.data(), logits.size(), 3).stable_and_confident,
            "history stopped one step early");
    require(sampler.step(logits.data(), logits.size(), 2).stable_and_confident,
            "stable confident canvas did not stop");
    sampler.initialize_canvas();
    require(!sampler.step(logits.data(), logits.size(), 4).stable_and_confident,
            "new canvas retained stopping history");
    auto no_history = settings();
    no_history.stability_threshold = 0;
    Sampler immediate(no_history, 42);
    require(immediate.step(logits.data(), logits.size(), 4).stable_and_confident,
            "history0 must permit immediate confidence stopping");
    logits[9] = logits[10] = logits[11] = 0;
    require(!immediate.step(logits.data(), logits.size(), 4).stable_and_confident,
            "uncertain canvas tail was ignored");
}

void test_native_seed_and_nonsticky_sampling() {
    auto options = settings();
    options.entropy_bound = 0;
    Sampler left(options, 11), right(options, 11);
    require(left.initialize_canvas() == right.initialize_canvas(),
            "native initial seed is not repeatable");
    std::vector<float> uniform(12, 0.F);
    bool saw_renoised_not_argmax = false;
    for (int i = 0; i < 20; ++i) {
        auto lhs = left.step(uniform.data(), uniform.size(), 4);
        auto rhs = right.step(uniform.data(), uniform.size(), 4);
        require(lhs.next_canvas == rhs.next_canvas && lhs.argmax_canvas == rhs.argmax_canvas,
                "native seeded requests differ");
        require(lhs.accepted == std::vector<bool>({true, false, false, false}),
                "zero bound must accept only first positive-entropy token");
        saw_renoised_not_argmax = saw_renoised_not_argmax || lhs.next_canvas != lhs.argmax_canvas;
        for (auto token : lhs.next_canvas) {
            require(token >= 0 && token < 3, "renoising escaped vocabulary");
        }
    }
    require(saw_renoised_not_argmax, "diffusion sampling was replaced by greedy argmax");
    std::vector<float> confident{20, 0, -20, 20, 0, -20, 20, 0, -20, 20, 0, -20};
    auto relaxed = settings();
    Sampler adaptive(relaxed, 7);
    require(adaptive.step(confident.data(), confident.size(), 4).accepted ==
                std::vector<bool>(4, true),
            "confident tokens were not accepted");
    require(adaptive.step(uniform.data(), uniform.size(), 3).accepted ==
                std::vector<bool>({true, false, false, false}),
            "accepted positions incorrectly became sticky");
}

void test_reject_invalid_inputs() {
    auto invalid = settings();
    invalid.canvas_length = 0;
    require_invalid([&] { Sampler sampler(invalid, 0); });
    invalid = settings();
    invalid.t_min = 2;
    require_invalid([&] { Sampler sampler(invalid, 0); });
    Sampler sampler(settings(), 1);
    std::vector<float> values(12, 0.F);
    require_invalid([&] { sampler.step(nullptr, values.size(), 1); });
    require_invalid([&] { sampler.step(values.data(), values.size() - 1, 1); });
    require_invalid([&] { sampler.temperature(0); });
    require_invalid([&] { sampler.temperature(5); });
    values[0] = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { sampler.step(values.data(), values.size(), 1); });
    values[0] = std::numeric_limits<float>::infinity();
    require_invalid([&] { sampler.step(values.data(), values.size(), 1); });
    values[0] = values[1] = values[2] = -std::numeric_limits<float>::infinity();
    require_invalid([&] { sampler.step(values.data(), values.size(), 1); });
}
} // namespace

int main() {
    try {
        test_reference_entropy_and_acceptance();
        test_stability_and_reset();
        test_native_seed_and_nonsticky_sampling();
        test_reject_invalid_inputs();
        std::cout << "DiffusionGemma sampler:4 test groups passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
