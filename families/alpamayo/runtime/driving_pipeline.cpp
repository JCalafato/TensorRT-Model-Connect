/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "driving_pipeline.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace trtmc::alpamayo {
namespace {

void require(bool valid, const char* message) {
    if (!valid)
        throw std::invalid_argument(std::string("Alpamayo driving: ") + message);
}

std::vector<TrajectoryDecoder> prepare_histories(const std::vector<DrivingHistoryInput>& histories,
                                                 const DrivingOptions& options,
                                                 const Tensor& noise) {
    const auto batch = histories.size(), waypoints = options.action_space.waypoints;
    require(batch > 0 && batch <= static_cast<uint64_t>(INT64_MAX) && waypoints > 0 &&
                waypoints <= static_cast<uint64_t>(INT64_MAX) &&
                waypoints <= std::numeric_limits<std::size_t>::max() / sizeof(float) / 2 / batch,
            "invalid batch or action size");
    require(options.denoising_steps > 0, "denoising steps must be positive");
    require(noise.data && noise.dtype == DType::kFloat32 &&
                noise.shape == std::vector<int64_t>{static_cast<int64_t>(batch),
                                                    static_cast<int64_t>(waypoints), 2},
            "initial noise must match history batch and action dimensions");
    const auto* values = static_cast<const float*>(noise.data);
    for (std::size_t i = 0; i < batch * waypoints * 2; ++i)
        require(std::isfinite(values[i]), "initial noise must be finite");
    std::vector<TrajectoryDecoder> decoders;
    decoders.reserve(batch);
    for (const auto& history : histories) {
        decoders.emplace_back(options.action_space);
        decoders.back().set_history(history.xyz, history.rotations);
    }
    return decoders;
}

} // namespace

std::vector<DrivingSample> finish_driving(ActionStep& action, const DrivingPrompt& prompt,
                                          ReasoningResult& reasoning,
                                          const std::vector<DrivingHistoryInput>& histories,
                                          const DrivingOptions& options,
                                          const Tensor& initial_noise) {
    auto decoders = prepare_histories(histories, options, initial_noise);
    const auto batch = histories.size(), row_size = options.action_space.waypoints * 2;
    require(reasoning.tokens.size() == batch && reasoning.action_offsets.size() == batch &&
                reasoning.rope_deltas.size() == batch && reasoning.generated > 0,
            "reasoning batch or generated count is invalid");
    std::vector<DrivingSample> result(batch);
    for (std::size_t row = 0; row < batch; ++row) {
        const auto& tokens = reasoning.tokens[row];
        require(static_cast<uint64_t>(reasoning.generated) <= tokens.size(),
                "generated suffix exceeds token sequence");
        result[row].generated_tokens.assign(tokens.end() - reasoning.generated, tokens.end());
        result[row].text = prompt.extract_text(result[row].generated_tokens);
    }
    action.set_generated_context(reasoning.cache.action_prefix(), reasoning.action_offsets,
                                 reasoning.rope_deltas, options.positions.rope_theta);
    const auto actions = action.integrate(initial_noise, options.denoising_steps);
    require(actions.size() == batch * row_size, "action output size is inconsistent");
    for (std::size_t row = 0; row < batch; ++row) {
        result[row].normalized_actions.assign(actions.begin() + row * row_size,
                                              actions.begin() + (row + 1) * row_size);
        result[row].trajectory = decoders[row].decode(result[row].normalized_actions);
    }
    return result;
}

std::vector<DrivingSample>
generate_driving(TextDecoder& text, ActionStep& action, const DrivingPrompt& prompt,
                 const std::vector<std::vector<std::array<int64_t, 3>>>& grids,
                 const std::vector<VisionFeatures>& features,
                 const std::vector<DrivingHistoryInput>& histories, const DrivingOptions& options,
                 const Tensor& initial_noise) {
    prepare_histories(histories, options, initial_noise);
    require(grids.size() == histories.size(), "camera grid batch differs from histories");
    std::vector<std::vector<float>> xyz;
    xyz.reserve(histories.size());
    for (const auto& history : histories)
        xyz.push_back(history.xyz);
    const auto batch = prompt.encode_batch(grids, xyz, options.reasoning.pad_token);
    auto prepared = text.prepare(batch.tokens, batch.image_grids, features, batch.attention_mask,
                                 options.positions);
    auto reasoning = text.generate(prepared.tensors(), prepared.positions.deltas,
                                   options.positions.rope_theta, options.reasoning);
    return finish_driving(action, prompt, reasoning, histories, options, initial_noise);
}

std::vector<DrivingSample> infer_driving(const VisionPreprocessor& processor, VisionEncoder& vision,
                                         TextDecoder& text, ActionStep& action,
                                         const DrivingPrompt& prompt,
                                         const std::vector<std::vector<DrivingImageInput>>& images,
                                         const std::vector<DrivingHistoryInput>& histories,
                                         const DrivingOptions& options,
                                         const Tensor& initial_noise) {
    prepare_histories(histories, options, initial_noise);
    require(images.size() == histories.size(), "camera batch differs from histories");
    std::vector<PreparedVisionInput> prepared;
    std::vector<std::vector<std::array<int64_t, 3>>> grids;
    std::vector<std::vector<float>> xyz;
    for (std::size_t row = 0; row < images.size(); ++row) {
        require(!images[row].empty(), "driving observations require camera images");
        grids.emplace_back();
        xyz.push_back(histories[row].xyz);
        for (const auto& image : images[row]) {
            prepared.push_back(
                processor.prepare(image.rgb, image.bytes, image.height, image.width));
            grids.back().push_back(prepared.back().grid);
        }
    }
    // Validate prompt framing/history/capacity before any visual execution.
    prompt.encode_batch(grids, xyz, options.reasoning.pad_token);
    std::vector<VisionFeatures> features;
    features.reserve(prepared.size());
    for (auto& frame : prepared)
        features.push_back(vision.encode(frame.tensors()));
    return generate_driving(text, action, prompt, grids, features, histories, options,
                            initial_noise);
}

} // namespace trtmc::alpamayo
