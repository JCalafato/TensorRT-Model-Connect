/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "action_step.h"
#include "history_tokenizer.h"
#include "text_decoder.h"
#include "trajectory.h"
#include "vision_preprocessor.h"

namespace trtmc::alpamayo {

struct DrivingHistoryInput {
    std::vector<float> xyz, rotations;
};

struct DrivingImageInput {
    const uint8_t* rgb;
    std::size_t bytes;
    int32_t height, width;
};

struct DrivingOptions {
    ImagePositionConfig positions;
    ActionSpaceParameters action_space;
    ReasoningOptions reasoning;
    int denoising_steps{10};
};

struct DrivingSample {
    std::vector<int32_t> generated_tokens;
    DrivingText text;
    std::vector<float> normalized_actions;
    PhysicalTrajectory trajectory;
};

// Native family-internal orchestration, not a substitute public Task. Rows are
// explicit sampled observations: camera order and history are preserved. Noise
// is FP32 [rows, waypoints, 2]; its RNG policy belongs to the eventual Task owner.
// These boundaries also let validation isolate the same real generated cache
// without rerunning vision/reasoning or substituting an unconditioned expert.
std::vector<DrivingSample> finish_driving(ActionStep& action, const DrivingPrompt& prompt,
                                          ReasoningResult& reasoning,
                                          const std::vector<DrivingHistoryInput>& histories,
                                          const DrivingOptions& options,
                                          const Tensor& initial_noise);

std::vector<DrivingSample>
generate_driving(TextDecoder& text, ActionStep& action, const DrivingPrompt& prompt,
                 const std::vector<std::vector<std::array<int64_t, 3>>>& grids,
                 const std::vector<VisionFeatures>& features,
                 const std::vector<DrivingHistoryInput>& histories, const DrivingOptions& options,
                 const Tensor& initial_noise);

std::vector<DrivingSample> infer_driving(const VisionPreprocessor& processor, VisionEncoder& vision,
                                         TextDecoder& text, ActionStep& action,
                                         const DrivingPrompt& prompt,
                                         const std::vector<std::vector<DrivingImageInput>>& images,
                                         const std::vector<DrivingHistoryInput>& histories,
                                         const DrivingOptions& options,
                                         const Tensor& initial_noise);

} // namespace trtmc::alpamayo
