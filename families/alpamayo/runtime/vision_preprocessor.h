/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::alpamayo {

struct VisionPreprocessConfig {
    int32_t patch_size;
    int32_t temporal_patch_size;
    int32_t spatial_merge_size;
    int32_t num_position_embeddings;
    int32_t head_width;
    int64_t min_pixels;
    int64_t max_pixels;
    int64_t patch_capacity;
    std::array<float, 3> image_mean;
    std::array<float, 3> image_std;
    double rescale_factor;
};

// Buffers own their data; views are obtained after any move. One still image
// repeats its frame for temporal patching and yields a single grid time step.
struct PreparedVisionInput {
    std::array<int64_t, 3> grid;
    int64_t flat_patch;
    int64_t head_width;
    std::vector<float> pixels;
    std::vector<int32_t> position_indices;
    std::vector<float> position_weights;
    std::vector<float> mask;
    std::vector<float> cosine;
    std::vector<float> sine;

    TensorMap tensors();
};

// Native CPU preparation for decoded, contiguous HWC uint8 RGB images. The
// caller supplies processor metadata and the actual engine's patch capacity.
// No request is silently shrunk or truncated to fit an engine profile.
class VisionPreprocessor {
  public:
    explicit VisionPreprocessor(VisionPreprocessConfig config);
    PreparedVisionInput prepare(const uint8_t* rgb, std::size_t bytes, int32_t height,
                                int32_t width) const;

  private:
    VisionPreprocessConfig config_;
    int32_t position_side_;
};

} // namespace trtmc::alpamayo
