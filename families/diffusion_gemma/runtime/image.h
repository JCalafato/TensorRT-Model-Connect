/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/language.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc::diffusion_gemma {
struct ImageConfig {
    int patch_size{0}, pooling_kernel{0}, soft_tokens{0}, hidden_size{0};
    int image_token{0}, begin_token{0}, end_token{0};
};
struct ImagePatches {
    std::vector<float> pixels;
    std::vector<std::int32_t> positions;
    int soft_tokens{0};
};
ImagePatches image_patches(const internal::ImageView& image, const ImageConfig& config);

class VisionEncoder {
  public:
    VisionEncoder(std::unique_ptr<ITrtModule> module, ImageConfig config);
    const ImageConfig& config() const { return config_; }
    std::vector<float> encode(const ImagePatches& input);

  private:
    std::unique_ptr<ITrtModule> module_;
    ImageConfig config_;
};
} // namespace trtmc::diffusion_gemma
