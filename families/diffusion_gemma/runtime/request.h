/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/internal/config.h"
#include "trtmc/internal/language.h"

namespace trtmc::diffusion_gemma {
std::string trim_chat_text(const std::string& text);
std::string format_prompt(const std::string& prompt, const TextGenerationConfig& request);
struct ImagePrompt {
    std::string text;
    std::vector<internal::ImageView> images;
};
Span<const internal::ConfigField> image_request_fields(bool images = true);
TextGenerationConfig image_request_config(internal::ConfigView supplied, bool images = true);
ImagePrompt format_images_prompt(const internal::ImagesTextToTextRequest& input,
                                 const TextGenerationConfig& request);
} // namespace trtmc::diffusion_gemma
