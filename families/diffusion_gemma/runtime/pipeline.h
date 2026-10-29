/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "image.h"
#include "native_runner.h"
#include "request.h"
#include "tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"

namespace trtmc::diffusion_gemma {

class Pipeline : public ITextGeneration,
                 public internal::IModel,
                 public internal::ITextContinuation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, NativeConfig config,
             std::unique_ptr<ITokenizer> tokenizer, std::int32_t bos_token_id);
    TextResult generate(const std::string& prompt, const TextGenerationConfig& request) override;
    std::int32_t default_max_new_tokens() const override { return 128; }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::TextContinuationRequest&, internal::ConfigView) override;

  protected:
    NativeRunner runner_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::int32_t bos_token_id_;
};
class VisionPipeline final : public Pipeline,
                             public IVisionLanguageGeneration,
                             public internal::IImagesTextToText {
  public:
    VisionPipeline(std::unique_ptr<ITrtModule> engine, NativeConfig config,
                   std::unique_ptr<ITokenizer> tokenizer, std::int32_t bos_token_id,
                   std::unique_ptr<VisionEncoder> vision,
                   std::string primary_task = "vision_language_generation");
    using Pipeline::generate;
    using Pipeline::run;
    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::ImagesTextToTextRequest&, internal::ConfigView) override;
    TextResult generate(const std::string& prompt, const float* image_pixels,
                        std::int32_t image_height, std::int32_t image_width,
                        const TextGenerationConfig& request) override;

  private:
    TextResult generate_images(const std::string& formatted,
                               const std::vector<internal::ImageView>& images,
                               const TextGenerationConfig& request);
    std::unique_ptr<VisionEncoder> vision_;
    std::string primary_task_;
};
} // namespace trtmc::diffusion_gemma
