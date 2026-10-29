/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"

#include "response.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace trtmc::diffusion_gemma {
Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, NativeConfig config,
                   std::unique_ptr<ITokenizer> tokenizer, std::int32_t bos_token_id)
    : runner_(std::move(engine), std::move(config)), tokenizer_(std::move(tokenizer)),
      bos_token_id_(bos_token_id) {
    if (!tokenizer_)
        throw std::invalid_argument("DiffusionGemma requires a tokenizer");
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ITextContinuation>(*this, image_request_fields(false))};
}

TextResult Pipeline::run(const internal::TextContinuationRequest& input,
                         internal::ConfigView supplied) {
    const auto* text = std::get_if<std::string_view>(&input.prefix);
    if (!text)
        throw std::invalid_argument("DiffusionGemma continuation requires a UTF-8 text prefix");
    return generate(std::string(*text), image_request_config(supplied, false));
}

TextResult Pipeline::generate(const std::string& prompt, const TextGenerationConfig& request) {
    auto tokens = tokenizer_->encode(format_prompt(prompt, request));
    if (request.use_chat_template && tokens.size() >= 2 && tokens[0] == bos_token_id_ &&
        tokens[1] == bos_token_id_)
        tokens.erase(tokens.begin());
    auto result = runner_.generate_ids(tokens, request);
    result.text = decode_response(*tokenizer_, result.token_ids);
    return result;
}
VisionPipeline::VisionPipeline(std::unique_ptr<ITrtModule> engine, NativeConfig config,
                               std::unique_ptr<ITokenizer> tokenizer, std::int32_t bos_token_id,
                               std::unique_ptr<VisionEncoder> vision, std::string primary_task)
    : Pipeline(std::move(engine), std::move(config), std::move(tokenizer), bos_token_id),
      vision_(std::move(vision)), primary_task_(std::move(primary_task)) {
    if (!vision_)
        throw std::invalid_argument("DiffusionGemma requires a vision engine");
}

TextResult VisionPipeline::generate(const std::string& prompt, const float* image_pixels,
                                    std::int32_t height, std::int32_t width,
                                    const TextGenerationConfig& request) {
    if (!image_pixels || height <= 0 || width <= 0)
        throw std::invalid_argument("DiffusionGemma requires a nonempty RGB image");
    internal::ImageView image;
    image.data = image_pixels;
    image.height = height;
    image.width = width;
    image.channels = 3;
    image.byte_size = static_cast<std::uint64_t>(height) * width * 3 * sizeof(float);
    image.format = internal::ImageFormat::Float32;
    const std::string marker = "<|image|>";
    std::string content = prompt;
    if (content.find(marker) == std::string::npos)
        content = marker + (request.use_chat_template ? trim_chat_text(prompt) : prompt);
    return generate_images(format_prompt(content, request), {image}, request);
}

std::vector<internal::TaskInstance> VisionPipeline::task_bindings() {
    auto bindings = Pipeline::task_bindings();
    bindings.push_back(internal::bind<internal::IImagesTextToText>(*this, image_request_fields()));
    return bindings;
}

TextResult VisionPipeline::run(const internal::ImagesTextToTextRequest& input,
                               internal::ConfigView supplied) {
    const auto request = image_request_config(supplied);
    const auto prompt = format_images_prompt(input, request);
    return generate_images(prompt.text, prompt.images, request);
}

TextResult VisionPipeline::generate_images(const std::string& formatted,
                                           const std::vector<internal::ImageView>& images,
                                           const TextGenerationConfig& request) {
    const auto& config = vision_->config();
    auto unexpanded = tokenizer_->encode(formatted);
    if (request.use_chat_template && unexpanded.size() >= 2 && unexpanded[0] == bos_token_id_ &&
        unexpanded[1] == bos_token_id_)
        unexpanded.erase(unexpanded.begin());
    if (static_cast<std::size_t>(
            std::count(unexpanded.begin(), unexpanded.end(), config.image_token)) != images.size())
        throw std::invalid_argument("DiffusionGemma image markers must match supplied images");

    // Validate every image and the complete expanded request before GPU execution.
    std::vector<ImagePatches> patches;
    patches.reserve(images.size());
    for (const auto& image : images)
        patches.push_back(image_patches(image, config));
    std::vector<std::int32_t> tokens, blocks;
    std::size_t index = 0;
    for (auto token : unexpanded) {
        if (token != config.image_token) {
            tokens.push_back(token);
            blocks.push_back(0);
            continue;
        }
        tokens.push_back(config.begin_token);
        blocks.push_back(0);
        tokens.insert(tokens.end(), patches[index].soft_tokens, config.image_token);
        blocks.insert(blocks.end(), patches[index].soft_tokens,
                      static_cast<std::int32_t>(index + 1));
        tokens.push_back(config.end_token);
        blocks.push_back(0);
        ++index;
    }
    runner_.validate_request(tokens, request);
    ConditionedPrompt media;
    if (!images.empty()) {
        media.blocks = std::move(blocks);
        media.features.resize(tokens.size() * config.hidden_size, 0.0F);
        std::size_t offset = 0;
        for (std::size_t i = 0; i < patches.size(); ++i) {
            while (media.blocks[offset] != static_cast<std::int32_t>(i + 1))
                ++offset;
            const auto features = vision_->encode(patches[i]);
            std::copy(features.begin(), features.end(),
                      media.features.begin() + offset * config.hidden_size);
            offset += patches[i].soft_tokens;
        }
    }
    auto result = runner_.generate_ids(tokens, request, media);
    result.text = decode_response(*tokenizer_, result.token_ids);
    return result;
}
} // namespace trtmc::diffusion_gemma
