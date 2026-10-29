/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "request.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace trtmc::diffusion_gemma {
std::string trim_chat_text(const std::string& text) {
    static constexpr std::string_view whitespace[]{
        "\x09",         "\x0a",         "\x0b",         "\x0c",         "\x0d",
        "\x1c",         "\x1d",         "\x1e",         "\x1f",         "\x20",
        "\xc2\x85",     "\xc2\xa0",     "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
        "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86",
        "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
        "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"};
    std::size_t first = 0, last = text.size();
    bool trimmed = true;
    while (trimmed && first < last) {
        trimmed = false;
        for (const auto space : whitespace) {
            if (space.size() <= last - first && text.compare(first, space.size(), space) == 0) {
                first += space.size();
                trimmed = true;
                break;
            }
        }
    }
    trimmed = true;
    while (trimmed && first < last) {
        trimmed = false;
        for (const auto space : whitespace) {
            if (space.size() <= last - first &&
                text.compare(last - space.size(), space.size(), space) == 0) {
                last -= space.size();
                trimmed = true;
                break;
            }
        }
    }
    return text.substr(first, last - first);
}

std::string format_prompt(const std::string& prompt, const TextGenerationConfig& request) {
    if (!request.use_chat_template) {
        if (!request.system_prompt.empty())
            throw std::invalid_argument("DiffusionGemma system prompt requires chat formatting");
        return prompt;
    }
    // Checkpoint single-user template; the public Task supplies no tool history.
    std::string result = "<bos>";
    if (request.enable_thinking || !request.system_prompt.empty()) {
        result += "<|turn>system\n";
        if (request.enable_thinking)
            result += "<|think|>\n";
        result += trim_chat_text(request.system_prompt) + "<turn|>\n";
    }
    return result + "<|turn>user\n" + trim_chat_text(prompt) + "<turn|>\n<|turn>model\n";
}

Span<const internal::ConfigField> image_request_fields(bool images) {
    using internal::ConfigKind;
    static const std::vector<internal::ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, std::int64_t{128}, "Generated token budget"},
        {"num_steps", ConfigKind::I64, std::int64_t{-1}, "Denoising steps; -1 uses checkpoint"},
        {"seed", ConfigKind::I64, std::int64_t{-1}, "Random seed; -1 leaves unspecified"},
        {"block_length", ConfigKind::I64, std::int64_t{0}, "Checkpoint canvas; 0 uses checkpoint"},
        {"confidence_threshold", ConfigKind::F64, -1.0, "Stopping confidence; -1 uses checkpoint"},
        {"use_chat_template", ConfigKind::Bool, true, "Apply the checkpoint provider template"},
        {"enable_thinking", ConfigKind::Bool, true, "Enable the provider thinking instruction"},
        {"system_prompt", ConfigKind::String, std::string_view{}, "Optional system instruction"},
    };
    static const auto text_fields = [] {
        auto copy = fields;
        copy[5].default_value = false;
        return copy;
    }();
    const auto& selected = images ? fields : text_fields;
    return {selected.data(), selected.size()};
}

TextGenerationConfig image_request_config(internal::ConfigView supplied, bool images) {
    const auto fields = image_request_fields(images);
    internal::validate_config(fields, supplied);
    const auto integer = [&](const char* name) {
        const auto value = *internal::config_get<std::int64_t>(supplied, fields, name);
        if (value < std::numeric_limits<std::int32_t>::min() ||
            value > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument(std::string("DiffusionGemma integer outside int32: ") +
                                        name);
        return static_cast<std::int32_t>(value);
    };
    TextGenerationConfig config;
    config.max_new_tokens = integer("max_new_tokens");
    config.num_steps = integer("num_steps");
    config.seed = integer("seed");
    config.block_length = integer("block_length");
    const auto confidence = *internal::config_get<double>(supplied, fields, "confidence_threshold");
    if (!std::isfinite(confidence) || confidence > std::numeric_limits<float>::max() ||
        (confidence < 0 && confidence != -1))
        throw std::invalid_argument("DiffusionGemma invalid confidence threshold");
    config.confidence_threshold = static_cast<float>(confidence);
    config.use_chat_template = *internal::config_get<bool>(supplied, fields, "use_chat_template");
    config.enable_thinking = *internal::config_get<bool>(supplied, fields, "enable_thinking");
    config.system_prompt =
        *internal::config_get<std::string_view>(supplied, fields, "system_prompt");
    return config;
}

namespace {
std::string strip_thinking(std::string_view text) {
    // Provider template removes each <|channel>...<channel|> segment from
    // assistant history; do not replay hidden reasoning as answer content.
    std::string result;
    while (true) {
        const auto end = text.find("<channel|>");
        auto part = text.substr(0, end);
        const auto start = part.find("<|channel>");
        result.append(part.substr(0, start));
        if (end == std::string_view::npos)
            break;
        text.remove_prefix(end + std::string_view("<channel|>").size());
    }
    return trim_chat_text(result);
}
} // namespace

ImagePrompt format_images_prompt(const internal::ImagesTextToTextRequest& input,
                                 const TextGenerationConfig& request) {
    using internal::MessageRole;
    if (input.messages.empty() || !input.tools.empty())
        throw std::invalid_argument("DiffusionGemma requires messages without tool definitions");
    if (!request.use_chat_template &&
        (input.messages.size() != 1 || input.messages[0].role != MessageRole::User ||
         !request.system_prompt.empty()))
        throw std::invalid_argument("DiffusionGemma raw image prompt requires one user message");
    ImagePrompt result;
    auto append_parts = [&](const internal::ImagesTextMessage& message, bool system) {
        std::string content;
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<internal::TextPartView>(&part)) {
                const std::string value(text->text);
                content += !request.use_chat_template               ? value
                           : message.role == MessageRole::Assistant ? strip_thinking(value)
                                                                    : trim_chat_text(value);
                if (system)
                    content += " "; // Provider system content-parts include a trailing space.
            } else if (const auto* image = std::get_if<internal::ImageView>(&part)) {
                if (system)
                    throw std::invalid_argument(
                        "DiffusionGemma system messages cannot contain images");
                content += "<|image|>";
                result.images.push_back(*image);
            } else {
                throw std::invalid_argument(
                    "DiffusionGemma image task does not map tool or reasoning parts");
            }
        }
        return content;
    };
    if (!request.use_chat_template) {
        result.text = append_parts(input.messages[0], false);
        return result;
    }
    result.text = "<bos>";
    const bool has_system = input.messages[0].role == MessageRole::System ||
                            input.messages[0].role == MessageRole::Developer;
    if (has_system && !request.system_prompt.empty())
        throw std::invalid_argument("DiffusionGemma system instruction supplied twice");
    if (has_system || request.enable_thinking || !request.system_prompt.empty()) {
        result.text += "<|turn>system\n";
        if (request.enable_thinking)
            result.text += "<|think|>\n";
        result.text += has_system ? append_parts(input.messages[0], true)
                                  : trim_chat_text(request.system_prompt);
        result.text += "<turn|>\n";
    }
    bool previous_assistant = false;
    for (std::size_t index = has_system ? 1 : 0; index < input.messages.size(); ++index) {
        const auto& message = input.messages[index];
        const bool assistant = message.role == MessageRole::Assistant;
        if (!assistant && message.role != MessageRole::User)
            throw std::invalid_argument(
                "DiffusionGemma expects user/assistant turns after the system");
        if (!assistant || !previous_assistant)
            result.text += assistant ? "<|turn>model\n" : "<|turn>user\n";
        result.text += append_parts(message, false) + "<turn|>\n";
        previous_assistant = assistant;
    }
    result.text += "<|turn>model\n";
    return result;
}
} // namespace trtmc::diffusion_gemma
