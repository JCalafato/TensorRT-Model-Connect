/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "request.h"

#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace trtmc;
using namespace trtmc::internal;
using namespace trtmc::diffusion_gemma;

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Function>
void invalid(Function fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid request was accepted");
}
} // namespace

int main() {
    TextGenerationConfig config;
    config.use_chat_template = true;
    config.enable_thinking = false;
    float pixel1 = 0, pixel2 = 1;
    ImageView first, second;
    first.data = &pixel1;
    second.data = &pixel2;
    std::vector<ImagesTextPart> parts{TextPartView{"  Compare "}, first, TextPartView{" and "},
                                      second, TextPartView{" please.  "}};
    ImagesTextMessage message{MessageRole::User, {parts.data(), parts.size()}};
    ImagesTextToTextRequest request{{&message, 1}};
    auto formatted = format_images_prompt(request, config);
    require(formatted.text ==
                "<bos><|turn>user\nCompare<|image|>and<|image|>please.<turn|>\n<|turn>model\n",
            "ordered text/image template differs");
    require(formatted.images.size() == 2 && formatted.images[0].data == first.data &&
                formatted.images[1].data == second.data,
            "image order was lost");
    nlohmann::json outputs;
    outputs["two_images"] = formatted.text;
    config.enable_thinking = true;
    outputs["thinking"] = format_images_prompt(request, config).text;
    require(outputs["thinking"] == "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nCompare<|"
                                   "image|>and<|image|>please.<turn|>\n<|turn>model\n",
            "thinking instruction differs");

    std::vector<ImagesTextPart> system_parts{TextPartView{" Be concise. "},
                                             TextPartView{" Answer carefully. "}};
    std::vector<ImagesTextPart> answer_parts{
        TextPartView{" <|channel>thought\nprivate<channel|> First. "}};
    std::vector<ImagesTextPart> continued_parts{TextPartView{" Second. "}};
    std::vector<ImagesTextMessage> history{
        {MessageRole::Developer, {system_parts.data(), system_parts.size()}},
        message,
        {MessageRole::Assistant, {answer_parts.data(), answer_parts.size()}},
        {MessageRole::Assistant, {continued_parts.data(), continued_parts.size()}},
        message};
    ImagesTextToTextRequest conversation{{history.data(), history.size()}};
    auto multi = format_images_prompt(conversation, config);
    outputs["history"] = multi.text;
    require(multi.images.size() == 4, "historical images were omitted");
    require(multi.text.find("private") == std::string::npos, "hidden reasoning was replayed");
    require(multi.text.find("First.<turn|>\nSecond.<turn|>\n") != std::string::npos,
            "assistant continuation was changed");
    config.system_prompt = "duplicate";
    invalid([&] { format_images_prompt(conversation, config); });
    config.system_prompt.clear();
    config.use_chat_template = false;
    outputs["raw"] = format_images_prompt(request, config).text;
    require(outputs["raw"] == "  Compare <|image|> and <|image|> please.  ",
            "raw image request lost whitespace");
    invalid([&] { format_images_prompt(conversation, config); });
    invalid([&] { format_images_prompt({}, config); });
    auto role = message.role;
    message.role = MessageRole::Tool;
    invalid([&] { format_images_prompt(request, config); });
    message.role = role;
    parts[0] = ReasoningPartView{"not a text part"};
    invalid([&] { format_images_prompt(request, config); });

    auto defaults = image_request_config({});
    require(defaults.use_chat_template && defaults.max_new_tokens == 128 &&
                defaults.num_steps == -1,
            "typed request defaults differ");
    std::vector<ConfigEntry> options{{"max_new_tokens", std::int64_t{6}},
                                     {"num_steps", std::int64_t{2}},
                                     {"seed", std::int64_t{17}}};
    auto configured = image_request_config({options.data(), options.size()});
    require(configured.max_new_tokens == 6 && configured.num_steps == 2 && configured.seed == 17,
            "typed options were not preserved");
    for (ConfigEntry bad :
         std::vector<ConfigEntry>{{"temperature", 0.5},
                                  {"seed", true},
                                  {"seed", std::numeric_limits<std::int64_t>::max()},
                                  {"confidence_threshold", std::numeric_limits<double>::infinity()},
                                  {"confidence_threshold", -0.5}})
        invalid([&] { image_request_config({&bad, 1}); });
    options.push_back(options.front());
    invalid([&] { image_request_config({options.data(), options.size()}); });
    std::cout << outputs.dump(2) << '\n';
}
