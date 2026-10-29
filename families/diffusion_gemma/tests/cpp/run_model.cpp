/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Owning E2E through the public model loader and Tasks, not a private family factory.
#include "trtmc/language.hpp"
#include "trtmc/text.hpp"

#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 5)
        return 2;
    try {
        const auto model = trtmc::Model::load(argv[1], {argv[2]});
        nlohmann::json requests, output = nlohmann::json::array();
        std::ifstream(argv[3]) >> requests;
        for (const auto& request : requests) {
            const auto prompt = request.at("prompt").get<std::string>();
            trtmc::Config config{{"max_new_tokens", request.at("max_new_tokens").get<int>()},
                                 {"num_steps", request.at("num_steps").get<int>()},
                                 {"seed", request.at("seed").get<int>()},
                                 {"use_chat_template", request.value("use_chat_template", true)},
                                 {"enable_thinking", request.value("enable_thinking", false)}};
            std::deque<std::vector<std::uint8_t>> pixels;
            std::vector<trtmc::ImagesTextPart> parts;
            for (const auto& image : request.at("images")) {
                std::ifstream file(image.at("path").get<std::string>(), std::ios::binary);
                if (!file)
                    throw std::runtime_error("Cannot read RGB fixture");
                pixels.emplace_back(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>());
                parts.emplace_back(trtmc::ImageInput{{pixels.back().data(), pixels.back().size()},
                                                     image.at("height"),
                                                     image.at("width")});
            }
            auto result = [&]() {
                if (parts.empty())
                    return model.task<trtmc::TextContinuation>().run({prompt}, config);
                parts.emplace_back(trtmc::TextPart{prompt});
                return model.task<trtmc::ImagesTextToText>().run(
                    trtmc::ImagesTextToTextRequest::from_parts(std::move(parts)), config);
            }();
            auto ids = result.token_ids();
            output.push_back({{"id", request.at("id")},
                              {"text", std::string(result.text())},
                              {"token_ids", std::vector<std::int32_t>(ids.begin(), ids.end())}});
            std::cout << output.back().dump() << std::endl;
            std::ofstream(argv[4]) << output.dump(2) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
