/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "response.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace trtmc::diffusion_gemma;
namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
} // namespace
int main() {
    // A real owning BPE tokenizer, not a mock answer renderer.
    auto json = nlohmann::json::parse(R"({
      "model":{"type":"BPE","vocab":{"<unk>":0,"<|channel>":1,"<channel|>":2,
        "thought":3,"Ċ":4,"42":5,"<turn|>":6,"prefix":7},"merges":[]},
      "added_tokens":[{"id":1,"content":"<|channel>","special":true},
        {"id":2,"content":"<channel|>","special":true},
        {"id":6,"content":"<turn|>","special":true}],
      "decoder":{"type":"ByteLevel"}
    })");
    auto serialized = json.dump();
    auto tokenizer = CreateBpeTokenizer(serialized.data(), serialized.size());
    const std::vector<std::int32_t> generated{1, 3, 4, 2, 5, 6};
    require(tokenizer->decode(generated) == "thought\n42",
            "fixture must reproduce channel-label leak");
    require(decode_response(*tokenizer, generated) == "42", "empty thought channel leaked");
    require(decode_response(*tokenizer, {1, 3, 4, 7, 2, 5, 6}) == "42",
            "reasoning leaked into final answer");
    require(decode_response(*tokenizer, {7, 1, 3, 4, 2, 5}) == "prefix42",
            "visible prefix was removed");
    require(decode_response(*tokenizer, {3, 4, 5}) == "thought\n42",
            "ordinary thought text was removed");
    require(decode_response(*tokenizer, {1, 3, 4, 7}).empty(),
            "unfinished hidden channel was exposed");
    require(decode_response(*tokenizer, {5, 6}) == "42", "channel-free output changed");
    require(decode_response(*tokenizer, {}).empty(), "empty answer changed");
    json["added_tokens"] = nlohmann::json::array();
    json["model"]["vocab"].erase("<|channel>");
    json["model"]["vocab"].erase("<channel|>");
    serialized = json.dump();
    tokenizer = CreateBpeTokenizer(serialized.data(), serialized.size());
    require(decode_response(*tokenizer, {5}) == "42", "tokenizer without channel tokens changed");
}
