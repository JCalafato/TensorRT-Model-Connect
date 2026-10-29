/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "response.h"

namespace trtmc::diffusion_gemma {
std::string decode_response(const ITokenizer& tokenizer, const std::vector<std::int32_t>& ids) {
    const auto begin = tokenizer.id_for_token("<|channel>");
    const auto end = tokenizer.id_for_token("<channel|>");
    if (begin < 0 || end < 0)
        return tokenizer.decode(ids);
    // Match the provider's assistant-content strip_thinking operation BEFORE
    // decoding removes special tokens. Otherwise even an empty thought channel
    // leaks its ordinary-text label ("thought\n") into the final answer.
    // The caller retains the original token_ids for auditing or other rendering.
    std::vector<std::int32_t> answer;
    answer.reserve(ids.size());
    bool in_channel = false;
    for (auto token : ids) {
        if (token == begin)
            in_channel = true;
        else if (token == end)
            in_channel = false;
        else if (!in_channel)
            answer.push_back(token);
    }
    return tokenizer.decode(answer);
}
} // namespace trtmc::diffusion_gemma
