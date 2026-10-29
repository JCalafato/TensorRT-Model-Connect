/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "history_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::alpamayo {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("Alpamayo history tokens: " + message);
}
// Match Python str.strip rather than the active C locale. Tokenizer decoding
// already returns UTF-8; invalid sequences are treated as non-whitespace.
std::string trim_text(const std::string& text) {
    std::size_t begin = text.size(), end = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto start = offset;
        const auto lead = static_cast<uint8_t>(text[offset++]);
        uint32_t code = lead;
        const auto continuation = lead < 0x80             ? 0
                                  : (lead & 0xe0) == 0xc0 ? 1
                                  : (lead & 0xf0) == 0xe0 ? 2
                                  : (lead & 0xf8) == 0xf0 ? 3
                                                          : 0;
        if (continuation > 0 && offset + continuation <= text.size()) {
            code = lead & ((1U << (6 - continuation)) - 1);
            for (int i = 0; i < continuation; ++i) {
                const auto next = static_cast<uint8_t>(text[offset]);
                if ((next & 0xc0) != 0x80) {
                    code = 0xfffd;
                    break;
                }
                code = (code << 6) | (next & 0x3f);
                ++offset;
            }
        }
        const bool space = (code >= 0x09 && code <= 0x0d) || (code >= 0x1c && code <= 0x20) ||
                           code == 0x85 || code == 0xa0 || code == 0x1680 ||
                           (code >= 0x2000 && code <= 0x200a) || code == 0x2028 || code == 0x2029 ||
                           code == 0x202f || code == 0x205f || code == 0x3000;
        if (!space) {
            begin = std::min(begin, start);
            end = offset;
        }
    }
    return begin < end ? text.substr(begin, end - begin) : std::string{};
}
} // namespace

HistoryTokenizer::HistoryTokenizer(HistoryTokenConfig config) : config_(config) {
    require(config.points > 0 && config.points <= std::numeric_limits<std::size_t>::max() / 3,
            "invalid history length");
    // FP32 quantization must represent every bin index exactly.
    require(config.bins > 1 && config.bins <= (1 << 24) && config.token_offset >= 0 &&
                config.token_offset <= std::numeric_limits<int32_t>::max() - (config.bins - 1),
            "invalid history vocabulary");
    require(config.placeholder_token >= 0 &&
                (config.placeholder_token < config.token_offset ||
                 config.placeholder_token > config.token_offset + config.bins - 1),
            "history placeholder overlaps the discrete vocabulary");
    for (std::size_t axis = 0; axis < 3; ++axis)
        require(std::isfinite(config.minimum[axis]) && std::isfinite(config.maximum[axis]) &&
                    config.minimum[axis] < config.maximum[axis] &&
                    std::isfinite(config.maximum[axis] - config.minimum[axis]),
                "invalid coordinate quantization range");
}

std::vector<int32_t> HistoryTokenizer::encode(const std::vector<float>& history_xyz) const {
    require(history_xyz.size() == config_.points * 3, "wrong history XYZ shape");
    for (const auto value : history_xyz)
        require(std::isfinite(value), "non-finite history coordinate");
    std::vector<int32_t> result(history_xyz.size());
    for (std::size_t i = 0; i < history_xyz.size(); ++i) {
        const auto axis = i % 3;
        const float previous = i < 3 ? 0.0F : history_xyz[i - 3];
        const float delta = history_xyz[i] - previous;
        const float shifted = delta - config_.minimum[axis];
        const float unit = shifted / (config_.maximum[axis] - config_.minimum[axis]);
        const float scaled = unit * static_cast<float>(config_.bins - 1);
        require(std::isfinite(scaled), "coordinate delta overflow");
        // Saturating before rounding is equivalent to the provider's final
        // clamp, and avoids converting unbounded values to a narrow integer.
        const float bounded = std::clamp(scaled, 0.0F, static_cast<float>(config_.bins - 1));
        const auto lower = static_cast<int32_t>(std::floor(bounded));
        const float fraction = bounded - static_cast<float>(lower);
        const auto rounded = lower + (fraction > 0.5F || (fraction == 0.5F && lower % 2 != 0));
        result[i] = config_.token_offset + rounded;
    }
    return result;
}

std::vector<int32_t> HistoryTokenizer::fuse(const std::vector<int32_t>& prompt,
                                            const std::vector<float>& history_xyz) const {
    require(static_cast<std::size_t>(std::count(prompt.begin(), prompt.end(),
                                                config_.placeholder_token)) == config_.points * 3,
            "prompt must contain exactly one placeholder per history coordinate");
    const auto tokens = encode(history_xyz);
    auto result = prompt;
    std::size_t position = 0;
    for (auto& token : result)
        if (token == config_.placeholder_token)
            token = tokens[position++];
    return result;
}

DrivingPrompt::DrivingPrompt(std::unique_ptr<ITokenizer> tokenizer, HistoryTokenConfig history,
                             int64_t spatial_merge_size, std::size_t token_capacity)
    : tokenizer_(std::move(tokenizer)), history_(history), merge_size_(spatial_merge_size),
      capacity_(token_capacity) {
    require(tokenizer_ && merge_size_ > 0 && capacity_ > 0, "invalid driving prompt configuration");
    require(history.points <= capacity_ / 3, "history exceeds prompt capacity");
    const auto token = [&](const char* text) {
        const auto id = tokenizer_->id_for_token(text);
        require(id >= 0 && tokenizer_->encode(text) == std::vector<int32_t>{id},
                std::string("missing atomic driving token ") + text);
        return id;
    };
    image_ = token("<|image_pad|>");
    vision_start_ = token("<|vision_start|>");
    vision_end_ = token("<|vision_end|>");
    require(token("<|traj_history|>") == history.placeholder_token,
            "history placeholder disagrees with tokenizer");
    for (const auto* name : {"<|im_start|>", "<|im_end|>", "<|traj_history_start|>",
                             "<|traj_history_end|>", "<|cot_start|>"})
        token(name);
    for (int32_t i = 0; i < history.bins; ++i)
        require(!tokenizer_->token_for_id(history.token_offset + i).empty(),
                "history vocabulary is missing from tokenizer");
    prefix_ = tokenizer_->encode(
        "<|im_start|>system\nYou are a driving assistant that generates safe and accurate "
        "actions.<|im_end|>\n<|im_start|>user\n");
    std::string suffix = "<|traj_history_start|>";
    for (std::size_t i = 0; i < history.points * 3; ++i)
        suffix += "<|traj_history|>";
    suffix += "<|traj_history_end|>output the chain-of-thought reasoning of the driving process, "
              "then output the future trajectory.<|im_end|>\n<|im_start|>assistant\n<|cot_start|>";
    suffix_ = tokenizer_->encode(suffix);
    require(prefix_.size() <= capacity_ && suffix_.size() <= capacity_ - prefix_.size(),
            "driving framing exceeds prompt capacity");
}

std::vector<int32_t> DrivingPrompt::encode(const std::vector<std::array<int64_t, 3>>& image_grids,
                                           const std::vector<float>& history_xyz) const {
    require(!image_grids.empty(), "driving prompt requires camera images");
    std::size_t total = prefix_.size() + suffix_.size();
    std::vector<std::size_t> counts;
    for (const auto& grid : image_grids) {
        require(grid[0] > 0 && grid[1] > 0 && grid[2] > 0 && grid[1] % merge_size_ == 0 &&
                    grid[2] % merge_size_ == 0,
                "invalid camera grid");
        require(capacity_ - total >= 2, "camera markers exceed prompt capacity");
        const auto available = capacity_ - total - 2;
        const auto h = grid[1] / merge_size_, w = grid[2] / merge_size_;
        require(static_cast<uint64_t>(grid[0]) <=
                    available / static_cast<uint64_t>(h) / static_cast<uint64_t>(w),
                "camera tokens exceed prompt capacity");
        const auto count = static_cast<std::size_t>(grid[0]) * static_cast<std::size_t>(h) *
                           static_cast<std::size_t>(w);
        counts.push_back(count);
        total += count + 2;
    }
    auto result = prefix_;
    result.reserve(total);
    for (const auto count : counts) {
        result.push_back(vision_start_);
        result.insert(result.end(), count, image_);
        result.push_back(vision_end_);
    }
    result.insert(result.end(), suffix_.begin(), suffix_.end());
    return history_.fuse(result, history_xyz);
}

DrivingBatch
DrivingPrompt::encode_batch(const std::vector<std::vector<std::array<int64_t, 3>>>& image_grids,
                            const std::vector<std::vector<float>>& history_xyz,
                            int32_t pad_token) const {
    require(!image_grids.empty() && image_grids.size() == history_xyz.size(),
            "camera/history batch mismatch");
    require(!tokenizer_->token_for_id(pad_token).empty(), "invalid driving padding token");
    DrivingBatch result;
    std::size_t length = 0;
    for (std::size_t row = 0; row < image_grids.size(); ++row) {
        result.tokens.push_back(encode(image_grids[row], history_xyz[row]));
        length = std::max(length, result.tokens.back().size());
        result.image_grids.insert(result.image_grids.end(), image_grids[row].begin(),
                                  image_grids[row].end());
    }
    require(result.tokens.size() <=
                std::numeric_limits<std::size_t>::max() / sizeof(int32_t) / length,
            "driving batch size overflow");
    for (auto& tokens : result.tokens) {
        const auto padding = length - tokens.size();
        std::vector<int32_t> mask(length, 1);
        std::fill_n(mask.begin(), padding, 0);
        tokens.insert(tokens.begin(), padding, pad_token);
        result.attention_mask.push_back(std::move(mask));
    }
    return result;
}

DrivingText DrivingPrompt::extract_text(const std::vector<int32_t>& generated_tokens) const {
    const auto text = decode(generated_tokens, false);
    const auto extract = [&](const char* field) {
        const auto start_token = std::string("<|") + field + "_start|>";
        const auto end_token = std::string("<|") + field + "_end|>";
        const auto end = text.find(end_token);
        if (end == std::string::npos)
            return std::string{};
        const auto before = text.substr(0, end);
        const auto start = before.rfind(start_token);
        return trim_text(start == std::string::npos ? before
                                                    : before.substr(start + start_token.size()));
    };
    return {extract("cot"), extract("meta_action"), extract("answer")};
}

std::string DrivingPrompt::decode(const std::vector<int32_t>& tokens,
                                  bool skip_special_tokens) const {
    for (const auto id : tokens)
        require(!tokenizer_->token_for_id(id).empty(), "output token outside tokenizer vocabulary");
    return tokenizer_->decode(tokens, skip_special_tokens);
}

} // namespace trtmc::alpamayo
