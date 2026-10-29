/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tokenizer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::alpamayo {

struct HistoryTokenConfig {
    std::size_t points;
    int32_t bins;
    int32_t token_offset;
    int32_t placeholder_token;
    std::array<float, 3> minimum;
    std::array<float, 3> maximum;
};

// The trained history tokenizer encodes XYZ deltas, not the action expert's
// acceleration/curvature vocabulary. The first delta is relative to zero; all
// subsequent deltas are relative to the preceding supplied history waypoint.
class HistoryTokenizer {
  public:
    explicit HistoryTokenizer(HistoryTokenConfig config);
    std::vector<int32_t> encode(const std::vector<float>& history_xyz) const;
    std::vector<int32_t> fuse(const std::vector<int32_t>& prompt,
                              const std::vector<float>& history_xyz) const;

  private:
    HistoryTokenConfig config_;
};

struct DrivingBatch {
    std::vector<std::vector<int32_t>> tokens;
    std::vector<std::vector<int32_t>> attention_mask;
    std::vector<std::array<int64_t, 3>> image_grids;
};

struct DrivingText {
    std::string cot, meta_action, answer;
};

// The trained driving protocol uses ordered camera images and one history
// trajectory. Prompt framing belongs here, not in a shared chat-template API.
class DrivingPrompt {
  public:
    DrivingPrompt(std::unique_ptr<ITokenizer> tokenizer, HistoryTokenConfig history,
                  int64_t spatial_merge_size, std::size_t token_capacity);
    std::vector<int32_t> encode(const std::vector<std::array<int64_t, 3>>& image_grids,
                                const std::vector<float>& history_xyz) const;
    DrivingBatch encode_batch(const std::vector<std::vector<std::array<int64_t, 3>>>& image_grids,
                              const std::vector<std::vector<float>>& history_xyz,
                              int32_t pad_token) const;
    std::string decode(const std::vector<int32_t>& tokens, bool skip_special_tokens = false) const;
    DrivingText extract_text(const std::vector<int32_t>& generated_tokens) const;

  private:
    std::unique_ptr<ITokenizer> tokenizer_;
    HistoryTokenizer history_;
    int64_t merge_size_;
    std::size_t capacity_;
    int32_t image_, vision_start_, vision_end_;
    std::vector<int32_t> prefix_, suffix_;
};

} // namespace trtmc::alpamayo
