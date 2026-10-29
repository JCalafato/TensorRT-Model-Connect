/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "vision_encoder.h"

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace trtmc::alpamayo {

// Owning logits and compact growing KV. Views survive subsequent evaluations
// while this object remains alive and unchanged; they never alias module buffers.
struct TextState {
    std::unordered_map<std::string, std::vector<uint8_t>> storage;
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
    DType cache_dtype{DType::kFloat32};

    TensorMap tensors();
    TensorMap past();
    TensorMap action_prefix();
};

// Alpamayo's driving prompt contains ordered camera images, not video-token
// messages. Grid order is row-major across the batch, as in the pinned provider.
struct ImagePositionConfig {
    int32_t image_token;
    int32_t video_token;
    int32_t vision_start_token;
    int32_t vision_end_token;
    int64_t spatial_merge_size;
    int64_t head_width;
    std::array<int64_t, 3> mrope_section;
    float rope_theta;
};

struct TextPositions {
    std::vector<int64_t> axes; // [3, batch, length], T/H/W order.
    std::vector<int64_t> deltas;
    std::vector<uint8_t> cosine;
    std::vector<uint8_t> sine;
    int64_t batch{0}, length{0}, width{0};
    DType dtype{DType::kFloat32};

    TensorMap tensors(); // Owning rotary views [batch, 1, length, width].
};

TextPositions prepare_image_positions(const std::vector<std::vector<int32_t>>& tokens,
                                      const std::vector<std::array<int64_t, 3>>& image_grids,
                                      const std::vector<std::vector<int32_t>>& attention_mask,
                                      const ImagePositionConfig& config, DType dtype);

// Owns the complete native prefill. Visual values are copied without changing
// precision; nonvisual DeepStack entries remain zero and initial KV is empty.
struct TextPrefill {
    TextPositions positions;
    std::vector<int32_t> tokens, visual_mask;
    std::vector<uint8_t> visual;
    std::vector<std::vector<uint8_t>> deepstack;
    std::vector<float> mask;
    int64_t hidden{0}, layers{0}, heads{0};

    TensorMap tensors();
};

struct SamplingOptions {
    float temperature{0.6F}; // Zero selects greedy decoding.
    float top_p{0.98F};
    int64_t top_k{0}; // Zero disables top-k.
    int32_t suppressed_begin{0};
    int32_t suppressed_end{0};
};

// Provider order: suppress discrete trajectory IDs, temperature, top-k, top-p.
// Uniform draws are supplied independently so distribution and RNG contracts can
// be validated separately. Seed repeatability is native, not HF RNG identity.
std::vector<double> sampling_probabilities(const std::vector<float>& logits,
                                           const SamplingOptions& options);
int32_t draw_token(const std::vector<double>& probabilities, double uniform);

struct ReasoningOptions {
    SamplingOptions sampling;
    int64_t max_new_tokens{128};
    int32_t future_start_token{0};
    int32_t pad_token{0};
    std::vector<int32_t> eos_tokens;
    uint64_t seed{0};
};

// Tracks provider stopping separately from the returned padded sequence. After
// every row has emitted future_start, one additional sample is required: the
// preceding forward then places future_start itself into the conditioning KV.
class ReasoningHistory {
  public:
    ReasoningHistory(std::vector<std::vector<int32_t>> prompt, const ReasoningOptions& options);
    bool append(const std::vector<int32_t>& selected);
    bool finished(std::size_t row) const;
    std::vector<std::vector<int32_t>> padded() const;
    std::vector<int64_t> action_offsets() const;
    int64_t generated() const { return generated_; }

  private:
    std::vector<std::vector<int32_t>> tokens_;
    ReasoningOptions options_;
    std::vector<bool> future_found_;
    std::vector<bool> finished_;
    int64_t generated_{0};
    bool stopped_{false};
};

struct ReasoningResult {
    TextState cache;
    std::vector<std::vector<int32_t>> tokens;
    std::vector<int64_t> action_offsets;
    std::vector<int64_t> rope_deltas;
    int64_t generated{0};
};

// Actual native VLM execution through the public MC backend. prepare owns
// media placement and causal masking. evaluate accepts prepared rotary tables;
// generate additionally owns sampling and incremental tables with real KV.
class TextDecoder {
  public:
    explicit TextDecoder(std::unique_ptr<ITrtModule> module);
    // Image features and grids follow prompt order across rows. Padding is
    // explicit, never inferred from token identity. Preparation is host-only
    // and rejected requests leave the reusable engine context untouched.
    TextPrefill prepare(const std::vector<std::vector<int32_t>>& tokens,
                        const std::vector<std::array<int64_t, 3>>& image_grids,
                        const std::vector<VisionFeatures>& images,
                        const std::vector<std::vector<int32_t>>& attention_mask,
                        const ImagePositionConfig& config) const;

    TextState evaluate(const TensorMap& inputs);

    // Prepared multimodal prefill -> real autoregressive reasoning and immutable
    // action-prefix cache. Incremental tokens use scalar positions plus each
    // row's multimodal rotary delta; prefill retains its supplied 3-axis tables.
    ReasoningResult generate(const TensorMap& prepared, const std::vector<int64_t>& rope_deltas,
                             float rope_theta, const ReasoningOptions& options);

  private:
    void validate(const std::string& name, const Tensor& tensor,
                  const std::vector<int64_t>& expected, DType dtype) const;

    std::unique_ptr<ITrtModule> module_;
    int64_t layers_{0};
    int64_t heads_{0};
    int64_t width_{0};
    int64_t hidden_{0};
    int64_t vocab_{0};
    int64_t deepstack_{0};
    DType dtype_{DType::kFloat32};
};

} // namespace trtmc::alpamayo
