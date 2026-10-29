/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sampler.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::diffusion_gemma {

struct NativeConfig {
    SamplingSettings sampling;
    std::int32_t hidden_size{0};
    std::int32_t max_sequence_length{0};
    std::vector<std::int32_t> cache_widths;
    std::vector<std::int32_t> eos_token_ids;
};

struct ConditionedPrompt {
    std::vector<std::int32_t> blocks;
    std::vector<float> features;
};

// The shared-weight engine runs in two phases. Only encoder calls may commit
// K/V. Denoising reads that immutable prefix and carries soft embeddings on GPU.
class NativeRunner {
  public:
    NativeRunner(std::unique_ptr<ITrtModule> engine, NativeConfig config);
    TextResult generate_ids(const std::vector<std::int32_t>& prompt,
                            const TextGenerationConfig& request,
                            const ConditionedPrompt& media = {});

    void validate_request(const std::vector<std::int32_t>& prompt,
                          const TextGenerationConfig& request) const;

  private:
    SamplingSettings settings_for(const TextGenerationConfig& request) const;
    void reset();
    void clear_conditioning(std::size_t length);
    void execute(const std::vector<std::int32_t>& tokens, bool decoder, float temperature,
                 const ConditionedPrompt& media = {});
    void encode(const std::vector<std::int32_t>& tokens, const ConditionedPrompt& media = {});
    void copy_conditioning(std::size_t length);
    void* pointer(const std::string& name) const;
    // Outlives the execution context that holds its external binding.
    DeviceTensor conditioning_;
    std::unique_ptr<ITrtModule> engine_;
    NativeConfig config_;
    DType dtype_;
    std::int32_t prefix_length_{0};
    std::vector<float> logits_;
};

} // namespace trtmc::diffusion_gemma
