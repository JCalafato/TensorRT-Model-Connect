/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc::alpamayo {

// Owning outputs preserve the engine's FP16/FP32 values exactly. Tensor views
// stay valid while this object is alive and unchanged, including after the
// encoder processes a later frame.
struct VisionFeatures {
    int64_t tokens{0};
    int64_t width{0};
    DType dtype{DType::kFloat32};
    std::vector<std::uint8_t> features;
    std::vector<std::vector<std::uint8_t>> deepstack;

    TensorMap tensors();
};

// Executes prepared visual patches through the MC C++ TensorRT backend. Image
// decoding, patch ordering and rotary/interpolation preparation belong to the
// family's preprocessing stage, not to a shared runtime or Python callback.
class VisionEncoder {
  public:
    VisionEncoder(std::unique_ptr<ITrtModule> module, int64_t spatial_merge_size,
                  int64_t position_embeddings, std::size_t deepstack_count);

    // Validates the entire input set before execution, so rejected inputs do
    // not change the reusable engine context. Output storage is caller-owned.
    VisionFeatures encode(const TensorMap& inputs);

  private:
    void validate(const std::string& name, const Tensor& tensor,
                  const std::vector<int64_t>& shape) const;

    std::unique_ptr<ITrtModule> module_;
    int64_t merge_unit_{0};
    int64_t positions_{0};
    int64_t flat_patch_{0};
    int64_t head_width_{0};
    int64_t feature_width_{0};
    std::size_t deepstack_count_{0};
    DType output_dtype_{DType::kFloat32};
};

} // namespace trtmc::alpamayo
