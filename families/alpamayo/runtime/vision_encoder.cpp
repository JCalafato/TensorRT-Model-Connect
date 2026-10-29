/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_encoder.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::alpamayo {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("Alpamayo vision: " + message);
}

std::size_t checked_bytes(const std::vector<int64_t>& shape, DType dtype) {
    auto count = dtype_size(dtype);
    require(count != 0 && !shape.empty(), "invalid tensor dtype/rank");
    for (const auto dimension : shape) {
        require(dimension > 0 && static_cast<uint64_t>(dimension) <=
                                     std::numeric_limits<std::size_t>::max() / count,
                "invalid tensor size");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

} // namespace

TensorMap VisionFeatures::tensors() {
    TensorMap outputs{{"features", {features.data(), {tokens, width}, dtype}}};
    for (std::size_t i = 0; i < deepstack.size(); ++i)
        outputs.emplace("deepstack_" + std::to_string(i),
                        Tensor{deepstack[i].data(), {tokens, width}, dtype});
    return outputs;
}

VisionEncoder::VisionEncoder(std::unique_ptr<ITrtModule> module, int64_t spatial_merge_size,
                             int64_t position_embeddings, std::size_t deepstack_count)
    : module_(std::move(module)), positions_(position_embeddings) {
    require(module_ && module_->ok(), "engine is unavailable");
    require(spatial_merge_size > 0 &&
                spatial_merge_size <= std::numeric_limits<int64_t>::max() / spatial_merge_size,
            "invalid spatial merge size");
    merge_unit_ = spatial_merge_size * spatial_merge_size;
    require(positions_ > 0 && positions_ <= std::numeric_limits<int32_t>::max(),
            "invalid position table size");
    require(module_->optimization_profile_count() == 1, "expected one visual profile");
    require(module_->input_info().size() == 6, "expected six prepared visual inputs");
    for (const auto* name :
         {"pixels", "position_indices", "position_weights", "mask", "cosine", "sine"}) {
        const auto dtype =
            std::string(name) == "position_indices" ? DType::kInt32 : DType::kFloat32;
        require(module_->has_input(name) && module_->tensor_dtype(name) == dtype,
                std::string("missing or incorrectly typed input ") + name);
    }
    const auto pixels = module_->tensor_shape("pixels");
    const auto rotary = module_->tensor_shape("cosine");
    require(pixels.size() == 2 && pixels[1] > 0, "invalid flattened patch width");
    require(rotary.size() == 3 && rotary[1] == 1 && rotary[2] > 0, "invalid rotary width");
    flat_patch_ = pixels[1];
    head_width_ = rotary[2];
    for (const auto selector :
         {ProfileShapeSelector::kMin, ProfileShapeSelector::kOpt, ProfileShapeSelector::kMax}) {
        const auto shape = module_->input_profile_shape("pixels", 0, selector);
        require(shape.size() == 2 && shape[0] > 0 && shape[0] % merge_unit_ == 0 &&
                    shape[1] == flat_patch_,
                "invalid patch profile");
        const auto count = shape[0];
        for (const auto& [name, expected] :
             std::vector<std::pair<std::string, std::vector<int64_t>>>{
                 {"position_indices", {4, count}},
                 {"position_weights", {4, count}},
                 {"mask", {1, count, count}},
                 {"cosine", {count, 1, head_width_}},
                 {"sine", {count, 1, head_width_}}})
            require(module_->input_profile_shape(name, 0, selector) == expected,
                    "inconsistent patch profile for " + name);
    }
    require(module_->has_output("features") && deepstack_count > 0 &&
                deepstack_count < module_->output_info().size() &&
                module_->output_info().size() - 1 == deepstack_count,
            "missing final or DeepStack features");
    const auto shape = module_->tensor_shape("features");
    require(shape.size() == 2 && shape[1] > 0, "invalid feature width");
    feature_width_ = shape[1];
    output_dtype_ = module_->tensor_dtype("features");
    require(output_dtype_ == DType::kFloat16 || output_dtype_ == DType::kFloat32,
            "features must use FP16 or FP32 storage");
    deepstack_count_ = deepstack_count;
    for (std::size_t i = 0; i < deepstack_count_; ++i) {
        const auto name = "deepstack_" + std::to_string(i);
        require(module_->has_output(name) && module_->tensor_dtype(name) == output_dtype_ &&
                    module_->tensor_shape(name) == shape,
                "inconsistent intermediate output " + name);
    }
    module_->set_timing_label("alpamayo.vision");
}

void VisionEncoder::validate(const std::string& name, const Tensor& tensor,
                             const std::vector<int64_t>& shape) const {
    require(tensor.data != nullptr && tensor.shape == shape, "invalid shape/data for " + name);
    require(tensor.dtype == module_->tensor_dtype(name), "invalid dtype for " + name);
    checked_bytes(shape, tensor.dtype);
    const auto minimum = module_->input_profile_shape(name, 0, ProfileShapeSelector::kMin);
    const auto maximum = module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax);
    require(minimum.size() == shape.size() && maximum.size() == shape.size(),
            "invalid profile rank for " + name);
    for (std::size_t axis = 0; axis < shape.size(); ++axis)
        require(shape[axis] >= minimum[axis] && shape[axis] <= maximum[axis],
                "shape outside engine capacity for " + name);
    if (tensor.dtype == DType::kInt32) {
        const auto* indices = static_cast<const int32_t*>(tensor.data);
        for (std::size_t i = 0; i < tensor.numel(); ++i)
            require(indices[i] >= 0 && indices[i] < positions_,
                    "position index outside learned table");
    } else {
        const auto* values = static_cast<const float*>(tensor.data);
        for (std::size_t i = 0; i < tensor.numel(); ++i) {
            const bool masked = name == "mask" && std::isinf(values[i]) && values[i] < 0;
            require(std::isfinite(values[i]) || masked, "non-finite input " + name);
        }
    }
}

VisionFeatures VisionEncoder::encode(const TensorMap& inputs) {
    require(inputs.size() == 6, "incomplete prepared input set");
    const auto pixels = inputs.find("pixels");
    require(pixels != inputs.end() && pixels->second.shape.size() == 2, "missing patch shape");
    const auto count = pixels->second.shape[0];
    require(count > 0 && count % merge_unit_ == 0, "patch count is not a complete merge group");
    for (const auto& [name, shape] : std::vector<std::pair<std::string, std::vector<int64_t>>>{
             {"pixels", {count, flat_patch_}},
             {"position_indices", {4, count}},
             {"position_weights", {4, count}},
             {"mask", {1, count, count}},
             {"cosine", {count, 1, head_width_}},
             {"sine", {count, 1, head_width_}}}) {
        const auto found = inputs.find(name);
        require(found != inputs.end(), "missing input " + name);
        validate(name, found->second, shape);
    }
    VisionFeatures result;
    result.tokens = count / merge_unit_;
    result.width = feature_width_;
    result.dtype = output_dtype_;
    result.deepstack.resize(deepstack_count_);
    const std::vector<int64_t> shape{result.tokens, result.width};
    const auto bytes = checked_bytes(shape, result.dtype);
    const auto outputs = module_->forward(inputs);
    if (outputs.size() != 1 + deepstack_count_)
        throw std::runtime_error("Alpamayo vision: incomplete output set");
    for (std::size_t i = 0; i <= deepstack_count_; ++i) {
        const auto name = i == 0 ? std::string("features") : "deepstack_" + std::to_string(i - 1);
        const auto found = outputs.find(name);
        if (found == outputs.end() || !found->second.data || found->second.shape != shape ||
            found->second.dtype != result.dtype)
            throw std::runtime_error("Alpamayo vision: invalid output " + name);
        const auto& output = found->second;
        for (std::size_t index = 0; index < output.numel(); ++index) {
            const bool finite =
                result.dtype == DType::kFloat16
                    ? (static_cast<const uint16_t*>(output.data)[index] & 0x7c00U) != 0x7c00U
                    : std::isfinite(static_cast<const float*>(output.data)[index]);
            if (!finite)
                throw std::runtime_error("Alpamayo vision: non-finite output " + name);
        }
        auto& storage = i == 0 ? result.features : result.deepstack[i - 1];
        storage.resize(bytes);
        std::memcpy(storage.data(), output.data, bytes);
    }
    return result;
}

} // namespace trtmc::alpamayo
