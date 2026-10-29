/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::diffusion_gemma {
namespace {
// Checkpoint image processing uses torchvision RGB8 antialiased bicubic
// coefficients, independently of the TensorRT device executing the graph.
struct Filter {
    int start;
    std::vector<int32_t> weights;
    int precision{22};
};

double cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.5 * x - 2.5) * x) * x + 1;
    return x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0;
}

std::vector<Filter> filters(int source, int target, bool torchvision_coefficients) {
    const double scale = static_cast<double>(source) / target;
    const double stretch = std::max(1.0, scale);
    std::vector<Filter> result;
    std::vector<std::vector<double>> normalized;
    double largest = 0;
    for (int out = 0; out < target; ++out) {
        const double center = (out + 0.5) * scale;
        const int start = std::max(0, static_cast<int>(center - 2 * stretch + 0.5));
        const int end = std::min(source, static_cast<int>(center + 2 * stretch + 0.5));
        std::vector<double> values;
        double sum = 0;
        for (int in = start; in < end; ++in) {
            const double value = cubic((in - center + 0.5) / stretch);
            values.push_back(value);
            sum += value;
        }
        for (auto& value : values) {
            value /= sum;
            largest = std::max(largest, value);
        }
        normalized.push_back(std::move(values));
        result.push_back({start, {}, 22});
    }
    int precision = 22;
    if (torchvision_coefficients) {
        // Match torchvision's RGB8 antialias coefficient quantization, not
        // Pillow's fixed 22-bit coefficients. This is input preprocessing
        // semantics, independent of the GPU executing the model.
        // Reference: PyTorch UpSampleKernel.cpp, _compute_index_ranges_int16_weights.
        precision = 0;
        while (precision < 22 && std::lround(largest * (int64_t{1} << (precision + 1))) < 32768)
            ++precision;
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].precision = precision;
        for (double value : normalized[i])
            result[i].weights.push_back(
                static_cast<int32_t>(std::lround(value * (int64_t{1} << precision))));
    }
    return result;
}

std::vector<uint8_t> resize_rgb(const std::vector<uint8_t>& source, int width, int height,
                                int target_width, int target_height,
                                bool torchvision_coefficients) {
    const auto horizontal = filters(width, target_width, torchvision_coefficients);
    std::vector<uint8_t> intermediate(static_cast<std::size_t>(height) * target_width * 3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < target_width; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto& filter = horizontal[x];
                int64_t value = int64_t{1} << (filter.precision - 1);
                for (std::size_t i = 0; i < filter.weights.size(); ++i)
                    value +=
                        static_cast<int64_t>(filter.weights[i]) *
                        source[(static_cast<std::size_t>(y) * width + filter.start + i) * 3 + c];
                intermediate[(static_cast<std::size_t>(y) * target_width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp<int64_t>(value >> filter.precision, 0, 255));
            }
    const auto vertical = filters(height, target_height, torchvision_coefficients);
    std::vector<uint8_t> output(static_cast<std::size_t>(target_height) * target_width * 3);
    for (int y = 0; y < target_height; ++y)
        for (int x = 0; x < target_width; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto& filter = vertical[y];
                int64_t value = int64_t{1} << (filter.precision - 1);
                for (std::size_t i = 0; i < filter.weights.size(); ++i)
                    value += static_cast<int64_t>(filter.weights[i]) *
                             intermediate[((filter.start + i) * target_width + x) * 3 + c];
                output[(static_cast<std::size_t>(y) * target_width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp<int64_t>(value >> filter.precision, 0, 255));
            }
    return output;
}

} // namespace

ImagePatches image_patches(const internal::ImageView& image, const ImageConfig& config) {
    if (!image.data || !image.width || !image.height || image.channels != 3 ||
        image.width > INT32_MAX || image.height > INT32_MAX ||
        (image.format != internal::ImageFormat::UInt8 &&
         image.format != internal::ImageFormat::Float32))
        throw std::invalid_argument(
            "DiffusionGemma requires a nonempty host RGB8 or RGB float image");
    const uint64_t count = static_cast<uint64_t>(image.width) * image.height * 3;
    const std::size_t item = image.format == internal::ImageFormat::UInt8 ? 1 : sizeof(float);
    if (count > std::numeric_limits<std::size_t>::max() / item || image.byte_size != count * item)
        throw std::invalid_argument("DiffusionGemma RGB storage does not match its dimensions");
    if (config.patch_size <= 0 || config.pooling_kernel < 2 || config.soft_tokens <= 0)
        throw std::invalid_argument("DiffusionGemma invalid image patch geometry");
    std::vector<uint8_t> pixels(static_cast<std::size_t>(count));
    if (image.format == internal::ImageFormat::UInt8) {
        std::memcpy(pixels.data(), image.data, pixels.size());
    } else {
        const auto* bytes = static_cast<const unsigned char*>(image.data);
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            float value;
            std::memcpy(&value, bytes + i * sizeof(float), sizeof(float));
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::invalid_argument(
                    "DiffusionGemma RGB float pixels must be finite in [0,1]");
            pixels[i] = static_cast<uint8_t>(std::lround(value * 255));
        }
    }
    const int teacher_patch = config.patch_size, kernel = config.pooling_kernel;
    const int patch = teacher_patch;
    const int patches = config.soft_tokens * kernel * kernel;
    const int multiple = teacher_patch * kernel;
    const double scale = std::sqrt(static_cast<double>(patches) * teacher_patch * teacher_patch /
                                   (static_cast<double>(image.width) * image.height));
    int height = static_cast<int>(std::floor(scale * image.height / multiple)) * multiple;
    int width = static_cast<int>(std::floor(scale * image.width / multiple)) * multiple;
    if (height == 0) {
        height = multiple;
        width =
            static_cast<int>(std::min<double>(
                std::floor(static_cast<double>(image.width) / image.height), config.soft_tokens)) *
            multiple;
    } else if (width == 0) {
        width = multiple;
        height =
            static_cast<int>(std::min<double>(
                std::floor(static_cast<double>(image.height) / image.width), config.soft_tokens)) *
            multiple;
    }
    if (width <= 0 || height <= 0 ||
        static_cast<int64_t>(width) * height > int64_t{patches} * teacher_patch * teacher_patch)
        throw std::invalid_argument("DiffusionGemma resized image exceeds its patch budget");
    const auto resized = resize_rgb(pixels, image.width, image.height, width, height, true);
    const int columns = width / patch, rows = height / patch;
    ImagePatches result;
    result.soft_tokens = rows * columns / (kernel * kernel);
    const int capacity = patches;
    result.pixels.assign(static_cast<std::size_t>(capacity) * patch * patch * 3, 0);
    result.positions.assign(static_cast<std::size_t>(capacity) * 2, -1);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < columns; ++x) {
            const int index = y * columns + x;
            result.positions[2 * index] = x;
            result.positions[2 * index + 1] = y;
            for (int py = 0; py < patch; ++py)
                for (int px = 0; px < patch; ++px)
                    for (int c = 0; c < 3; ++c)
                        result
                            .pixels[((static_cast<std::size_t>(index) * patch + py) * patch + px) *
                                        3 +
                                    c] =
                            resized[((y * patch + py) * width + x * patch + px) * 3 + c] *
                            (1.0F / 255.0F);
        }
    return result;
}

VisionEncoder::VisionEncoder(std::unique_ptr<ITrtModule> module, ImageConfig config)
    : module_(std::move(module)), config_(config) {
    if (!module_ || !module_->ok() || config_.hidden_size <= 0)
        throw std::runtime_error("DiffusionGemma invalid native vision engine");
}

std::vector<float> VisionEncoder::encode(const ImagePatches& input) {
    const int kernel = config_.pooling_kernel;
    const int patch = config_.patch_size;
    const int patches = config_.soft_tokens * kernel * kernel;
    TensorMap inputs;
    inputs["pixel_values"] = {
        const_cast<float*>(input.pixels.data()), {patches, 3 * patch * patch}, DType::kFloat32};
    inputs["pixel_position_ids"] = {
        const_cast<int32_t*>(input.positions.data()), {patches, 2}, DType::kInt32};
    const auto outputs = module_->forward(inputs);
    const auto& features = outputs.at("image_features");
    const auto& mask = outputs.at("image_feature_mask");
    if (features.dtype != DType::kFloat32 || mask.dtype != DType::kInt32 ||
        features.numel() != static_cast<std::size_t>(config_.soft_tokens) * config_.hidden_size ||
        mask.numel() != static_cast<std::size_t>(config_.soft_tokens))
        throw std::runtime_error("DiffusionGemma vision engine output contract mismatch");
    const auto* values = static_cast<const float*>(features.data);
    const auto* valid = static_cast<const int32_t*>(mask.data);
    std::vector<float> result;
    for (int i = 0; i < config_.soft_tokens; ++i)
        if (valid[i])
            result.insert(result.end(), values + i * config_.hidden_size,
                          values + (i + 1) * config_.hidden_size);
    if (result.size() != static_cast<std::size_t>(input.soft_tokens) * config_.hidden_size ||
        !std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error(
            "DiffusionGemma vision engine returned invalid conditioned features");
    return result;
}

} // namespace trtmc::diffusion_gemma
