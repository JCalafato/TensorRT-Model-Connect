/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::alpamayo {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("Alpamayo preprocessing: " + message);
}

std::size_t checked_count(std::initializer_list<int64_t> dimensions, std::size_t item_size) {
    std::size_t count = 1;
    for (const auto dimension : dimensions) {
        require(dimension > 0 && static_cast<uint64_t>(dimension) <=
                                     std::numeric_limits<std::size_t>::max() / item_size / count,
                "tensor size overflow");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

// uint8 antialiased bicubic, including fixed-point coefficient conversion and
// per-axis uint8 rounding. Float interpolation before normalization is not
// equivalent to the pinned provider's fast image processor.
struct AntialiasWeights {
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int16_t> values;
    int32_t stride{0};
    uint32_t precision{0};
};

static double keys_cubic(double x) {
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

static int32_t aligned_coefficient_stride(int32_t kernel_size) {
    int32_t stride = kernel_size;
    while (stride % static_cast<int32_t>(sizeof(int32_t)) != 0)
        ++stride;
    return stride;
}

static double build_floating_antialias_weights(int32_t input_size, int32_t output_size,
                                               double scale, double support, int32_t kernel_size,
                                               AntialiasWeights& result,
                                               std::vector<double>& floating) {
    double maximum_weight = 0.0;
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double center = scale * (output_index + 0.5);
        const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
        const int32_t start = static_cast<int32_t>(std::max(center - support + 0.5, 0.0));
        const int32_t size =
            std::clamp(static_cast<int32_t>(
                           std::min(center + support + 0.5, static_cast<double>(input_size))) -
                           start,
                       0, kernel_size);
        result.starts[static_cast<std::size_t>(output_index)] = start;
        result.sizes[static_cast<std::size_t>(output_index)] = size;

        double total = 0.0;
        double* row = floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        for (int32_t index = 0; index < size; ++index) {
            row[index] = keys_cubic((index + start - center + 0.5) * inverse_scale);
            total += row[index];
        }
        if (total != 0.0) {
            for (int32_t index = 0; index < size; ++index) {
                row[index] /= total;
                maximum_weight = std::max(maximum_weight, row[index]);
            }
        }
    }
    return maximum_weight;
}

static void quantize_antialias_weights(const std::vector<double>& floating, int32_t output_size,
                                       int32_t kernel_size, double maximum_weight,
                                       AntialiasWeights& result) {
    // Select the greatest fixed-point precision whose largest coefficient fits
    // in int16, exactly as PyTorch's uint8 antialias implementation does.
    for (; result.precision < 22; ++result.precision) {
        const int32_t next = static_cast<int32_t>(
            0.5 + maximum_weight * static_cast<double>(uint32_t{1} << (result.precision + 1)));
        if (next >= (1 << 15))
            break;
    }

    const double multiplier = static_cast<double>(uint32_t{1} << result.precision);
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const double* source =
            floating.data() + static_cast<std::size_t>(output_index) * kernel_size;
        int16_t* destination =
            result.values.data() + static_cast<std::size_t>(output_index) * result.stride;
        for (int32_t index = 0; index < kernel_size; ++index) {
            const double value = source[index] * multiplier;
            destination[index] =
                static_cast<int16_t>(value < 0.0 ? static_cast<int32_t>(value - 0.5)
                                                 : static_cast<int32_t>(value + 0.5));
        }
    }
}

static AntialiasWeights make_bicubic_antialias_weights(int32_t input_size, int32_t output_size) {
    constexpr int32_t interp_size = 4;
    const double scale = static_cast<double>(input_size) / output_size;
    const double support = scale >= 1.0 ? (interp_size * 0.5) * scale : interp_size * 0.5;
    require(std::ceil(support) <= (std::numeric_limits<int32_t>::max() - 8) / 2,
            "resize support exceeds the index range");
    const int32_t kernel_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;

    // PyTorch pads each int16 coefficient row to a 32-bit-aligned size in its
    // optimized uint8 path. The padding does not participate in convolution.
    AntialiasWeights result;
    result.stride = aligned_coefficient_stride(kernel_size);
    result.starts.resize(static_cast<std::size_t>(output_size));
    result.sizes.resize(static_cast<std::size_t>(output_size));
    result.values.assign(static_cast<std::size_t>(output_size) * result.stride, 0);

    std::vector<double> floating(checked_count({output_size, kernel_size}, sizeof(double)), 0.0);
    const double maximum_weight = build_floating_antialias_weights(
        input_size, output_size, scale, support, kernel_size, result, floating);
    quantize_antialias_weights(floating, output_size, kernel_size, maximum_weight, result);
    return result;
}

static uint8_t fixed_point_pixel(const uint8_t* source, int32_t source_stride,
                                 const int16_t* weights, int32_t count, uint32_t precision) {
    int32_t value = int32_t{1} << (precision - 1);
    for (int32_t index = 0; index < count; ++index)
        value += static_cast<int32_t>(source[static_cast<std::size_t>(index) * source_stride]) *
                 weights[index];
    return static_cast<uint8_t>(std::clamp(value >> precision, 0, 255));
}

static bool valid_resize_dimensions(const unsigned char* raw, int32_t width, int32_t height,
                                    int32_t target_width, int32_t target_height) {
    return raw != nullptr && width > 0 && height > 0 && target_width > 0 && target_height > 0;
}

static std::vector<unsigned char> resize_bicubic_horizontal(const unsigned char* raw, int32_t width,
                                                            int32_t height, int32_t target_width) {
    const auto weights = make_bicubic_antialias_weights(width, target_width);
    std::vector<unsigned char> resized(static_cast<std::size_t>(height) * target_width * 3);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            const int32_t start = weights.starts[static_cast<std::size_t>(x)];
            const int32_t count = weights.sizes[static_cast<std::size_t>(x)];
            const int16_t* coefficients =
                weights.values.data() + static_cast<std::size_t>(x) * weights.stride;
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset =
                    (static_cast<std::size_t>(y) * width + start) * 3 + channel;
                const auto destination_offset =
                    (static_cast<std::size_t>(y) * target_width + x) * 3 + channel;
                resized[destination_offset] = fixed_point_pixel(
                    raw + source_offset, 3, coefficients, count, weights.precision);
            }
        }
    }
    return resized;
}

static std::vector<unsigned char> resize_bicubic_vertical(const unsigned char* raw, int32_t width,
                                                          int32_t height, int32_t target_height) {
    const auto weights = make_bicubic_antialias_weights(height, target_height);
    std::vector<unsigned char> resized(static_cast<std::size_t>(target_height) * width * 3);
    const int32_t row_stride = width * 3;
    for (int32_t y = 0; y < target_height; ++y) {
        const int32_t start = weights.starts[static_cast<std::size_t>(y)];
        const int32_t count = weights.sizes[static_cast<std::size_t>(y)];
        const int16_t* coefficients =
            weights.values.data() + static_cast<std::size_t>(y) * weights.stride;
        for (int32_t x = 0; x < width; ++x) {
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto source_offset =
                    (static_cast<std::size_t>(start) * width + x) * 3 + channel;
                const auto destination_offset =
                    (static_cast<std::size_t>(y) * width + x) * 3 + channel;
                resized[destination_offset] = fixed_point_pixel(
                    raw + source_offset, row_stride, coefficients, count, weights.precision);
            }
        }
    }
    return resized;
}

std::vector<unsigned char> resize_bicubic_antialias(const unsigned char* raw, int32_t width,
                                                    int32_t height, int32_t target_width,
                                                    int32_t target_height) {
    if (!valid_resize_dimensions(raw, width, height, target_width, target_height))
        return {};
    if (width == target_width && height == target_height) {
        return {raw, raw + static_cast<std::size_t>(width) * height * 3};
    }

    std::vector<unsigned char> horizontal;
    const unsigned char* vertical_source = raw;
    if (width != target_width) {
        horizontal = resize_bicubic_horizontal(raw, width, height, target_width);
        vertical_source = horizontal.data();
    }

    if (height == target_height)
        return horizontal;

    return resize_bicubic_vertical(vertical_source, target_width, height, target_height);
}

std::array<int32_t, 2> smart_resize(int32_t height, int32_t width,
                                    const VisionPreprocessConfig& config) {
    require(height > 0 && width > 0, "image dimensions must be positive");
    require(static_cast<double>(std::max(height, width)) / std::min(height, width) <= 200.0,
            "image aspect ratio exceeds 200");
    const int64_t factor = static_cast<int64_t>(config.patch_size) * config.spatial_merge_size;
    const auto round_even = [factor](int32_t extent) {
        const auto quotient = extent / factor;
        const auto twice_remainder = 2 * (extent % factor);
        return (quotient +
                (twice_remainder > factor || (twice_remainder == factor && quotient % 2 != 0))) *
               factor;
    };
    double h = static_cast<double>(round_even(height));
    double w = static_cast<double>(round_even(width));
    const double source_area = static_cast<double>(height) * width;
    if (h * w > static_cast<double>(config.max_pixels)) {
        const auto beta = std::sqrt(source_area / static_cast<double>(config.max_pixels));
        h = std::max(1.0, std::floor(height / beta / factor)) * factor;
        w = std::max(1.0, std::floor(width / beta / factor)) * factor;
    } else if (h * w < static_cast<double>(config.min_pixels)) {
        const auto beta = std::sqrt(static_cast<double>(config.min_pixels) / source_area);
        h = std::ceil(height * beta / factor) * factor;
        w = std::ceil(width * beta / factor) * factor;
    }
    require(h > 0 && w > 0 && h <= std::numeric_limits<int32_t>::max() &&
                w <= std::numeric_limits<int32_t>::max() / 3,
            "resized geometry exceeds the index range");
    return {static_cast<int32_t>(h), static_cast<int32_t>(w)};
}

float linear_position(int32_t index, int32_t count, int32_t side) {
    if (count == 1)
        return 0.0F;
    const float end = static_cast<float>(side - 1);
    const float step = end / static_cast<float>(count - 1);
    // Match the reference's two-ended FP32 linspace, including its rounding.
    return index < count / 2 ? step * index
                             : std::fma(-step, static_cast<float>(count - index - 1), end);
}

} // namespace

TensorMap PreparedVisionInput::tensors() {
    const auto count = grid[1] * grid[2];
    return {{"pixels", {pixels.data(), {count, flat_patch}, DType::kFloat32}},
            {"position_indices", {position_indices.data(), {4, count}, DType::kInt32}},
            {"position_weights", {position_weights.data(), {4, count}, DType::kFloat32}},
            {"mask", {mask.data(), {1, count, count}, DType::kFloat32}},
            {"cosine", {cosine.data(), {count, 1, head_width}, DType::kFloat32}},
            {"sine", {sine.data(), {count, 1, head_width}, DType::kFloat32}}};
}

VisionPreprocessor::VisionPreprocessor(VisionPreprocessConfig config) : config_(config) {
    require(
        config.patch_size > 0 && config.temporal_patch_size > 0 && config.spatial_merge_size > 0 &&
            config.patch_size <= std::numeric_limits<int32_t>::max() / config.spatial_merge_size,
        "invalid patch dimensions");
    require(config.num_position_embeddings > 0, "invalid learned position table");
    position_side_ = static_cast<int32_t>(std::sqrt(config.num_position_embeddings));
    require(static_cast<int64_t>(position_side_) * position_side_ == config.num_position_embeddings,
            "learned positions must form a square");
    require(config.head_width > 0 && config.head_width % 4 == 0, "invalid rotary head width");
    require(config.min_pixels > 0 && config.max_pixels >= config.min_pixels &&
                config.patch_capacity > 0,
            "invalid image bounds or engine capacity");
    require(std::isfinite(config.rescale_factor) && config.rescale_factor > 0 &&
                std::isfinite(static_cast<float>(1.0 / config.rescale_factor)),
            "invalid image rescaling");
    const float inverse_rescale = static_cast<float>(1.0 / config.rescale_factor);
    for (std::size_t channel = 0; channel < 3; ++channel) {
        require(std::isfinite(config.image_mean[channel]) &&
                    std::isfinite(config.image_mean[channel] * inverse_rescale) &&
                    std::isfinite(config.image_std[channel]) && config.image_std[channel] > 0 &&
                    std::isfinite(config.image_std[channel] * inverse_rescale) &&
                    config.image_std[channel] * inverse_rescale > 0,
                "invalid image normalization");
    }
    // Establish fixed patch sizing before accepting any image or allocating buffers.
    const auto flat = checked_count(
        {3, config.temporal_patch_size, config.patch_size, config.patch_size}, sizeof(float));
    require(flat <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "patch width overflow");
}

PreparedVisionInput VisionPreprocessor::prepare(const uint8_t* rgb, std::size_t bytes,
                                                int32_t height, int32_t width) const {
    require(rgb && height > 0 && width > 0 && width <= std::numeric_limits<int32_t>::max() / 3,
            "invalid RGB image geometry/data");
    require(bytes == checked_count({height, width, 3}, 1),
            "RGB byte count does not match geometry");
    const auto shape = smart_resize(height, width, config_);
    const auto gh = shape[0] / config_.patch_size;
    const auto gw = shape[1] / config_.patch_size;
    const int64_t count = static_cast<int64_t>(gh) * gw;
    require(count <= config_.patch_capacity, "image exceeds engine patch capacity");
    const auto flat = static_cast<int64_t>(checked_count(
        {3, config_.temporal_patch_size, config_.patch_size, config_.patch_size}, sizeof(float)));
    const auto pixel_count = checked_count({count, flat}, sizeof(float));
    const auto index_count = checked_count({4, count}, sizeof(float));
    const auto mask_count = checked_count({count, count}, sizeof(float));
    const auto rotary_count = checked_count({count, config_.head_width}, sizeof(float));
    checked_count({shape[0], shape[1], 3}, 1);
    checked_count({height, shape[1], 3}, 1);
    auto resized = resize_bicubic_antialias(rgb, width, height, shape[1], shape[0]);
    PreparedVisionInput result;
    result.grid = {1, gh, gw};
    result.flat_patch = flat;
    result.head_width = config_.head_width;
    result.pixels.resize(pixel_count);
    result.position_indices.resize(index_count);
    result.position_weights.resize(index_count);
    result.mask.assign(mask_count, 0.0F);
    result.cosine.resize(rotary_count);
    result.sine.resize(rotary_count);
    const auto merge = config_.spatial_merge_size;
    const auto patch = config_.patch_size;
    const auto frequencies = config_.head_width / 4;
    std::vector<float> inverse_frequency(frequencies);
    for (int32_t i = 0; i < frequencies; ++i)
        inverse_frequency[i] =
            1.0F / std::pow(10000.0F, static_cast<float>(2 * i) / (config_.head_width / 2));
    const float inverse_rescale = static_cast<float>(1.0 / config_.rescale_factor);
    std::size_t p = 0;
    for (int32_t by = 0; by < gh; by += merge) {
        for (int32_t bx = 0; bx < gw; bx += merge) {
            for (int32_t iy = 0; iy < merge; ++iy) {
                for (int32_t ix = 0; ix < merge; ++ix, ++p) {
                    const auto y = by + iy;
                    const auto x = bx + ix;
                    std::size_t destination = p * flat;
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        const float mean = config_.image_mean[channel] * inverse_rescale;
                        const float stddev = config_.image_std[channel] * inverse_rescale;
                        for (int32_t t = 0; t < config_.temporal_patch_size; ++t) {
                            for (int32_t py = 0; py < patch; ++py) {
                                for (int32_t px = 0; px < patch; ++px) {
                                    const auto offset =
                                        (static_cast<std::size_t>(y * patch + py) * shape[1] +
                                         x * patch + px) *
                                            3 +
                                        channel;
                                    result.pixels[destination++] =
                                        (resized[offset] - mean) / stddev;
                                }
                            }
                        }
                    }
                    const float yp = linear_position(y, gh, position_side_);
                    const float xp = linear_position(x, gw, position_side_);
                    const auto y0 = static_cast<int32_t>(yp), x0 = static_cast<int32_t>(xp);
                    const auto y1 = std::min(y0 + 1, position_side_ - 1);
                    const auto x1 = std::min(x0 + 1, position_side_ - 1);
                    const float dy = yp - y0, dx = xp - x0;
                    const std::array<int32_t, 4> ids{
                        y0 * position_side_ + x0, y0 * position_side_ + x1,
                        y1 * position_side_ + x0, y1 * position_side_ + x1};
                    const std::array<float, 4> weights{(1 - dy) * (1 - dx), (1 - dy) * dx,
                                                       dy * (1 - dx), dy * dx};
                    for (std::size_t corner = 0; corner < 4; ++corner) {
                        result.position_indices[corner * count + p] = ids[corner];
                        result.position_weights[corner * count + p] = weights[corner];
                    }
                    for (int32_t i = 0; i < config_.head_width; ++i) {
                        const auto part = i % (2 * frequencies);
                        const float phase = static_cast<float>(part < frequencies ? y : x) *
                                            inverse_frequency[part % frequencies];
                        result.cosine[p * config_.head_width + i] = std::cos(phase);
                        result.sine[p * config_.head_width + i] = std::sin(phase);
                    }
                }
            }
        }
    }
    return result;
}

} // namespace trtmc::alpamayo
