/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc::alpamayo {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("Alpamayo text: " + message);
}

std::size_t checked_bytes(const std::vector<int64_t>& shape, DType dtype) {
    auto count = dtype_size(dtype);
    require(count != 0 && !shape.empty(), "invalid tensor dtype/rank");
    for (const auto dimension : shape) {
        require(dimension >= 0, "negative tensor size");
        require(count == 0 || static_cast<uint64_t>(dimension) <=
                                  std::numeric_limits<std::size_t>::max() / count,
                "tensor size overflow");
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

bool finite(const Tensor& tensor, bool allow_negative_infinity = false) {
    if (tensor.dtype == DType::kFloat16) {
        const auto* data = static_cast<const uint16_t*>(tensor.data);
        for (std::size_t i = 0; i < tensor.numel(); ++i)
            if ((data[i] & 0x7c00U) == 0x7c00U)
                return false;
    } else if (tensor.dtype == DType::kFloat32) {
        const auto* data = static_cast<const float*>(tensor.data);
        for (std::size_t i = 0; i < tensor.numel(); ++i)
            if (!std::isfinite(data[i]) &&
                !(allow_negative_infinity && std::isinf(data[i]) && data[i] < 0))
                return false;
    }
    return true;
}

} // namespace

TensorMap TextPositions::tensors() {
    const std::vector<int64_t> shape{batch, 1, length, width};
    return {{"cosine", Tensor{cosine.data(), shape, dtype}},
            {"sine", Tensor{sine.data(), shape, dtype}}};
}

TextPositions prepare_image_positions(const std::vector<std::vector<int32_t>>& tokens,
                                      const std::vector<std::array<int64_t, 3>>& image_grids,
                                      const std::vector<std::vector<int32_t>>& attention_mask,
                                      const ImagePositionConfig& config, DType dtype) {
    require(!tokens.empty() && !tokens[0].empty(), "empty image-position prompt");
    require(config.image_token >= 0 && config.video_token >= 0 && config.vision_start_token >= 0 &&
                config.vision_end_token >= 0 && config.spatial_merge_size > 0 &&
                config.head_width > 0 && config.head_width <= INT32_MAX &&
                config.head_width % 2 == 0 && std::isfinite(config.rope_theta) &&
                config.rope_theta > 0,
            "invalid image-position configuration");
    const std::array<int32_t, 4> special{config.image_token, config.video_token,
                                         config.vision_start_token, config.vision_end_token};
    for (std::size_t i = 0; i < special.size(); ++i)
        for (std::size_t j = i + 1; j < special.size(); ++j)
            require(special[i] != special[j], "image-position special tokens must be distinct");
    const auto half = config.head_width / 2;
    for (const auto section : config.mrope_section)
        require(section >= 0 && section <= half, "invalid interleaved rotary section");
    require(std::accumulate(config.mrope_section.begin(), config.mrope_section.end(), int64_t{0}) ==
                    half &&
                (config.mrope_section[1] == 0 || 3 * config.mrope_section[1] - 2 < half) &&
                (config.mrope_section[2] == 0 || 3 * config.mrope_section[2] - 1 < half),
            "inconsistent interleaved rotary sections");
    require(dtype == DType::kFloat32 || dtype == DType::kFloat16, "invalid rotary-table precision");
    const auto count = tokens.size(), length = tokens[0].size();
    require(length <= static_cast<uint64_t>(INT64_MAX) &&
                count <= std::numeric_limits<std::size_t>::max() / sizeof(int64_t) / 3 / length,
            "image-position shape overflow");
    require(attention_mask.empty() || attention_mask.size() == count,
            "invalid image-position attention batch");
    TextPositions result;
    result.batch = static_cast<int64_t>(count);
    result.length = static_cast<int64_t>(length);
    result.width = config.head_width;
    result.dtype = dtype;
    result.axes.assign(3 * count * length, 1);
    std::size_t grid_index = 0;
    for (std::size_t row = 0; row < count; ++row) {
        require(tokens[row].size() == length &&
                    (attention_mask.empty() || attention_mask[row].size() == length),
                "nonrectangular image-position prompt");
        std::vector<std::size_t> active;
        for (std::size_t column = 0; column < length; ++column) {
            const auto mask = attention_mask.empty() ? 1 : attention_mask[row][column];
            require(mask == 0 || mask == 1, "image-position attention mask must be binary");
            require(tokens[row][column] >= 0, "negative image-position token");
            if (mask)
                active.push_back(column);
        }
        require(!active.empty(), "image-position row has no active tokens");
        const auto id = [&](std::size_t column) { return tokens[row][active[column]]; };
        const auto put = [&](int axis, std::size_t column, int64_t value) {
            result.axes[(static_cast<std::size_t>(axis) * count + row) * length + active[column]] =
                value;
        };
        int64_t next = 0;
        for (std::size_t column = 0; column < active.size();) {
            require(id(column) != config.video_token,
                    "Alpamayo camera prompts use image grids, not video tokens");
            if (id(column) == config.vision_start_token)
                require(column + 1 < active.size() && id(column + 1) == config.image_token,
                        "vision start requires an image block");
            if (id(column) != config.image_token) {
                require(next < INT64_MAX, "text position overflow");
                for (int axis = 0; axis < 3; ++axis)
                    put(axis, column, next);
                ++next;
                ++column;
                continue;
            }
            require(column > 0 && id(column - 1) == config.vision_start_token &&
                        grid_index < image_grids.size(),
                    "image token lacks a matching grid/start");
            const auto grid = image_grids[grid_index++];
            require(grid[0] > 0 && grid[1] > 0 && grid[2] > 0 &&
                        grid[1] % config.spatial_merge_size == 0 &&
                        grid[2] % config.spatial_merge_size == 0,
                    "invalid image grid or merge divisibility");
            const auto t = grid[0], h = grid[1] / config.spatial_merge_size,
                       w = grid[2] / config.spatial_merge_size;
            const auto remaining = active.size() - column;
            require(static_cast<uint64_t>(t) <=
                        remaining / static_cast<uint64_t>(h) / static_cast<uint64_t>(w),
                    "image grid exceeds available tokens");
            const auto size = static_cast<std::size_t>(t * h * w);
            require(size < remaining && id(column + size) == config.vision_end_token,
                    "image block length or end marker mismatch");
            require(next <= INT64_MAX - std::max({t, h, w}), "image position overflow");
            for (std::size_t i = 0; i < size; ++i) {
                require(id(column + i) == config.image_token, "interrupted image token block");
                put(0, column + i, next + static_cast<int64_t>(i) / (h * w));
                put(1, column + i, next + static_cast<int64_t>(i) / w % h);
                put(2, column + i, next + static_cast<int64_t>(i) % w);
            }
            next += std::max({t, h, w});
            column += size;
        }
        // With no grids the provider derives the maximum from the masked
        // position tensor itself, whose padding positions equal one.
        if (image_grids.empty() && active.size() != length)
            next = std::max(next, int64_t{2});
        result.deltas.push_back(next - result.length);
    }
    require(grid_index == image_grids.size(), "unused image grids in prompt");
    result.cosine.resize(checked_bytes({result.batch, 1, result.length, result.width}, dtype));
    result.sine.resize(result.cosine.size());
    std::vector<float> inverse(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i)
        inverse[i] = 1.0F / std::pow(config.rope_theta, static_cast<float>(2 * i) /
                                                            static_cast<float>(config.head_width));
    const auto store = [&](std::vector<uint8_t>& bytes, std::size_t index, float value) {
        if (dtype == DType::kFloat32)
            std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
        else {
            const auto converted = __float2half_rn(value);
            std::memcpy(bytes.data() + index * sizeof(converted), &converted, sizeof(converted));
        }
    };
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < length; ++column) {
            for (int64_t index = 0; index < config.head_width; ++index) {
                const auto frequency = index % half;
                const int axis = frequency % 3 == 1 && frequency < config.mrope_section[1] * 3 ? 1
                                 : frequency % 3 == 2 && frequency < config.mrope_section[2] * 3
                                     ? 2
                                     : 0;
                const auto position =
                    result.axes[(static_cast<std::size_t>(axis) * count + row) * length + column];
                const float angle = static_cast<float>(position) * inverse[frequency];
                const auto output =
                    (row * length + column) * static_cast<std::size_t>(config.head_width) + index;
                store(result.cosine, output, std::cos(angle));
                store(result.sine, output, std::sin(angle));
            }
        }
    }
    return result;
}

std::vector<double> sampling_probabilities(const std::vector<float>& logits,
                                           const SamplingOptions& options) {
    require(!logits.empty() && logits.size() <= static_cast<std::size_t>(INT32_MAX),
            "invalid sampling vocabulary");
    require(std::isfinite(options.temperature) && options.temperature >= 0 &&
                std::isfinite(options.top_p) && options.top_p >= 0 && options.top_p <= 1 &&
                options.top_k >= 0,
            "invalid sampling options");
    require(options.suppressed_begin >= 0 && options.suppressed_end >= options.suppressed_begin &&
                static_cast<std::size_t>(options.suppressed_end) <= logits.size(),
            "invalid suppressed token interval");
    std::vector<float> scores = logits;
    std::vector<int32_t> order;
    int32_t maximum = -1;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        require(std::isfinite(logits[i]) || (std::isinf(logits[i]) && logits[i] < 0),
                "invalid sampling logits");
        if (static_cast<int64_t>(i) >= options.suppressed_begin &&
            static_cast<int64_t>(i) < options.suppressed_end)
            continue;
        if (!std::isfinite(logits[i]))
            continue;
        if (maximum < 0 || logits[i] > logits[static_cast<std::size_t>(maximum)])
            maximum = static_cast<int32_t>(i);
        if (options.temperature > 0) {
            scores[i] /= options.temperature;
            require(std::isfinite(scores[i]), "temperature-scaled score overflow");
        }
        order.push_back(static_cast<int32_t>(i));
    }
    require(maximum >= 0, "no finite permitted sampling token");
    std::vector<double> probabilities(logits.size(), 0.0);
    if (options.temperature == 0) {
        probabilities[static_cast<std::size_t>(maximum)] = 1.0;
        return probabilities;
    }
    std::sort(order.begin(), order.end(), [&](int32_t left, int32_t right) {
        return scores[left] == scores[right] ? left < right : scores[left] < scores[right];
    });
    std::size_t first = 0;
    if (options.top_k > 0 && static_cast<uint64_t>(options.top_k) < order.size()) {
        const float threshold =
            scores[order[order.size() - static_cast<std::size_t>(options.top_k)]];
        // Keep all ties at the kth score, as the provider's strict-less mask does.
        while (scores[order[first]] < threshold)
            ++first;
    }
    double total = 0;
    for (std::size_t i = first; i < order.size(); ++i) {
        probabilities[order[i]] = std::exp(static_cast<double>(scores[order[i]] - scores[maximum]));
        total += probabilities[order[i]];
    }
    if (options.top_p < 1) {
        double cumulative = 0;
        const double remove_mass = 1.0 - static_cast<double>(options.top_p);
        for (std::size_t i = first; i + 1 < order.size(); ++i) {
            cumulative += probabilities[order[i]] / total;
            if (cumulative <= remove_mass)
                probabilities[order[i]] = 0;
        }
    }
    total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    require(std::isfinite(total) && total > 0, "invalid sampling probability mass");
    for (auto& value : probabilities)
        value /= total;
    return probabilities;
}

int32_t draw_token(const std::vector<double>& probabilities, double uniform) {
    require(std::isfinite(uniform) && uniform >= 0 && uniform < 1, "uniform draw outside [0,1)");
    require(!probabilities.empty() && probabilities.size() <= static_cast<std::size_t>(INT32_MAX),
            "invalid sampling vocabulary");
    double total = 0;
    for (double value : probabilities) {
        require(std::isfinite(value) && value >= 0, "invalid sampling probability");
        total += value;
    }
    require(total > 0 && std::isfinite(total), "empty sampling distribution");
    const double target = uniform * total;
    double cumulative = 0;
    int32_t last = -1;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        if (probabilities[i] > 0)
            last = static_cast<int32_t>(i);
        cumulative += probabilities[i];
        if (target < cumulative)
            return static_cast<int32_t>(i);
    }
    return last; // Guard the final rounded cumulative endpoint.
}

ReasoningHistory::ReasoningHistory(std::vector<std::vector<int32_t>> prompt,
                                   const ReasoningOptions& options)
    : tokens_(std::move(prompt)), options_(options), future_found_(tokens_.size(), false),
      finished_(tokens_.size(), false) {
    require(!tokens_.empty() && !tokens_[0].empty() && options.max_new_tokens > 0 &&
                options.future_start_token >= 0 && options.pad_token >= 0,
            "invalid reasoning history parameters");
    for (const auto& row : tokens_)
        require(row.size() == tokens_[0].size() &&
                    std::all_of(row.begin(), row.end(), [](int32_t x) { return x >= 0; }),
                "invalid reasoning prompt rows");
    for (const auto token : options_.eos_tokens)
        require(token >= 0, "invalid EOS token");
}

bool ReasoningHistory::finished(std::size_t row) const {
    return finished_.at(row);
}

bool ReasoningHistory::append(const std::vector<int32_t>& selected) {
    require(!stopped_ && selected.size() == tokens_.size(), "invalid reasoning continuation");
    for (std::size_t row = 0; row < selected.size(); ++row)
        require(selected[row] >= 0 && (!finished_[row] || selected[row] == options_.pad_token),
                "finished reasoning rows require padding");
    const bool after_future =
        std::all_of(future_found_.begin(), future_found_.end(), [](bool value) { return value; });
    for (std::size_t row = 0; row < selected.size(); ++row) {
        tokens_[row].push_back(selected[row]);
        future_found_[row] = future_found_[row] || selected[row] == options_.future_start_token;
        finished_[row] =
            finished_[row] || std::find(options_.eos_tokens.begin(), options_.eos_tokens.end(),
                                        selected[row]) != options_.eos_tokens.end();
    }
    ++generated_;
    stopped_ = after_future || generated_ == options_.max_new_tokens ||
               std::all_of(finished_.begin(), finished_.end(), [](bool value) { return value; });
    return stopped_;
}

std::vector<int64_t> ReasoningHistory::action_offsets() const {
    std::vector<int64_t> result;
    for (const auto& row : tokens_) {
        const auto first = std::find(row.begin(), row.end(), options_.future_start_token);
        result.push_back(first == row.end() ? static_cast<int64_t>(row.size())
                                            : static_cast<int64_t>(first - row.begin()) + 1);
    }
    return result;
}

std::vector<std::vector<int32_t>> ReasoningHistory::padded() const {
    auto result = tokens_;
    const auto offsets = action_offsets();
    for (std::size_t row = 0; row < result.size(); ++row)
        std::fill(result[row].begin() + offsets[row], result[row].end(), options_.pad_token);
    return result;
}

TensorMap TextState::tensors() {
    TensorMap result;
    for (auto& [name, bytes] : storage)
        result.emplace(name, Tensor{bytes.data(), shapes.at(name),
                                    name == "logits" ? DType::kFloat32 : cache_dtype});
    return result;
}

TensorMap TextState::past() {
    TensorMap result;
    for (const auto& [name, tensor] : tensors())
        if (name != "logits")
            result.emplace("past_" + name, tensor);
    return result;
}

TensorMap TextState::action_prefix() {
    TensorMap result;
    for (const auto& [name, tensor] : tensors())
        if (name != "logits")
            result.emplace("prefix_" + name, tensor);
    return result;
}

TextDecoder::TextDecoder(std::unique_ptr<ITrtModule> module) : module_(std::move(module)) {
    require(module_ && module_->ok(), "engine is unavailable");
    require(module_->optimization_profile_count() == 1, "expected one VLM profile");
    const auto outputs = module_->output_info();
    require(outputs.size() >= 3 && (outputs.size() - 1) % 2 == 0, "invalid output set");
    layers_ = static_cast<int64_t>((outputs.size() - 1) / 2);
    const auto input_count = static_cast<int64_t>(module_->input_info().size());
    require(input_count >= 6 + 2 * layers_, "invalid input set");
    deepstack_ = input_count - 6 - 2 * layers_;
    require(deepstack_ <= layers_, "too many DeepStack levels");
    for (const auto* name :
         {"tokens", "visual", "visual_mask", "mask", "cosine", "sine", "past_key_0"})
        require(module_->has_input(name), std::string("missing input ") + name);
    require(module_->has_output("logits") && module_->tensor_dtype("logits") == DType::kFloat32,
            "missing FP32 logits");
    const auto logits = module_->tensor_shape("logits");
    const auto visual = module_->input_profile_shape("visual", 0, ProfileShapeSelector::kMax);
    const auto key = module_->input_profile_shape("past_key_0", 0, ProfileShapeSelector::kMax);
    require(logits.size() == 3 && logits[2] > 0, "invalid vocabulary shape");
    require(visual.size() == 3 && visual[0] > 0 && visual[1] > 0 && visual[2] > 0,
            "invalid visual embedding profile");
    require(key.size() == 4 && key[0] == visual[0] && key[1] > 0 && key[2] > 0 && key[3] > 0,
            "invalid prefix profile");
    vocab_ = logits[2];
    hidden_ = visual[2];
    heads_ = key[1];
    width_ = key[3];
    dtype_ = module_->tensor_dtype("visual");
    require(dtype_ == DType::kFloat16 || dtype_ == DType::kFloat32,
            "unsupported embedding precision");
    for (const auto& [name, dtype] :
         std::vector<std::pair<std::string, DType>>{{"tokens", DType::kInt32},
                                                    {"visual_mask", DType::kInt32},
                                                    {"mask", DType::kFloat32},
                                                    {"cosine", dtype_},
                                                    {"sine", dtype_}})
        require(module_->tensor_dtype(name) == dtype, "invalid dtype for " + name);
    for (int64_t i = 0; i < deepstack_; ++i) {
        const auto name = "deepstack_" + std::to_string(i);
        require(module_->has_input(name) && module_->tensor_dtype(name) == dtype_ &&
                    module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax) == visual,
                "inconsistent DeepStack input " + name);
    }
    for (int64_t i = 0; i < layers_; ++i) {
        for (const auto* kind : {"key_", "value_"}) {
            const auto output = kind + std::to_string(i);
            const auto input = "past_" + output;
            require(module_->has_input(input) && module_->has_output(output) &&
                        module_->tensor_dtype(input) == dtype_ &&
                        module_->tensor_dtype(output) == dtype_ &&
                        module_->input_profile_shape(input, 0, ProfileShapeSelector::kMax) == key,
                    "inconsistent cache pair " + output);
        }
    }
    module_->set_timing_label("alpamayo.text");
}

TensorMap TextPrefill::tensors() {
    auto result = positions.tensors();
    const auto batch = positions.batch, length = positions.length;
    const auto dtype = positions.dtype;
    result.emplace("tokens", Tensor{tokens.data(), {batch, length}, DType::kInt32});
    result.emplace("visual_mask", Tensor{visual_mask.data(), {batch, length, 1}, DType::kInt32});
    result.emplace("visual", Tensor{visual.data(), {batch, length, hidden}, dtype});
    result.emplace("mask", Tensor{mask.data(), {batch, 1, length, length}, DType::kFloat32});
    for (std::size_t i = 0; i < deepstack.size(); ++i)
        result.emplace("deepstack_" + std::to_string(i),
                       Tensor{deepstack[i].data(), {batch, length, hidden}, dtype});
    for (int64_t i = 0; i < layers; ++i)
        for (const auto* kind : {"past_key_", "past_value_"})
            result.emplace(kind + std::to_string(i),
                           Tensor{nullptr, {batch, heads, 0, positions.width}, dtype});
    return result;
}

TextPrefill TextDecoder::prepare(const std::vector<std::vector<int32_t>>& tokens,
                                 const std::vector<std::array<int64_t, 3>>& image_grids,
                                 const std::vector<VisionFeatures>& images,
                                 const std::vector<std::vector<int32_t>>& attention_mask,
                                 const ImagePositionConfig& config) const {
    require(!tokens.empty() && !tokens[0].empty() && config.head_width == width_,
            "invalid prefill prompt or rotary width");
    const auto minimum = module_->input_profile_shape("tokens", 0, ProfileShapeSelector::kMin);
    const auto maximum = module_->input_profile_shape("tokens", 0, ProfileShapeSelector::kMax);
    require(minimum.size() == 2 && maximum.size() == 2 &&
                tokens.size() >= static_cast<uint64_t>(minimum[0]) &&
                tokens.size() <= static_cast<uint64_t>(maximum[0]) &&
                tokens[0].size() >= static_cast<uint64_t>(minimum[1]) &&
                tokens[0].size() <= static_cast<uint64_t>(maximum[1]),
            "prefill exceeds engine capacity");
    require(images.size() == image_grids.size(), "prefill image/grid count mismatch");
    TextPrefill result;
    result.positions = prepare_image_positions(tokens, image_grids, attention_mask, config, dtype_);
    result.hidden = hidden_;
    result.layers = layers_;
    result.heads = heads_;
    const auto batch = result.positions.batch, length = result.positions.length;
    const auto row_bytes = checked_bytes({hidden_}, dtype_);
    // Validate every feature buffer before scattering any data. The already
    // validated image grids bound their products by the prompt length.
    for (std::size_t i = 0; i < images.size(); ++i) {
        const auto& image = images[i];
        const auto& grid = image_grids[i];
        const auto count =
            grid[0] * (grid[1] / config.spatial_merge_size) * (grid[2] / config.spatial_merge_size);
        require(image.tokens == count && image.width == hidden_ && image.dtype == dtype_ &&
                    image.deepstack.size() == static_cast<std::size_t>(deepstack_),
                "prefill visual feature shape, precision, or DeepStack mismatch");
        const auto bytes = checked_bytes({count, hidden_}, dtype_);
        const auto valid = [&](const std::vector<uint8_t>& storage) {
            return storage.size() == bytes &&
                   finite(Tensor{const_cast<uint8_t*>(storage.data()), {count, hidden_}, dtype_});
        };
        require(valid(image.features), "invalid prefill visual feature storage");
        for (const auto& level : image.deepstack)
            require(valid(level), "invalid prefill DeepStack storage");
    }
    const auto cells = checked_bytes({batch, length}, DType::kInt32) / sizeof(int32_t);
    result.tokens.reserve(cells);
    result.visual_mask.assign(cells, 0);
    result.visual.assign(checked_bytes({batch, length, hidden_}, dtype_), 0);
    result.deepstack.assign(static_cast<std::size_t>(deepstack_),
                            std::vector<uint8_t>(result.visual.size(), 0));
    result.mask.resize(checked_bytes({batch, 1, length, length}, DType::kFloat32) / sizeof(float));
    // The pinned eager provider uses the finite minimum of the embedding dtype.
    // Our graph accepts the additive mask in FP32, including for FP16 embeddings.
    const float masked =
        dtype_ == DType::kFloat16 ? -65504.0F : std::numeric_limits<float>::lowest();
    std::size_t image_index = 0;
    int64_t image_offset = 0;
    for (int64_t row = 0; row < batch; ++row) {
        for (int64_t column = 0; column < length; ++column) {
            const auto token = tokens[row][column];
            require(token < vocab_, "prefill token outside vocabulary");
            result.tokens.push_back(token);
            const auto active = attention_mask.empty() || attention_mask[row][column] != 0;
            const auto cell = static_cast<std::size_t>(row * length + column);
            if (active && token == config.image_token) {
                require(image_index < images.size(), "prefill image feature exhaustion");
                const auto& image = images[image_index];
                const auto source = static_cast<std::size_t>(image_offset) * row_bytes;
                const auto destination = cell * row_bytes;
                std::memcpy(result.visual.data() + destination, image.features.data() + source,
                            row_bytes);
                for (int64_t i = 0; i < deepstack_; ++i)
                    std::memcpy(result.deepstack[i].data() + destination,
                                image.deepstack[i].data() + source, row_bytes);
                result.visual_mask[cell] = 1;
                if (++image_offset == image.tokens) {
                    ++image_index;
                    image_offset = 0;
                }
            }
            for (int64_t key = 0; key < length; ++key) {
                const auto visible =
                    key <= column && (attention_mask.empty() || attention_mask[row][key] != 0);
                result.mask[cell * static_cast<std::size_t>(length) + key] =
                    visible ? 0.0F : masked;
            }
        }
    }
    require(image_index == images.size() && image_offset == 0, "unused prefill image features");
    // Reuse the complete engine admission contract without executing or changing
    // profile state. In particular, truly empty initial caches must be accepted.
    for (const auto& [name, tensor] : result.tensors())
        validate(name, tensor, tensor.shape, tensor.dtype);
    return result;
}

void TextDecoder::validate(const std::string& name, const Tensor& tensor,
                           const std::vector<int64_t>& expected, DType dtype) const {
    require(tensor.shape == expected && tensor.dtype == dtype, "invalid shape/dtype for " + name);
    const auto bytes = checked_bytes(expected, dtype);
    require(bytes == 0 || tensor.data != nullptr, "missing data for " + name);
    const auto minimum = module_->input_profile_shape(name, 0, ProfileShapeSelector::kMin);
    const auto maximum = module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax);
    require(minimum.size() == expected.size() && maximum.size() == expected.size(),
            "invalid profile rank for " + name);
    for (std::size_t axis = 0; axis < expected.size(); ++axis)
        require(expected[axis] >= minimum[axis] && expected[axis] <= maximum[axis],
                "shape outside engine capacity for " + name);
    require(finite(tensor, name == "mask"), "non-finite input " + name);
}

TextState TextDecoder::evaluate(const TensorMap& inputs) {
    require(inputs.size() == static_cast<std::size_t>(6 + deepstack_ + 2 * layers_),
            "incomplete input set");
    const auto token = inputs.find("tokens"), key = inputs.find("past_key_0");
    require(token != inputs.end() && token->second.shape.size() == 2, "missing token shape");
    require(key != inputs.end() && key->second.shape.size() == 4, "missing prefix shape");
    const auto batch = token->second.shape[0], length = token->second.shape[1];
    const auto past = key->second.shape[2];
    require(batch > 0 && length > 0 && past >= 0 &&
                past <= std::numeric_limits<int64_t>::max() - length,
            "invalid batch or sequence length");
    const auto total = past + length;
    std::vector<std::pair<std::string, std::vector<int64_t>>> shapes{
        {"tokens", {batch, length}},
        {"visual", {batch, length, hidden_}},
        {"visual_mask", {batch, length, 1}},
        {"mask", {batch, 1, length, total}},
        {"cosine", {batch, 1, length, width_}},
        {"sine", {batch, 1, length, width_}}};
    for (int64_t i = 0; i < deepstack_; ++i)
        shapes.emplace_back("deepstack_" + std::to_string(i),
                            std::vector<int64_t>{batch, length, hidden_});
    for (int64_t i = 0; i < layers_; ++i)
        for (const auto* kind : {"past_key_", "past_value_"})
            shapes.emplace_back(kind + std::to_string(i),
                                std::vector<int64_t>{batch, heads_, past, width_});
    // Validate every tensor and token before touching the reusable execution context.
    for (const auto& [name, shape] : shapes) {
        const auto found = inputs.find(name);
        require(found != inputs.end(), "missing input " + name);
        const auto dtype = name == "tokens" || name == "visual_mask" ? DType::kInt32
                           : name == "mask"                          ? DType::kFloat32
                                                                     : dtype_;
        validate(name, found->second, shape, dtype);
    }
    const auto* ids = static_cast<const int32_t*>(token->second.data);
    const auto& selection = inputs.at("visual_mask");
    const auto* mask = static_cast<const int32_t*>(selection.data);
    for (std::size_t i = 0; i < token->second.numel(); ++i) {
        require(ids[i] >= 0 && ids[i] < vocab_, "token outside vocabulary");
        require(mask[i] == 0 || mask[i] == 1, "visual mask must contain only zero or one");
    }
    const auto outputs = module_->forward(inputs);
    if (outputs.size() != static_cast<std::size_t>(1 + 2 * layers_))
        throw std::runtime_error("Alpamayo text: incomplete output set");
    TextState result;
    result.cache_dtype = dtype_;
    for (const auto& [name, tensor] : outputs) {
        const auto expected = name == "logits" ? std::vector<int64_t>{batch, 1, vocab_}
                                               : std::vector<int64_t>{batch, heads_, total, width_};
        const auto dtype = name == "logits" ? DType::kFloat32 : dtype_;
        if (!tensor.data || tensor.shape != expected || tensor.dtype != dtype || !finite(tensor))
            throw std::runtime_error("Alpamayo text: invalid or non-finite output " + name);
        auto& bytes = result.storage[name];
        bytes.resize(checked_bytes(expected, dtype));
        std::memcpy(bytes.data(), tensor.data, bytes.size());
        result.shapes.emplace(name, tensor.shape);
    }
    return result;
}

ReasoningResult TextDecoder::generate(const TensorMap& prepared,
                                      const std::vector<int64_t>& rope_deltas, float rope_theta,
                                      const ReasoningOptions& options) {
    const auto found = prepared.find("tokens");
    require(found != prepared.end() && found->second.shape.size() == 2 &&
                found->second.dtype == DType::kInt32 && found->second.data,
            "missing reasoning prefill tokens");
    const auto batch = found->second.shape[0], length = found->second.shape[1];
    require(batch > 0 && length > 0 && rope_deltas.size() == static_cast<std::size_t>(batch) &&
                std::isfinite(rope_theta) && rope_theta > 0 && width_ % 2 == 0,
            "invalid reasoning rotary parameters");
    require(options.max_new_tokens > 0 && options.future_start_token >= 0 &&
                options.future_start_token < vocab_ && options.pad_token >= 0 &&
                options.pad_token < vocab_,
            "invalid generation limits or special tokens");
    for (const auto token : options.eos_tokens)
        require(token >= 0 && token < vocab_, "EOS token outside vocabulary");
    const auto prefix_max =
        module_->input_profile_shape("past_key_0", 0, ProfileShapeSelector::kMax)[2];
    require(length <= INT64_MAX - options.max_new_tokens, "generation length overflow");
    require(options.max_new_tokens == 1 || length + options.max_new_tokens - 2 <= prefix_max,
            "generation exceeds cached engine capacity");
    const auto prefix = prepared.find("past_key_0");
    require(prefix != prepared.end() && prefix->second.shape.size() == 4 &&
                prefix->second.shape[2] == 0,
            "reasoning requires an empty initial cache");
    // Validate all sampling parameters before the first forward, even for a
    // request whose logits would otherwise make some invalid fields irrelevant.
    sampling_probabilities(std::vector<float>(static_cast<std::size_t>(vocab_), 0),
                           options.sampling);
    for (const auto delta : rope_deltas)
        require(delta >= -length && delta <= INT64_MAX - length - options.max_new_tokens,
                "reasoning rotary position overflow");
    const auto* prompt = static_cast<const int32_t*>(found->second.data);
    // Profile admission precedes reading user-provided prompt storage.
    validate("tokens", found->second, {batch, length}, DType::kInt32);
    std::vector<std::vector<int32_t>> rows;
    for (int64_t row = 0; row < batch; ++row)
        rows.emplace_back(prompt + row * length, prompt + (row + 1) * length);
    ReasoningHistory history(std::move(rows), options);
    auto state = evaluate(prepared);
    std::mt19937_64 random(options.seed);
    std::vector<int32_t> selected(static_cast<std::size_t>(batch));
    std::vector<int32_t> visual_mask(static_cast<std::size_t>(batch), 0);
    std::vector<uint8_t> zeros(checked_bytes({batch, 1, hidden_}, dtype_), 0);
    std::vector<uint8_t> cosine(checked_bytes({batch, 1, 1, width_}, dtype_)), sine(cosine.size());
    std::vector<float> inverse(static_cast<std::size_t>(width_ / 2));
    for (int64_t i = 0; i < width_ / 2; ++i)
        inverse[i] =
            1.0F / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(width_));
    const auto store = [&](std::vector<uint8_t>& output, std::size_t index, float value) {
        if (dtype_ == DType::kFloat32)
            std::memcpy(output.data() + index * sizeof(float), &value, sizeof(float));
        else {
            const auto half = __float2half_rn(value);
            std::memcpy(output.data() + index * sizeof(half), &half, sizeof(half));
        }
    };
    for (;;) {
        const auto outputs = state.tensors();
        const auto* logits = static_cast<const float*>(outputs.at("logits").data);
        for (int64_t row = 0; row < batch; ++row) {
            if (history.finished(static_cast<std::size_t>(row))) {
                selected[row] = options.pad_token;
                continue;
            }
            const std::vector<float> values(logits + row * vocab_, logits + (row + 1) * vocab_);
            const auto probabilities = sampling_probabilities(values, options.sampling);
            const double uniform = static_cast<double>(random() >> 11) * 0x1.0p-53;
            selected[row] = draw_token(probabilities, uniform);
        }
        if (history.append(selected))
            break;
        const auto past = state.shapes.at("key_0")[2];
        for (int64_t row = 0; row < batch; ++row) {
            const float position = static_cast<float>(past + rope_deltas[row]);
            for (int64_t i = 0; i < width_; ++i) {
                const float angle = position * inverse[i % (width_ / 2)];
                store(cosine, static_cast<std::size_t>(row * width_ + i), std::cos(angle));
                store(sine, static_cast<std::size_t>(row * width_ + i), std::sin(angle));
            }
        }
        std::vector<float> mask(
            checked_bytes({batch, 1, 1, past + 1}, DType::kFloat32) / sizeof(float), 0);
        // Retain the original padding/custom prefix exclusions at every decode
        // step; newly appended positions are visible. An all-zero mask would
        // incorrectly reintroduce the padded prompt's keys after prefill.
        const auto* prefill_mask = static_cast<const float*>(prepared.at("mask").data);
        for (int64_t row = 0; row < batch; ++row)
            std::copy_n(prefill_mask + (row * length + length - 1) * length, length,
                        mask.data() + row * (past + 1));
        auto inputs = state.past();
        inputs.emplace("tokens", Tensor{selected.data(), {batch, 1}, DType::kInt32});
        inputs.emplace("visual", Tensor{zeros.data(), {batch, 1, hidden_}, dtype_});
        inputs.emplace("visual_mask", Tensor{visual_mask.data(), {batch, 1, 1}, DType::kInt32});
        inputs.emplace("mask", Tensor{mask.data(), {batch, 1, 1, past + 1}, DType::kFloat32});
        inputs.emplace("cosine", Tensor{cosine.data(), {batch, 1, 1, width_}, dtype_});
        inputs.emplace("sine", Tensor{sine.data(), {batch, 1, 1, width_}, dtype_});
        for (int64_t i = 0; i < deepstack_; ++i)
            inputs.emplace("deepstack_" + std::to_string(i),
                           Tensor{zeros.data(), {batch, 1, hidden_}, dtype_});
        // evaluate copies its output before the old, externally provided prefix
        // is released; no in-place KV alias or synthetic expert context is used.
        state = evaluate(inputs);
    }
    return {std::move(state), history.padded(), history.action_offsets(), rope_deltas,
            history.generated()};
}

} // namespace trtmc::alpamayo
