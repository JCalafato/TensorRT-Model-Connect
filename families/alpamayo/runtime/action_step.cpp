/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "action_step.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::alpamayo {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("Alpamayo action: " + message);
}

} // namespace

std::vector<float> make_action_prefix_mask(const std::vector<std::vector<float>>& prompt_mask,
                                           const std::vector<int64_t>& offsets, int64_t length,
                                           int64_t waypoints) {
    require(length > 0 && waypoints > 0 && length <= INT64_MAX - waypoints && !offsets.empty() &&
                prompt_mask.size() == offsets.size(),
            "invalid action prompt mask dimensions");
    const auto columns = static_cast<std::size_t>(length + waypoints);
    std::size_t elements = offsets.size();
    for (const auto dimension : {static_cast<std::size_t>(waypoints), columns}) {
        require(elements <= std::numeric_limits<std::size_t>::max() / sizeof(float) / dimension,
                "action prompt mask size overflow");
        elements *= dimension;
    }
    const auto prompt_length = prompt_mask.front().size();
    require(prompt_length > 0 && prompt_length <= static_cast<uint64_t>(length),
            "action prompt mask exceeds cache");
    for (std::size_t row = 0; row < offsets.size(); ++row) {
        require(offsets[row] > 0 && offsets[row] <= length + 1 &&
                    prompt_length <= static_cast<uint64_t>(offsets[row]) &&
                    prompt_mask[row].size() == prompt_length,
                "invalid action prompt mask row or offset");
        require(std::all_of(prompt_mask[row].begin(), prompt_mask[row].end(),
                            [](float value) { return std::isfinite(value) && value <= 0; }),
                "action prompt mask must contain finite nonpositive values");
    }
    std::vector<float> mask(elements, 0);
    for (std::size_t row = 0; row < offsets.size(); ++row) {
        for (int64_t step = 0; step < waypoints; ++step) {
            auto* output =
                mask.data() + (row * static_cast<std::size_t>(waypoints) + step) * columns;
            std::copy(prompt_mask[row].begin(), prompt_mask[row].end(), output);
            for (int64_t position = offsets[row]; position < length; ++position)
                output[position] = std::numeric_limits<float>::lowest();
        }
    }
    return mask;
}

ActionStep::ActionStep(std::unique_ptr<ITrtModule> module) : module_(std::move(module)) {
    require(module_ && module_->ok(), "engine is unavailable");
    require(module_->optimization_profile_count() == 1, "expected one action profile");
    const auto inputs = module_->input_info();
    require(inputs.size() >= 7 && (inputs.size() - 5) % 2 == 0, "invalid input set");
    layers_ = static_cast<int64_t>((inputs.size() - 5) / 2);
    const auto action = module_->input_profile_shape("actions", 0, ProfileShapeSelector::kMax);
    const auto key = module_->input_profile_shape("prefix_key_0", 0, ProfileShapeSelector::kMax);
    require(action.size() == 3 && action[0] > 0 && action[1] > 0 && action[2] == 2,
            "invalid action profile");
    require(key.size() == 4 && key[0] == action[0] && key[1] > 0 && key[2] > 0 && key[3] > 0,
            "invalid prefix profile");
    waypoints_ = action[1];
    heads_ = key[1];
    head_dim_ = key[3];
    for (const auto* name : {"actions", "timesteps", "prefix_mask", "cosine", "sine"}) {
        require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kFloat32,
                std::string("missing FP32 input ") + name);
    }
    const auto kv_dtype = module_->tensor_dtype("prefix_key_0");
    require(kv_dtype == DType::kFloat16 || kv_dtype == DType::kFloat32,
            "prefix must use the engine's FP16 or FP32 precision");
    for (int64_t layer = 0; layer < layers_; ++layer) {
        for (const auto* kind : {"prefix_key_", "prefix_value_"}) {
            const auto name = kind + std::to_string(layer);
            require(module_->has_input(name) && module_->tensor_dtype(name) == kv_dtype,
                    "missing or inconsistent prefix input " + name);
            require(module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax) == key,
                    "inconsistent prefix profile " + name);
        }
    }
    require(module_->output_info().size() == 1 && module_->has_output("velocity") &&
                module_->tensor_dtype("velocity") == DType::kFloat32,
            "expected one FP32 velocity output");
    module_->set_timing_label("alpamayo.action");
}

void ActionStep::validate(const std::string& name, const Tensor& tensor,
                          const std::vector<int64_t>& expected) const {
    require(module_ && module_->has_input(name), "unknown input " + name);
    require(tensor.data != nullptr && tensor.shape == expected, "invalid shape/data for " + name);
    require(tensor.dtype == module_->tensor_dtype(name), "invalid dtype for " + name);
    const auto minimum = module_->input_profile_shape(name, 0, ProfileShapeSelector::kMin);
    const auto maximum = module_->input_profile_shape(name, 0, ProfileShapeSelector::kMax);
    require(minimum.size() == expected.size() && maximum.size() == expected.size(),
            "invalid profile rank for " + name);
    auto bytes = dtype_size(tensor.dtype);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(expected[i] > 0 && expected[i] >= minimum[i] && expected[i] <= maximum[i],
                "shape outside engine capacity for " + name);
        require(static_cast<uint64_t>(expected[i]) <=
                    std::numeric_limits<std::size_t>::max() / bytes,
                "tensor size overflow for " + name);
        bytes *= static_cast<std::size_t>(expected[i]);
    }
}

void ActionStep::set_context(const TensorMap& inputs) {
    require(module_ != nullptr, "engine is unavailable after a binding failure");
    require(inputs.size() == static_cast<std::size_t>(3 + 2 * layers_), "incomplete context");
    const auto first = inputs.find("prefix_key_0");
    require(first != inputs.end() && first->second.shape.size() == 4, "missing prefix key shape");
    const auto batch = first->second.shape[0];
    const auto prefix = first->second.shape[2];
    require(prefix > 0 && prefix <= std::numeric_limits<int64_t>::max() - waypoints_,
            "invalid prefix length");
    const std::vector<int64_t> kv{batch, heads_, prefix, head_dim_};
    for (const auto& [name, input] : inputs) {
        if (name == "prefix_mask")
            validate(name, input, {batch, 1, waypoints_, prefix + waypoints_});
        else if (name == "cosine" || name == "sine")
            validate(name, input, {batch, 1, waypoints_, head_dim_});
        else {
            require(name.rfind("prefix_key_", 0) == 0 || name.rfind("prefix_value_", 0) == 0,
                    "unexpected context input " + name);
            validate(name, input, kv);
        }
    }

    std::unordered_map<std::string, DeviceTensor> replacement;
    for (const auto& [name, input] : inputs) {
        DeviceTensor value(input.shape, input.dtype, module_->stream());
        if (!value.ok() || !value.copy_from_host(input.data))
            throw std::runtime_error("Alpamayo action: context upload failed for " + name);
        replacement.emplace(name, std::move(value));
    }
    module_->sync();
    ready_ = false;
    // Keep old external buffers alive until every binding has been replaced.
    auto previous = std::move(context_);
    context_ = std::move(replacement);
    try {
        for (auto& [name, value] : context_)
            module_->bind_external(name, value.data(), value.shape());
    } catch (...) {
        module_.reset();
        throw;
    }
    batch_ = batch;
    ready_ = true;
}

void ActionStep::set_generated_context(const TensorMap& prefix, const std::vector<int64_t>& offsets,
                                       const std::vector<int64_t>& rope_deltas,
                                       const std::vector<std::vector<float>>& prompt_mask,
                                       float rope_theta) {
    require(module_ && prefix.size() == static_cast<std::size_t>(2 * layers_),
            "incomplete generated prefix");
    const auto found = prefix.find("prefix_key_0");
    require(found != prefix.end() && found->second.shape.size() == 4,
            "missing generated prefix key");
    const auto batch = found->second.shape[0], length = found->second.shape[2];
    require(batch > 0 && length > 0 && length <= std::numeric_limits<int64_t>::max() - waypoints_ &&
                offsets.size() == static_cast<std::size_t>(batch) &&
                rope_deltas.size() == static_cast<std::size_t>(batch) && head_dim_ % 2 == 0 &&
                std::isfinite(rope_theta) && rope_theta > 0,
            "invalid generated prefix positions");
    const std::vector<int64_t> kv{batch, heads_, length, head_dim_};
    // Admit every cache and offset before allocating or replacing the previous
    // device context. A final uncached sample may place the fallback offset
    // exactly one beyond the cache, matching the provider's no-marker case.
    for (const auto& [name, value] : prefix) {
        require(name.rfind("prefix_key_", 0) == 0 || name.rfind("prefix_value_", 0) == 0,
                "unexpected generated prefix input " + name);
        validate(name, value, kv);
    }
    for (int64_t row = 0; row < batch; ++row) {
        require(offsets[row] > 0 && offsets[row] <= length + 1,
                "generated action offset outside sequence");
        require(rope_deltas[row] >= -offsets[row] &&
                    rope_deltas[row] <=
                        std::numeric_limits<int64_t>::max() - offsets[row] - waypoints_,
                "generated action rotary position overflow");
    }
    const auto elements = [](const std::vector<int64_t>& shape) {
        std::size_t size = 1;
        for (const auto value : shape) {
            require(value > 0 && static_cast<uint64_t>(value) <=
                                     std::numeric_limits<std::size_t>::max() / sizeof(float) / size,
                    "generated context size overflow");
            size *= static_cast<std::size_t>(value);
        }
        return size;
    };
    const std::vector<int64_t> rotary_shape{batch, 1, waypoints_, head_dim_};
    const std::vector<int64_t> mask_shape{batch, 1, waypoints_, length + waypoints_};
    auto mask = make_action_prefix_mask(prompt_mask, offsets, length, waypoints_);
    std::vector<float> cosine(elements(rotary_shape)), sine(cosine.size());
    std::vector<float> inverse(static_cast<std::size_t>(head_dim_ / 2));
    for (int64_t i = 0; i < head_dim_ / 2; ++i)
        inverse[i] =
            1.0F / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim_));
    for (int64_t row = 0; row < batch; ++row) {
        for (int64_t step = 0; step < waypoints_; ++step) {
            const float position = static_cast<float>(offsets[row] + rope_deltas[row] + step);
            for (int64_t i = 0; i < head_dim_; ++i) {
                const auto index =
                    static_cast<std::size_t>((row * waypoints_ + step) * head_dim_ + i);
                const float angle = position * inverse[i % (head_dim_ / 2)];
                cosine[index] = std::cos(angle);
                sine[index] = std::sin(angle);
            }
        }
    }
    auto inputs = prefix;
    inputs.emplace("cosine", Tensor{cosine.data(), rotary_shape, DType::kFloat32});
    inputs.emplace("sine", Tensor{sine.data(), rotary_shape, DType::kFloat32});
    inputs.emplace("prefix_mask", Tensor{mask.data(), mask_shape, DType::kFloat32});
    set_context(inputs);
}

std::vector<float> ActionStep::evaluate(const Tensor& actions, const Tensor& timesteps) {
    require(ready_ && module_, "context must be set before evaluating velocity");
    validate("actions", actions, {batch_, waypoints_, 2});
    validate("timesteps", timesteps, {batch_, 1, 1});
    const auto outputs = module_->forward({{"actions", actions}, {"timesteps", timesteps}});
    const auto found = outputs.find("velocity");
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.shape != actions.shape)
        throw std::runtime_error("Alpamayo action: invalid velocity output");
    std::vector<float> velocity(actions.numel());
    std::memcpy(velocity.data(), found->second.data, velocity.size() * sizeof(float));
    return velocity;
}

std::vector<float> ActionStep::integrate(const Tensor& initial_noise, int steps) {
    require(ready_ && module_, "context must be set before integrating actions");
    require(steps > 0, "Euler flow requires a positive step count");
    validate("actions", initial_noise, {batch_, waypoints_, 2});
    const auto* data = static_cast<const float*>(initial_noise.data);
    std::vector<float> actions(data, data + initial_noise.numel());
    require(std::all_of(actions.begin(), actions.end(), [](float x) { return std::isfinite(x); }),
            "initial action noise must be finite");
    std::vector<float> times(static_cast<std::size_t>(batch_));
    const Tensor action{actions.data(), initial_noise.shape, DType::kFloat32};
    const Tensor time{times.data(), {batch_, 1, 1}, DType::kFloat32};
    // Preserve FP32 linspace endpoints and differences, rather than assuming
    // every rounded interval is equal to the nominal reciprocal step count.
    const auto point = [steps](int index) {
        return static_cast<float>(static_cast<double>(index) / steps);
    };
    for (int step = 0; step < steps; ++step) {
        const float start = point(step);
        const float delta = point(step + 1) - start;
        std::fill(times.begin(), times.end(), start);
        const auto velocity = evaluate(action, time);
        for (std::size_t i = 0; i < actions.size(); ++i) {
            actions[i] += delta * velocity[i];
            if (!std::isfinite(actions[i]))
                throw std::runtime_error("Alpamayo action: non-finite Euler flow output");
        }
    }
    return actions;
}

} // namespace trtmc::alpamayo
