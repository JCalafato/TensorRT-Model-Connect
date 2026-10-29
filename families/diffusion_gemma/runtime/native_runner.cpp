/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace trtmc::diffusion_gemma {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("DiffusionGemma CUDA transfer: ") +
                                 cudaGetErrorString(status));
}
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::string cache_name(const char* kind, std::size_t layer) {
    return std::string(kind) + "_" + std::to_string(layer);
}
} // namespace

NativeRunner::NativeRunner(std::unique_ptr<ITrtModule> engine, NativeConfig config)
    : engine_(std::move(engine)), config_(std::move(config)) {
    if (!engine_ || !engine_->ok())
        throw std::invalid_argument("DiffusionGemma requires a valid native engine");
    // Validate the checkpoint sampler before allocating request state.
    Sampler validate(config_.sampling, 0);
    if (config_.hidden_size <= 0 || config_.max_sequence_length < config_.sampling.canvas_length ||
        config_.cache_widths.empty())
        throw std::invalid_argument("DiffusionGemma invalid native bundle geometry");
    for (auto eos : config_.eos_token_ids)
        if (eos < 0 || eos >= config_.sampling.vocabulary_size)
            throw std::invalid_argument("DiffusionGemma invalid bundle EOS token");
    for (const auto* name : {"token_id", "position_id", "cache_length", "decoder_phase"})
        if (!engine_->has_input(name) || engine_->tensor_dtype(name) != DType::kInt32)
            throw std::invalid_argument("DiffusionGemma invalid integer input contract");
    for (const auto* name : {"previous_soft_embeddings", "temperature", "logits_indices"})
        if (!engine_->has_input(name))
            throw std::invalid_argument("DiffusionGemma missing native input");
    dtype_ = engine_->tensor_dtype("previous_soft_embeddings");
    if (dtype_ != DType::kFloat32 && dtype_ != DType::kFloat16 && dtype_ != DType::kBFloat16)
        throw std::invalid_argument("DiffusionGemma invalid native compute dtype");
    if (!engine_->has_output("logits") || !engine_->has_output("next_soft_embeddings") ||
        engine_->tensor_dtype("logits") != DType::kFloat32 ||
        engine_->tensor_dtype("next_soft_embeddings") != dtype_)
        throw std::invalid_argument("DiffusionGemma invalid native output contract");
    for (std::size_t layer = 0; layer < config_.cache_widths.size(); ++layer) {
        const auto width = config_.cache_widths[layer];
        if (width <= 0)
            throw std::invalid_argument("DiffusionGemma invalid cache width");
        for (const auto* kind : {"k", "v"}) {
            const auto input = cache_name((std::string("cache_") + kind).c_str(), layer);
            const auto output = cache_name((std::string("present_") + kind).c_str(), layer);
            const std::vector<std::int64_t> shape{config_.max_sequence_length, width};
            if (!engine_->has_input(input) || !engine_->has_output(output) ||
                engine_->tensor_shape(input) != shape || engine_->tensor_dtype(input) != dtype_ ||
                engine_->tensor_dtype(output) != dtype_)
                throw std::invalid_argument("DiffusionGemma engine/cache metadata mismatch");
        }
    }
    const std::vector<std::int64_t> expected_tokens{config_.max_sequence_length};
    const std::vector<std::int64_t> expected_soft{config_.max_sequence_length, config_.hidden_size};
    if (engine_->input_profile_shape("token_id", 0, ProfileShapeSelector::kMax) !=
            expected_tokens ||
        engine_->input_profile_shape("previous_soft_embeddings", 0, ProfileShapeSelector::kMax) !=
            expected_soft ||
        engine_->tensor_dtype("temperature") != DType::kFloat32 ||
        engine_->tensor_dtype("logits_indices") != DType::kInt32)
        throw std::invalid_argument("DiffusionGemma engine profile/metadata mismatch");
    conditioning_ = DeviceTensor(expected_soft, dtype_, engine_->stream());
    if (!conditioning_.ok())
        throw std::runtime_error("DiffusionGemma conditioning buffer allocation failed");
    engine_->bind_external("previous_soft_embeddings", conditioning_.data());
    logits_.resize(static_cast<std::size_t>(config_.sampling.canvas_length) *
                   config_.sampling.vocabulary_size);
}

void* NativeRunner::pointer(const std::string& name) const {
    auto* value = engine_->device_ptr(name);
    if (!value)
        throw std::runtime_error("DiffusionGemma missing device buffer: " + name);
    return value;
}

SamplingSettings NativeRunner::settings_for(const TextGenerationConfig& request) const {
    auto settings = config_.sampling;
    if (request.num_steps != -1) {
        if (request.num_steps <= 0)
            throw std::invalid_argument("DiffusionGemma num_steps must be positive");
        settings.max_denoising_steps = request.num_steps;
    }
    if (request.block_length != 0 && request.block_length != settings.canvas_length)
        throw std::invalid_argument("DiffusionGemma block length must match the checkpoint canvas");
    if (request.confidence_threshold != -1.0F) {
        if (!std::isfinite(request.confidence_threshold) || request.confidence_threshold < 0)
            throw std::invalid_argument("DiffusionGemma confidence threshold must be nonnegative");
        settings.confidence_threshold = request.confidence_threshold;
    }
    return settings;
}

void NativeRunner::validate_request(const std::vector<std::int32_t>& prompt,
                                    const TextGenerationConfig& request) const {
    Sampler validate(settings_for(request), 0);
    if (prompt.empty() || request.max_new_tokens < 0 || request.seed < -1)
        throw std::invalid_argument(
            "DiffusionGemma requires a prompt and nonnegative output budget");
    for (auto token : prompt)
        if (token < 0 || token >= config_.sampling.vocabulary_size)
            throw std::invalid_argument("DiffusionGemma prompt token is outside the vocabulary");
    if (request.eos_token_id < -1 || request.eos_token_id >= config_.sampling.vocabulary_size)
        throw std::invalid_argument("DiffusionGemma requested EOS token is outside the vocabulary");
    if (request.temperature != 1.0F || request.top_k != 1 || request.top_p != 1.0F ||
        request.min_p != 0.0F || request.repetition_penalty != 1.0F ||
        request.guidance_scale != -1.0F || request.cfg_scale != -1.0F ||
        request.sde_gamma != -1.0F || !request.initial_latents.empty() ||
        !request.condition_latents.empty() || !request.condition_mask.empty() ||
        !request.sampling_steps.empty() || !request.sde_noises.empty() ||
        request.source_language_token_id != -1 || request.forced_bos_token_id != -1 ||
        !request.lora_adapter_id.empty() || request.stop_on_boxed_answer ||
        (request.text_generation_mode != "auto" && request.text_generation_mode != "diffusion"))
        throw std::invalid_argument(
            "DiffusionGemma uses checkpoint block-diffusion sampling, not AR controls");
    const auto canvas = static_cast<std::uint64_t>(config_.sampling.canvas_length);
    const auto canvases =
        (static_cast<std::uint64_t>(request.max_new_tokens) + canvas - 1) / canvas;
    if (prompt.size() > static_cast<std::size_t>(config_.max_sequence_length) ||
        canvases * canvas > static_cast<std::uint64_t>(config_.max_sequence_length) - prompt.size())
        throw std::invalid_argument(
            "DiffusionGemma request including its final whole canvas exceeds capacity");
}

void NativeRunner::reset() {
    engine_->reset_execution_context();
    prefix_length_ = 0;
    for (std::size_t layer = 0; layer < config_.cache_widths.size(); ++layer) {
        const auto bytes = static_cast<std::size_t>(config_.max_sequence_length) *
                           config_.cache_widths[layer] * dtype_size(dtype_);
        for (const auto* kind : {"cache_k", "cache_v"})
            checked(cudaMemsetAsync(pointer(cache_name(kind, layer)), 0, bytes, engine_->stream()));
    }
}

void NativeRunner::clear_conditioning(std::size_t length) {
    checked(cudaMemsetAsync(pointer("previous_soft_embeddings"), 0,
                            length * config_.hidden_size * dtype_size(dtype_), engine_->stream()));
}

void NativeRunner::execute(const std::vector<std::int32_t>& tokens, bool decoder, float temperature,
                           const ConditionedPrompt& media) {
    const auto length = static_cast<std::int64_t>(tokens.size());
    std::vector<std::int32_t> positions(tokens.size());
    std::iota(positions.begin(), positions.end(), prefix_length_);
    std::vector<std::int32_t> indices(decoder ? tokens.size() : 1);
    if (decoder)
        std::iota(indices.begin(), indices.end(), 0);
    else
        indices[0] = static_cast<std::int32_t>(tokens.size() - 1);
    std::int32_t phase = decoder ? 1 : 0;
    TensorMap inputs;
    inputs["token_id"] = {const_cast<std::int32_t*>(tokens.data()), {length}, DType::kInt32};
    inputs["position_id"] = {positions.data(), {length}, DType::kInt32};
    inputs["cache_length"] = {&prefix_length_, {1}, DType::kInt32};
    inputs["decoder_phase"] = {&phase, {1}, DType::kInt32};
    inputs["temperature"] = {&temperature, {1, 1}, DType::kFloat32};
    inputs["logits_indices"] = {
        indices.data(), {static_cast<std::int64_t>(indices.size())}, DType::kInt32};
    // Null data updates only the dynamic shape. The previous step's soft
    // embedding stays device-resident; no vocabulary-sized host round trip.
    inputs["previous_soft_embeddings"] = {nullptr, {length, config_.hidden_size}, dtype_};
    std::vector<std::int32_t> no_media;
    if (engine_->has_input("media_blocks")) {
        if (media.blocks.empty())
            no_media.assign(tokens.size(), 0);
        const auto& blocks = media.blocks.empty() ? no_media : media.blocks;
        inputs["media_blocks"] = {
            const_cast<std::int32_t*>(blocks.data()), {length}, DType::kInt32};
        // Feature storage persists on GPU. With all block IDs zero its values
        // are ignored; do not repeatedly upload an all-zero feature matrix.
        inputs["media_features"] = {
            media.features.empty() ? nullptr : const_cast<float*>(media.features.data()),
            {length, config_.hidden_size},
            DType::kFloat32};
    }
    engine_->forward_async(inputs);
    // Keep all host input buffers alive until their asynchronous uploads finish.
    engine_->sync();
}

void NativeRunner::encode(const std::vector<std::int32_t>& tokens, const ConditionedPrompt& media) {
    clear_conditioning(tokens.size());
    execute(tokens, false, config_.sampling.t_max, media);
    for (std::size_t layer = 0; layer < config_.cache_widths.size(); ++layer) {
        const auto row_bytes =
            static_cast<std::size_t>(config_.cache_widths[layer]) * dtype_size(dtype_);
        for (const auto* kind : {"k", "v"}) {
            auto* destination = static_cast<char*>(
                pointer(cache_name((std::string("cache_") + kind).c_str(), layer)));
            auto* source = pointer(cache_name((std::string("present_") + kind).c_str(), layer));
            checked(cudaMemcpyAsync(
                destination + static_cast<std::size_t>(prefix_length_) * row_bytes, source,
                tokens.size() * row_bytes, cudaMemcpyDeviceToDevice, engine_->stream()));
        }
    }
    checked(cudaStreamSynchronize(engine_->stream()));
    prefix_length_ += static_cast<std::int32_t>(tokens.size());
}

void NativeRunner::copy_conditioning(std::size_t length) {
    checked(cudaMemcpyAsync(pointer("previous_soft_embeddings"), pointer("next_soft_embeddings"),
                            length * config_.hidden_size * dtype_size(dtype_),
                            cudaMemcpyDeviceToDevice, engine_->stream()));
}

TextResult NativeRunner::generate_ids(const std::vector<std::int32_t>& prompt,
                                      const TextGenerationConfig& request,
                                      const ConditionedPrompt& media) {
    validate_request(prompt, request);
    if (!media.blocks.empty() || !media.features.empty()) {
        if (!engine_->has_input("media_blocks") || !engine_->has_input("media_features") ||
            engine_->tensor_dtype("media_features") != DType::kFloat32 ||
            media.blocks.size() != prompt.size() ||
            media.features.size() !=
                prompt.size() * static_cast<std::size_t>(config_.hidden_size) ||
            std::any_of(media.blocks.begin(), media.blocks.end(),
                        [](auto block) { return block < 0; }) ||
            !std::all_of(media.features.begin(), media.features.end(),
                         [](float value) { return std::isfinite(value); }))
            throw std::invalid_argument("DiffusionGemma invalid image-conditioned prompt");
    }
    const auto settings = settings_for(request);
    const auto seed = request.seed < 0 ? static_cast<std::uint64_t>(std::random_device{}())
                                       : static_cast<std::uint64_t>(request.seed);
    Sampler sampler(settings, seed);
    TextResult result;
    if (request.max_new_tokens == 0)
        return result;
    auto start = Clock::now();
    reset();
    encode(prompt, media);
    result.prefill_ms = elapsed(start);
    start = Clock::now();
    while (result.token_ids.size() < static_cast<std::size_t>(request.max_new_tokens)) {
        auto canvas = sampler.initialize_canvas();
        clear_conditioning(canvas.size());
        DenoisingStep step;
        for (auto remaining = settings.max_denoising_steps; remaining > 0; --remaining) {
            execute(canvas, true, sampler.temperature(remaining));
            checked(cudaMemcpyAsync(logits_.data(), pointer("logits"),
                                    logits_.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                    engine_->stream()));
            checked(cudaStreamSynchronize(engine_->stream()));
            step = sampler.step(logits_.data(), logits_.size(), remaining);
            if (step.stable_and_confident || remaining == 1)
                break;
            canvas = step.next_canvas;
            copy_conditioning(canvas.size());
        }
        bool finished = false;
        for (auto token : step.argmax_canvas) {
            result.token_ids.push_back(token);
            const bool eos =
                request.eos_token_id >= 0
                    ? token == request.eos_token_id
                    : std::find(config_.eos_token_ids.begin(), config_.eos_token_ids.end(),
                                token) != config_.eos_token_ids.end();
            if (eos ||
                result.token_ids.size() == static_cast<std::size_t>(request.max_new_tokens)) {
                finished = true;
                break;
            }
        }
        if (finished)
            break;
        // Commit the final argmax canvas through the encoder, NEVER the noisy
        // candidate or denoiser K/V; encoder/decoder layer scalars differ.
        encode(step.argmax_canvas);
    }
    result.decode_ms = elapsed(start);
    return result;
}

} // namespace trtmc::diffusion_gemma
