/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Family-owned component driver. It deliberately does not claim the public
// trajectory Task contract or replace the independent quality reference.
#include "action_step.h"

#include "driving_pipeline.h"
#include "history_tokenizer.h"
#include "text_decoder.h"
#include "trajectory.h"
#include "trtmc/runtime/trt_backend.h"
#include "vision_encoder.h"
#include "vision_preprocessor.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using nlohmann::json;
using namespace trtmc;

std::vector<char> read_bytes(const std::string& path, bool allow_empty = false) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open " + path);
    const auto size = file.tellg();
    if (size < 0 || (size == 0 && !allow_empty))
        throw std::runtime_error("Empty or unreadable file " + path);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Cannot read " + path);
    return bytes;
}

struct HostInputs {
    std::unordered_map<std::string, std::vector<char>> storage;
    TensorMap tensors;

    explicit HostInputs(const json& descriptions) {
        for (const auto& [name, description] : descriptions.items()) {
            const auto dtype = description.at("dtype").get<std::string>();
            if (dtype != "fp32" && dtype != "fp16" && dtype != "int32")
                throw std::invalid_argument("Unsupported test input dtype " + dtype);
            const auto path = description.at("path").get<std::string>();
            auto [file, inserted] = storage.try_emplace(path);
            if (inserted)
                file->second = read_bytes(path, true);
            auto& bytes = file->second;
            Tensor value{bytes.data(), description.at("shape").get<std::vector<int64_t>>(),
                         dtype == "fp32"   ? DType::kFloat32
                         : dtype == "fp16" ? DType::kFloat16
                                           : DType::kInt32};
            std::size_t count = dtype_size(value.dtype);
            for (const auto dimension : value.shape) {
                if (dimension < 0 ||
                    (count != 0 && static_cast<uint64_t>(dimension) >
                                       std::numeric_limits<std::size_t>::max() / count))
                    throw std::invalid_argument("Invalid tensor shape " + name);
                count *= static_cast<std::size_t>(dimension);
            }
            const auto offset = description.value("offset", std::size_t{0});
            if (value.shape.empty() || offset > bytes.size() || count > bytes.size() - offset ||
                (!description.contains("offset") && count != bytes.size()) ||
                offset % dtype_size(value.dtype) != 0)
                throw std::invalid_argument("Wrong tensor byte count/offset " + name);
            if (!bytes.empty())
                value.data = bytes.data() + offset;
            tensors.emplace(name, std::move(value));
        }
    }
};

trtmc::alpamayo::VisionPreprocessor preprocessor(const json& p) {
    return trtmc::alpamayo::VisionPreprocessor(
        {p.at("patch_size"), p.at("temporal_patch_size"), p.at("spatial_merge_size"),
         p.at("num_position_embeddings"), p.at("head_width"), p.at("min_pixels"),
         p.at("max_pixels"), p.at("patch_capacity"), p.at("image_mean").get<std::array<float, 3>>(),
         p.at("image_std").get<std::array<float, 3>>(), p.at("rescale_factor")});
}

trtmc::alpamayo::PreparedVisionInput
prepare_image(const trtmc::alpamayo::VisionPreprocessor& processor, const json& request) {
    auto raw = read_bytes(request.at("image"));
    return processor.prepare(
        request.value("null_data", false) ? nullptr : reinterpret_cast<const uint8_t*>(raw.data()),
        raw.size(), request.at("height"), request.at("width"));
}

void write_tensors(const TensorMap& tensors, const json& destinations) {
    std::unordered_map<std::string, std::ofstream> files;
    for (const auto& [name, tensor] : tensors) {
        const auto& destination = destinations.at(name);
        const auto path = destination.is_string() ? destination.get<std::string>()
                                                  : destination.at("path").get<std::string>();
        const auto offset = destination.is_string()
                                ? std::streamoff{0}
                                : destination.at("offset").get<std::streamoff>();
        if (offset < 0)
            throw std::invalid_argument("Negative tensor output offset");
        auto [file, inserted] = files.try_emplace(path);
        if (inserted)
            file->second.open(path, std::ios::binary);
        auto& output = file->second;
        output.seekp(offset);
        if (!output.write(static_cast<const char*>(tensor.data),
                          static_cast<std::streamsize>(tensor.nbytes())))
            throw std::runtime_error("Cannot write tensor " + name);
    }
}

int prepare_images(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    auto processor = preprocessor(manifest.at("preprocessing"));
    std::vector<std::pair<trtmc::alpamayo::PreparedVisionInput, json>> retained;
    for (const auto& request : manifest.at("requests")) {
        const bool expected = request.value("expect_error", false);
        bool rejected = false;
        try {
            // Alternate metadata exercises resize geometry and constructor admission.
            auto prepared = request.contains("preprocessing")
                                ? prepare_image(preprocessor(request.at("preprocessing")), request)
                                : prepare_image(processor, request);
            if (!expected)
                retained.emplace_back(std::move(prepared), request.at("outputs"));
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            std::cout << "Expected preprocessing rejection: " << error.what() << '\n';
        }
        if (expected != rejected)
            throw std::runtime_error("Preprocessing rejection expectation was not met");
    }
    for (auto& [prepared, outputs] : retained)
        write_tensors(prepared.tensors(), outputs);
    return 0;
}

int encode_vision(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    auto plan = read_bytes(manifest.at("engine"));
    const std::unique_ptr<IBackend, decltype(&trtmc_destroy_backend)> backend(
        trtmc_create_backend(), trtmc_destroy_backend);
    if (!backend)
        throw std::runtime_error("Model Connect backend creation failed");
    trtmc::alpamayo::VisionEncoder encoder(
        backend->create_module(plan.data(), plan.size(), {}), manifest.at("spatial_merge_size"),
        manifest.at("num_position_embeddings"), manifest.at("deepstack_count"));
    std::vector<char>().swap(plan);
    std::vector<std::pair<trtmc::alpamayo::VisionFeatures, json>> retained;
    for (const auto& request : manifest.at("requests")) {
        const bool expected = request.value("expect_error", false);
        bool rejected = false;
        try {
            auto features = [&]() {
                if (request.contains("image")) {
                    auto prepared =
                        prepare_image(preprocessor(manifest.at("preprocessing")), request);
                    return encoder.encode(prepared.tensors());
                }
                HostInputs inputs(request.at("inputs"));
                return encoder.encode(inputs.tensors);
            }();
            if (!expected)
                retained.emplace_back(std::move(features), request.at("outputs"));
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            std::cout << "Expected visual input rejection: " << error.what() << '\n';
        }
        if (expected != rejected)
            throw std::runtime_error("Visual input rejection expectation was not met");
    }
    // Deliberately retain outputs until after every later frame/reset, so this
    // driver also verifies the stage does not return overwritten module views.
    for (auto& [features, destinations] : retained) {
        for (const auto& [name, tensor] : features.tensors()) {
            std::ofstream file(destinations.at(name).get<std::string>(), std::ios::binary);
            if (!file.write(static_cast<const char*>(tensor.data),
                            static_cast<std::streamsize>(tensor.nbytes())))
                throw std::runtime_error("Cannot write visual output " + name);
        }
        std::cout << "Wrote " << features.tokens << " visual tokens and "
                  << features.deepstack.size() << " DeepStack outputs\n";
    }
    return 0;
}

trtmc::alpamayo::SamplingOptions sampling_options(const json& item) {
    trtmc::alpamayo::SamplingOptions options;
    options.temperature = item.value("temperature", options.temperature);
    options.top_p = item.value("top_p", options.top_p);
    options.top_k = item.value("top_k", options.top_k);
    options.suppressed_begin = item.value("suppressed_begin", options.suppressed_begin);
    options.suppressed_end = item.value("suppressed_end", options.suppressed_end);
    return options;
}

trtmc::alpamayo::ReasoningOptions reasoning_options(const json& item) {
    trtmc::alpamayo::ReasoningOptions options;
    options.sampling = sampling_options(item.value("sampling", json::object()));
    options.max_new_tokens = item.value("max_new_tokens", options.max_new_tokens);
    options.future_start_token = item.at("future_start_token");
    options.pad_token = item.at("pad_token");
    options.eos_tokens = item.value("eos_tokens", std::vector<int32_t>{});
    options.seed = item.value("seed", options.seed);
    return options;
}

int image_position_cases(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    json results = json::object();
    for (const auto& item : manifest.at("cases")) {
        const auto& p = item.at("config");
        const bool expected = item.value("expect_error", false);
        bool rejected = false;
        const auto name = item.at("name").get<std::string>();
        try {
            const trtmc::alpamayo::ImagePositionConfig config{
                p.at("image_token"),
                p.at("video_token"),
                p.at("vision_start_token"),
                p.at("vision_end_token"),
                p.at("spatial_merge_size"),
                p.at("head_width"),
                p.at("mrope_section").get<std::array<int64_t, 3>>(),
                p.at("rope_theta")};
            auto positions = trtmc::alpamayo::prepare_image_positions(
                item.at("tokens").get<std::vector<std::vector<int32_t>>>(),
                item.value("grids", std::vector<std::array<int64_t, 3>>{}),
                item.value("attention_mask", std::vector<std::vector<int32_t>>{}), config,
                item.value("precision", "fp32") == "fp16" ? DType::kFloat16 : DType::kFloat32);
            results[name] = {{"positions", positions.axes},
                             {"deltas", positions.deltas},
                             {"shape", {3, positions.batch, positions.length}}};
            write_tensors(positions.tensors(), item.at("outputs"));
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            results[name] = {{"rejected", true}, {"error", error.what()}};
        }
        if (expected != rejected)
            throw std::runtime_error("Image-position rejection expectation was not met: " + name);
    }
    std::ofstream output(manifest.at("output").get<std::string>());
    output << results.dump();
    if (!output)
        throw std::runtime_error("Cannot write image positions");
    return 0;
}

int reasoning_cases(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    json result = json::object();
    for (const auto& item : manifest.value("sampling", json::array())) {
        const bool expected = item.value("expect_error", false);
        bool rejected = false;
        const auto name = item.at("name").get<std::string>();
        try {
            auto logits = item.at("logits").get<std::vector<float>>();
            for (const auto& index : item.value("negative_infinity", json::array()))
                logits.at(index.get<std::size_t>()) = -std::numeric_limits<float>::infinity();
            if (item.contains("nan_logit"))
                logits.at(item.at("nan_logit").get<std::size_t>()) =
                    std::numeric_limits<float>::quiet_NaN();
            const auto probabilities = trtmc::alpamayo::sampling_probabilities(
                logits, sampling_options(item.value("options", json::object())));
            std::vector<int32_t> draws;
            for (const auto uniform : item.value("draws", std::vector<double>{}))
                draws.push_back(trtmc::alpamayo::draw_token(probabilities, uniform));
            result[name] = {{"probabilities", probabilities}, {"draws", draws}};
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            result[name] = {{"rejected", true}, {"error", error.what()}};
        }
        if (rejected != expected)
            throw std::runtime_error("Sampler rejection expectation was not met: " + name);
    }
    for (const auto& item : manifest.value("histories", json::array())) {
        const auto name = item.at("name").get<std::string>();
        const bool expected = item.value("expect_error", false);
        bool rejected = false;
        try {
            trtmc::alpamayo::ReasoningHistory history(
                item.at("prompt").get<std::vector<std::vector<int32_t>>>(),
                reasoning_options(item.at("options")));
            result[name] = json::array();
            for (const auto& step : item.value("steps", json::array())) {
                const bool expected_step = step.value("expect_error", false);
                bool rejected_step = false;
                try {
                    const bool stopped =
                        history.append(step.at("tokens").get<std::vector<int32_t>>());
                    result[name].push_back({{"stopped", stopped},
                                            {"generated", history.generated()},
                                            {"tokens", history.padded()},
                                            {"action_offsets", history.action_offsets()}});
                } catch (const std::invalid_argument& error) {
                    if (!expected_step)
                        throw;
                    rejected_step = true;
                    result[name].push_back({{"rejected", true}, {"error", error.what()}});
                }
                if (rejected_step != expected_step)
                    throw std::runtime_error("Continuation rejection expectation was not met: " +
                                             name);
            }
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            result[name] = {{"rejected", true}, {"error", error.what()}};
        }
        if (rejected != expected)
            throw std::runtime_error("History rejection expectation was not met: " + name);
    }
    std::ofstream output(manifest.at("output").get<std::string>());
    output << result.dump();
    if (!output)
        throw std::runtime_error("Cannot write reasoning cases");
    return 0;
}

std::unique_ptr<trtmc::alpamayo::DrivingPrompt> driving_prompt(const json& config) {
    const auto vocabulary = read_bytes(config.at("tokenizer"));
    const auto& p = config.at("parameters");
    return std::make_unique<trtmc::alpamayo::DrivingPrompt>(
        trtmc::alpamayo::CreateBpeTokenizer(vocabulary.data(), vocabulary.size(), false),
        trtmc::alpamayo::HistoryTokenConfig{p.at("points"), p.at("bins"), p.at("token_offset"),
                                            p.at("placeholder_token"),
                                            p.at("minimum").get<std::array<float, 3>>(),
                                            p.at("maximum").get<std::array<float, 3>>()},
        config.at("spatial_merge_size"), config.at("capacity"));
}

trtmc::alpamayo::TextPrefill prepare_text(trtmc::alpamayo::TextDecoder& decoder, const json& config,
                                          const TensorMap& inputs,
                                          const trtmc::alpamayo::DrivingPrompt* prompt = nullptr) {
    std::optional<trtmc::alpamayo::DrivingBatch> batch;
    if (config.contains("grids_by_row")) {
        if (!prompt)
            throw std::invalid_argument("Native driving tokenizer is unavailable");
        batch = prompt->encode_batch(
            config.at("grids_by_row").get<std::vector<std::vector<std::array<int64_t, 3>>>>(),
            config.at("history_xyz").get<std::vector<std::vector<float>>>(),
            config.at("pad_token"));
    }
    const auto grids =
        batch ? batch->image_grids : config.at("grids").get<std::vector<std::array<int64_t, 3>>>();
    std::vector<trtmc::alpamayo::VisionFeatures> images;
    for (std::size_t i = 0; i < grids.size(); ++i) {
        const auto name = "image_" + std::to_string(i);
        const auto& feature = inputs.at(name);
        if (feature.shape.size() != 2)
            throw std::invalid_argument("Invalid prefill feature rank");
        trtmc::alpamayo::VisionFeatures image;
        image.tokens = feature.shape[0];
        image.width = feature.shape[1];
        image.dtype = feature.dtype;
        image.features.resize(feature.nbytes());
        if (!image.features.empty())
            std::memcpy(image.features.data(), feature.data, image.features.size());
        for (std::size_t j = 0;; ++j) {
            const auto found = inputs.find(name + "_deepstack_" + std::to_string(j));
            if (found == inputs.end())
                break;
            if (found->second.shape != feature.shape || found->second.dtype != feature.dtype)
                throw std::invalid_argument("Invalid prefill DeepStack input");
            image.deepstack.emplace_back(found->second.nbytes());
            if (!image.deepstack.back().empty())
                std::memcpy(image.deepstack.back().data(), found->second.data,
                            image.deepstack.back().size());
        }
        images.push_back(std::move(image));
    }
    const auto& p = config.at("config");
    return decoder.prepare(
        batch ? batch->tokens : config.at("tokens").get<std::vector<std::vector<int32_t>>>(), grids,
        images,
        batch ? batch->attention_mask
              : config.value("attention_mask", std::vector<std::vector<int32_t>>{}),
        {p.at("image_token"), p.at("video_token"), p.at("vision_start_token"),
         p.at("vision_end_token"), p.at("spatial_merge_size"), p.at("head_width"),
         p.at("mrope_section").get<std::array<int64_t, 3>>(), p.at("rope_theta")});
}

int decode_text(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    auto plan = read_bytes(manifest.at("engine"));
    const std::unique_ptr<IBackend, decltype(&trtmc_destroy_backend)> backend(
        trtmc_create_backend(), trtmc_destroy_backend);
    if (!backend)
        throw std::runtime_error("Model Connect backend creation failed");
    trtmc::alpamayo::TextDecoder decoder(backend->create_module(plan.data(), plan.size(), {}));
    std::vector<char>().swap(plan);
    auto prompt = manifest.contains("driving") ? driving_prompt(manifest.at("driving")) : nullptr;
    std::vector<std::pair<trtmc::alpamayo::TextState, json>> retained;
    json summaries = json::array();
    for (const auto& request : manifest.at("requests")) {
        HostInputs inputs(request.at("inputs"));
        const auto mutations = request.value("mutations", json::object());
        for (const auto& [name, mutation] : mutations.items()) {
            auto& tensor = inputs.tensors.at(name);
            if (mutation.contains("shape"))
                tensor.shape = mutation.at("shape").get<std::vector<int64_t>>();
            if (mutation.contains("dtype"))
                tensor.dtype = DType::kFloat32;
        }
        if (request.value("reuse_cache", false)) {
            if (retained.empty())
                throw std::invalid_argument("No preceding text cache");
            for (const auto& [name, tensor] : retained.back().first.past())
                inputs.tensors.insert_or_assign(name, tensor);
        }
        const bool expected = request.value("expect_error", false);
        bool rejected = false;
        try {
            std::optional<trtmc::alpamayo::TextPrefill> prefill;
            if (request.contains("prefill")) {
                prefill =
                    prepare_text(decoder, request.at("prefill"), inputs.tensors, prompt.get());
                inputs.tensors = prefill->tensors();
                if (request.contains("prepared_outputs"))
                    write_tensors(inputs.tensors, request.at("prepared_outputs"));
            }
            auto state = [&]() {
                if (!request.contains("generation"))
                    return decoder.evaluate(inputs.tensors);
                auto generated = decoder.generate(
                    inputs.tensors,
                    prefill ? prefill->positions.deltas
                            : request.at("rope_deltas").get<std::vector<int64_t>>(),
                    prefill ? request.at("prefill").at("config").at("rope_theta").get<float>()
                            : request.at("rope_theta").get<float>(),
                    reasoning_options(request.at("generation")));
                summaries.push_back({{"name", request.at("name")},
                                     {"tokens", generated.tokens},
                                     {"action_offsets", generated.action_offsets},
                                     {"rope_deltas", generated.rope_deltas},
                                     {"prompt_mask", generated.prompt_mask},
                                     {"generated", generated.generated},
                                     {"cache_shape", generated.cache.shapes.at("key_0")}});
                if (prompt) {
                    json text = json::array();
                    for (const auto& row : generated.tokens) {
                        const std::vector<int32_t> output(row.end() - generated.generated,
                                                          row.end());
                        const auto extracted = prompt->extract_text(output);
                        text.push_back({{"cot", extracted.cot},
                                        {"meta_action", extracted.meta_action},
                                        {"answer", extracted.answer}});
                    }
                    summaries.back()["text"] = std::move(text);
                }
                return std::move(generated.cache);
            }();
            if (!expected)
                retained.emplace_back(std::move(state), request.value("outputs", json::object()));
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            std::cout << "Expected text rejection: " << error.what() << std::endl;
        }
        if (expected != rejected)
            throw std::runtime_error("Text rejection expectation was not met");
        std::cout << "Processed text request " << request.value("name", "unnamed") << std::endl;
    }
    // Keep output snapshots alive across all following requests and recovery.
    for (auto& [state, destinations] : retained)
        if (!destinations.empty())
            write_tensors(state.tensors(), destinations);
    if (manifest.contains("output")) {
        std::ofstream output(manifest.at("output").get<std::string>());
        output << summaries.dump();
        if (!output)
            throw std::runtime_error("Cannot write generated reasoning");
    }
    return 0;
}

int finish_driving_cases(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    auto plan = read_bytes(manifest.at("engine"));
    const std::unique_ptr<IBackend, decltype(&trtmc_destroy_backend)> backend(
        trtmc_create_backend(), trtmc_destroy_backend);
    if (!backend)
        throw std::runtime_error("Model Connect backend creation failed");
    trtmc::alpamayo::ActionStep action(backend->create_module(plan.data(), plan.size(), {}));
    std::vector<char>().swap(plan);
    auto prompt = driving_prompt(manifest.at("driving"));
    trtmc::alpamayo::ReasoningResult reasoning;
    const auto& generated = manifest.at("reasoning");
    reasoning.tokens = generated.at("tokens").get<std::vector<std::vector<int32_t>>>();
    reasoning.action_offsets = generated.at("action_offsets").get<std::vector<int64_t>>();
    reasoning.rope_deltas = generated.at("rope_deltas").get<std::vector<int64_t>>();
    reasoning.generated = generated.at("generated");
    // Earlier unpadded component fixtures predate this explicit handoff field.
    reasoning.prompt_mask = generated.value(
        "prompt_mask", std::vector<std::vector<float>>(reasoning.tokens.size(), {0.0F}));
    {
        HostInputs inputs(manifest.at("prefix"));
        for (const auto& [name, tensor] : inputs.tensors) {
            if (name.rfind("prefix_", 0) != 0)
                throw std::invalid_argument("Invalid reasoning cache name");
            const auto key = name.substr(7);
            reasoning.cache.cache_dtype = tensor.dtype;
            reasoning.cache.shapes[key] = tensor.shape;
            auto& bytes = reasoning.cache.storage[key];
            bytes.resize(tensor.nbytes());
            std::memcpy(bytes.data(), tensor.data, bytes.size());
        }
    }
    const auto& p = manifest.at("parameters");
    const auto& pos = manifest.at("positions");
    trtmc::alpamayo::DrivingOptions options{
        {pos.at("image_token"), pos.at("video_token"), pos.at("vision_start_token"),
         pos.at("vision_end_token"), pos.at("spatial_merge_size"), pos.at("head_width"),
         pos.at("mrope_section").get<std::array<int64_t, 3>>(), pos.at("rope_theta")},
        {p.at("n_waypoints"), p.at("history_points"), p.at("dt"), p.at("v_lambda"), p.at("v_ridge"),
         p.at("accel_mean"), p.at("accel_std"), p.at("curvature_mean"), p.at("curvature_std")},
        {}};
    json output = json::object();
    for (const auto& request : manifest.at("requests")) {
        const bool expected = request.value("expect_error", false);
        bool rejected = false;
        const auto name = request.at("name").get<std::string>();
        try {
            HostInputs noise({{"noise", request.at("noise")}});
            std::vector<trtmc::alpamayo::DrivingHistoryInput> histories;
            for (const auto& h : request.at("histories"))
                histories.push_back({h.at("xyz").get<std::vector<float>>(),
                                     h.at("rotations").get<std::vector<float>>()});
            options.denoising_steps = request.value("steps", 10);
            const auto samples = trtmc::alpamayo::finish_driving(
                action, *prompt, reasoning, histories, options, noise.tensors.at("noise"));
            json rows = json::array();
            for (const auto& sample : samples)
                rows.push_back({{"tokens", sample.generated_tokens},
                                {"text",
                                 {{"cot", sample.text.cot},
                                  {"meta_action", sample.text.meta_action},
                                  {"answer", sample.text.answer}}},
                                {"actions", sample.normalized_actions},
                                {"xyz", sample.trajectory.xyz},
                                {"rotations", sample.trajectory.rotations},
                                {"initial_velocity", sample.trajectory.initial_velocity}});
            output[name] = std::move(rows);
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            output[name] = {{"rejected", true}, {"error", error.what()}};
        }
        if (expected != rejected)
            throw std::runtime_error("Driving rejection expectation was not met: " + name);
    }
    std::ofstream destination(manifest.at("output").get<std::string>());
    destination << output.dump();
    if (!destination)
        throw std::runtime_error("Cannot write completed driving samples");
    return 0;
}

int driving_prompts(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    auto prompt = driving_prompt(manifest);
    json results = json::object();
    for (const auto& item : manifest.at("cases")) {
        const bool expected = item.value("expect_error", false);
        bool rejected = false;
        try {
            const auto name = item.at("name").get<std::string>();
            if (item.contains("decode")) {
                const auto ids = item.at("decode").get<std::vector<int32_t>>();
                results[name] = {{"text", prompt->decode(ids, false)},
                                 {"text_without_special", prompt->decode(ids, true)}};
                if (item.value("extract_text", false)) {
                    const auto text = prompt->extract_text(ids);
                    results[name]["extracted"] = {{"cot", text.cot},
                                                  {"meta_action", text.meta_action},
                                                  {"answer", text.answer}};
                }
            } else if (item.contains("grids_by_row")) {
                const auto batch = prompt->encode_batch(
                    item.at("grids_by_row").get<std::vector<std::vector<std::array<int64_t, 3>>>>(),
                    item.at("history_xyz").get<std::vector<std::vector<float>>>(),
                    item.at("pad_token"));
                results[name] = {{"tokens", batch.tokens},
                                 {"attention_mask", batch.attention_mask},
                                 {"image_grids", batch.image_grids}};
            } else {
                auto history = item.at("history_xyz").get<std::vector<float>>();
                if (item.value("nonfinite_history", false))
                    history.at(0) = std::numeric_limits<float>::quiet_NaN();
                results[name] = prompt->encode(
                    item.at("grids").get<std::vector<std::array<int64_t, 3>>>(), history);
            }
        } catch (const std::invalid_argument& error) {
            if (!expected)
                throw;
            rejected = true;
            results[item.at("name").get<std::string>()] = {{"rejected", true},
                                                           {"error", error.what()}};
        }
        if (rejected != expected)
            throw std::runtime_error("Driving prompt rejection expectation was not met");
    }
    std::ofstream output(manifest.at("output").get<std::string>());
    output << results.dump();
    if (!output)
        throw std::runtime_error("Cannot write driving prompts");
    return 0;
}

int history_tokens(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    const auto& p = manifest.at("parameters");
    trtmc::alpamayo::HistoryTokenizer tokenizer(
        {p.at("points"), p.at("bins"), p.at("token_offset"), p.at("placeholder_token"),
         p.at("minimum").get<std::array<float, 3>>(), p.at("maximum").get<std::array<float, 3>>()});
    json results = json::object();
    for (const auto& item : manifest.at("cases")) {
        bool rejected = false;
        const bool expected = item.value("expect_error", false);
        try {
            auto xyz = item.at("history_xyz").get<std::vector<float>>();
            if (item.value("nonfinite_history", false))
                xyz.at(0) = std::numeric_limits<float>::quiet_NaN();
            const auto prompt = item.at("prompt").get<std::vector<int32_t>>();
            results[item.at("name").get<std::string>()] = {
                {"history_tokens", tokenizer.encode(xyz)}, {"prompt", tokenizer.fuse(prompt, xyz)}};
        } catch (const std::invalid_argument&) {
            if (!expected)
                throw;
            rejected = true;
            results[item.at("name").get<std::string>()] = {{"rejected", true}};
        }
        if (rejected != expected)
            throw std::runtime_error("History token rejection expectation was not met");
    }
    std::ofstream output(manifest.at("output").get<std::string>());
    output << results.dump();
    if (!output)
        throw std::runtime_error("Cannot write history tokens");
    return 0;
}

int decode_trajectories(const std::string& path) {
    std::ifstream source(path);
    const auto manifest = json::parse(source);
    const auto& p = manifest.at("parameters");
    trtmc::alpamayo::TrajectoryDecoder decoder(
        {p.at("n_waypoints").get<std::size_t>(), p.at("history_points").get<std::size_t>(),
         p.at("dt").get<double>(), p.at("v_lambda").get<double>(), p.at("v_ridge").get<double>(),
         p.at("accel_mean").get<float>(), p.at("accel_std").get<float>(),
         p.at("curvature_mean").get<float>(), p.at("curvature_std").get<float>()});
    json results = json::object();
    for (const auto& item : manifest.at("cases")) {
        if (item.contains("history_xyz")) {
            const bool expected = item.value("expect_history_error", false);
            bool rejected = false;
            try {
                decoder.set_history(item.at("history_xyz").get<std::vector<float>>(),
                                    item.at("history_rotations").get<std::vector<float>>());
            } catch (const std::invalid_argument&) {
                if (!expected)
                    throw;
                rejected = true;
            }
            if (rejected != expected)
                throw std::runtime_error("History rejection expectation was not met");
        }
        const bool expected = item.value("expect_decode_error", false);
        bool rejected = false;
        try {
            auto actions = item.at("actions").get<std::vector<float>>();
            if (item.value("nonfinite_actions", false))
                actions.at(0) = std::numeric_limits<float>::quiet_NaN();
            const auto trajectory = decoder.decode(actions);
            results[item.at("name").get<std::string>()] = {
                {"xyz", trajectory.xyz},
                {"rotations", trajectory.rotations},
                {"initial_velocity", trajectory.initial_velocity}};
        } catch (const std::invalid_argument&) {
            if (!expected)
                throw;
            rejected = true;
            results[item.at("name").get<std::string>()] = {{"rejected", true}};
        }
        if (rejected != expected)
            throw std::runtime_error("Action rejection expectation was not met");
    }
    std::ofstream output(manifest.at("output").get<std::string>());
    output << results.dump();
    if (!output)
        throw std::runtime_error("Cannot write decoded trajectories");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: alpamayo_action_step_test ENGINE REQUESTS.json\n";
        return 2;
    }
    try {
        if (std::string(argv[1]) == "--finish-driving")
            return finish_driving_cases(argv[2]);
        if (std::string(argv[1]) == "--prompt")
            return driving_prompts(argv[2]);
        if (std::string(argv[1]) == "--positions")
            return image_position_cases(argv[2]);
        if (std::string(argv[1]) == "--reasoning")
            return reasoning_cases(argv[2]);
        if (std::string(argv[1]) == "--text")
            return decode_text(argv[2]);
        if (std::string(argv[1]) == "--history")
            return history_tokens(argv[2]);
        if (std::string(argv[1]) == "--preprocess")
            return prepare_images(argv[2]);
        if (std::string(argv[1]) == "--vision")
            return encode_vision(argv[2]);
        if (std::string(argv[1]) == "--trajectory")
            return decode_trajectories(argv[2]);
        std::ifstream manifest(argv[2]);
        const auto requests = json::parse(manifest).at("requests");
        auto plan = read_bytes(argv[1]);
        const std::unique_ptr<IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        if (!backend)
            throw std::runtime_error("Model Connect backend creation failed");
        trtmc::alpamayo::ActionStep action(backend->create_module(plan.data(), plan.size(), {}));
        std::vector<char>().swap(plan);
        for (const auto& request : requests) {
            if (request.contains("context") || request.contains("generated_context")) {
                const auto generated = request.value("generated_context", json::object());
                HostInputs context(generated.empty() ? request.at("context")
                                                     : generated.at("inputs"));
                const bool expect_error = request.value("expect_context_error", false);
                bool rejected = false;
                try {
                    if (generated.empty())
                        action.set_context(context.tensors);
                    else
                        action.set_generated_context(
                            context.tensors, generated.at("offsets").get<std::vector<int64_t>>(),
                            generated.at("rope_deltas").get<std::vector<int64_t>>(),
                            generated.value("prompt_mask",
                                            std::vector<std::vector<float>>(
                                                generated.at("offsets").size(), {0.0F})),
                            generated.at("rope_theta").get<float>());
                } catch (const std::invalid_argument& error) {
                    if (!expect_error)
                        throw;
                    rejected = true;
                    std::cout << "Expected context rejection: " << error.what() << '\n';
                }
                if (expect_error != rejected)
                    throw std::runtime_error("Invalid context was unexpectedly accepted");
            }
            for (const auto& step : request.value("steps", json::array())) {
                HostInputs inputs(step.at("inputs"));
                std::vector<float> output;
                const bool expect_error = step.value("expect_flow_error", false);
                bool rejected = false;
                try {
                    output = step.contains("integration_steps")
                                 ? action.integrate(inputs.tensors.at("actions"),
                                                    step.at("integration_steps").get<int>())
                                 : action.evaluate(inputs.tensors.at("actions"),
                                                   inputs.tensors.at("timesteps"));
                } catch (const std::invalid_argument& error) {
                    if (!expect_error)
                        throw;
                    rejected = true;
                    std::cout << "Expected flow rejection: " << error.what() << '\n';
                }
                if (expect_error != rejected)
                    throw std::runtime_error("Invalid flow was unexpectedly accepted");
                if (rejected)
                    continue;
                std::ofstream file(step.at("output").get<std::string>(), std::ios::binary);
                if (!file.write(reinterpret_cast<const char*>(output.data()),
                                static_cast<std::streamsize>(output.size() * sizeof(float))))
                    throw std::runtime_error("Cannot write action output");
                std::cout << "Wrote " << output.size() << " FP32 action-stage values\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
