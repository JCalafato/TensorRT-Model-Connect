/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc::diffusion_gemma {
namespace {
std::vector<char> require_section(const BundleReader& bundle, const std::string& name) {
    const auto* section = bundle.find_section(name);
    if (!section || section->length == 0)
        throw std::runtime_error("DiffusionGemma missing bundle section: " + name);
    return bundle.read_section(name);
}
NativeConfig parse(const nlohmann::json& root) {
    if (root.at("schema_version").get<int>() != 1)
        throw std::invalid_argument("DiffusionGemma unsupported runtime metadata schema");
    NativeConfig config;
    config.hidden_size = root.at("hidden_size").get<std::int32_t>();
    config.max_sequence_length = root.at("max_sequence_length").get<std::int32_t>();
    config.cache_widths = root.at("cache_widths").get<std::vector<std::int32_t>>();
    const auto& generation = root.at("generation");
    config.eos_token_ids = generation.at("eos_token_ids").get<std::vector<std::int32_t>>();
    auto& s = config.sampling;
    s.vocabulary_size = root.at("vocab_size").get<std::int32_t>();
    s.canvas_length = generation.at("canvas_length").get<std::int32_t>();
    s.max_denoising_steps = generation.at("max_denoising_steps").get<std::int32_t>();
    s.stability_threshold = generation.at("stability_threshold").get<std::int32_t>();
    s.t_max = generation.at("t_max").get<float>();
    s.t_min = generation.at("t_min").get<float>();
    s.entropy_bound = generation.at("entropy_bound").get<float>();
    s.confidence_threshold = generation.at("confidence_threshold").get<float>();
    return config;
}
} // namespace

ITask* create(const FamilyContext& context) {
    const auto metadata = require_section(context.reader, "runtime.json");
    const auto root = nlohmann::json::parse(metadata.begin(), metadata.end());
    auto config = parse(root);
    auto tokenizer_data = require_section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(), true);
    const auto plan = require_section(context.reader, "engine.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("DiffusionGemma failed to load native engine");
    engine->set_timing_label("diffusion_gemma shared encoder/denoiser");
    if (context.reader.info().task == "vision_language_generation" ||
        context.reader.info().task == "images_text_to_text") {
        const auto data = require_section(context.reader, "image_config.json");
        const auto image = nlohmann::json::parse(data.begin(), data.end());
        ImageConfig image_config;
        image_config.patch_size = image.at("patch_size");
        image_config.pooling_kernel = image.at("pooling_kernel");
        image_config.soft_tokens = image.at("soft_tokens");
        image_config.hidden_size = image.at("hidden_size");
        image_config.image_token = image.at("image_token");
        image_config.begin_token = image.at("begin_token");
        image_config.end_token = image.at("end_token");
        for (auto token :
             {image_config.image_token, image_config.begin_token, image_config.end_token})
            if (token < 0 || token >= config.sampling.vocabulary_size)
                throw std::invalid_argument("DiffusionGemma invalid image marker metadata");
        if (image_config.hidden_size != config.hidden_size)
            throw std::invalid_argument("DiffusionGemma vision/text hidden size mismatch");
        auto bytes = require_section(context.reader, "vision.plan");
        auto module = context.backend.create_module(bytes.data(), bytes.size(), {});
        auto vision = std::make_unique<VisionEncoder>(std::move(module), image_config);
        return new VisionPipeline(std::move(engine), std::move(config), std::move(tokenizer),
                                  root.at("bos_token_id").get<std::int32_t>(), std::move(vision),
                                  context.reader.info().task);
    }
    return new Pipeline(std::move(engine), std::move(config), std::move(tokenizer),
                        root.at("bos_token_id").get<std::int32_t>());
}
} // namespace trtmc::diffusion_gemma

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("diffusion_gemma does not support --kv-cache-size");
    return trtmc::diffusion_gemma::create(context);
}
