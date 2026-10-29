/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "driving_record.h"
#ifdef TRTMC_HAS_EDGE_LLM
#include "edge_llm/adapter.h"
#endif
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::alpamayo {
namespace {
using Json = nlohmann::json;

class Pipeline final : public IStructuredDecision {
  public:
    explicit Pipeline(const FamilyContext& context) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("alpamayo does not support --kv-cache-size");
        const auto bytes = context.reader.read_section("runtime.json");
        const auto config = Json::parse(bytes.begin(), bytes.end());
        if (config.at("version") != 1 || config.at("precision") != "fp16" ||
            config.at("execution") != "native")
            throw std::invalid_argument("Unsupported Alpamayo native bundle contract");
        max_batch_ = config.at("max_batch_size").get<std::size_t>();
        if (max_batch_ != 1 && max_batch_ != 6)
            throw std::invalid_argument("Alpamayo batch capacity must be one or six");
        const auto& p = config.at("positions");
        options_.positions = {p.at("image_token"),
                              p.at("video_token"),
                              p.at("vision_start_token"),
                              p.at("vision_end_token"),
                              p.at("spatial_merge_size"),
                              p.at("head_width"),
                              p.at("mrope_section").get<std::array<int64_t, 3>>(),
                              p.at("rope_theta")};
        const auto& a = config.at("action_space");
        options_.action_space = {
            a.at("n_waypoints"), a.at("history_points"), a.at("dt"),
            a.at("v_lambda"),    a.at("v_ridge"),        a.at("accel_mean"),
            a.at("accel_std"),   a.at("curvature_mean"), a.at("curvature_std")};
        // Validate the physical contract even before creating any GPU module.
        TrajectoryDecoder validate_action(options_.action_space);
        const auto& r = config.at("reasoning");
        options_.reasoning.max_new_tokens = r.at("max_new_tokens");
        options_.reasoning.future_start_token = r.at("future_start_token");
        options_.reasoning.pad_token = r.at("pad_token");
        options_.reasoning.eos_tokens = r.at("eos_tokens").get<std::vector<int32_t>>();
        options_.reasoning.sampling = {r.at("temperature"), r.at("top_p"), r.at("top_k"),
                                       r.at("suppressed_begin"), r.at("suppressed_end")};
        const auto& h = config.at("history");
        const auto tokenizer = context.reader.read_section("tokenizer.json");
        prompt_ = std::make_unique<DrivingPrompt>(
            CreateBpeTokenizer(tokenizer.data(), tokenizer.size()),
            HistoryTokenConfig{h.at("points"), h.at("bins"), h.at("token_offset"),
                               h.at("placeholder_token"),
                               h.at("minimum").get<std::array<float, 3>>(),
                               h.at("maximum").get<std::array<float, 3>>()},
            options_.positions.spatial_merge_size, config.at("prompt_capacity"));
        const auto& v = config.at("vision");
        processor_ = std::make_unique<VisionPreprocessor>(VisionPreprocessConfig{
            v.at("patch_size"), v.at("temporal_patch_size"), v.at("spatial_merge_size"),
            v.at("num_position_embeddings"), v.at("head_width"), v.at("min_pixels"),
            v.at("max_pixels"), v.at("patch_capacity"),
            v.at("image_mean").get<std::array<float, 3>>(),
            v.at("image_std").get<std::array<float, 3>>(), v.at("rescale_factor")});
        auto module = [&](const std::string& name) {
            const auto plan = context.reader.read_section(name);
            auto engine = context.backend.create_module(plan.data(), plan.size(), {});
            if (!engine || !engine->ok())
                throw std::runtime_error("Cannot load Alpamayo " + name);
            return engine;
        };
        vision_ = std::make_unique<VisionEncoder>(module("vision.plan"), v.at("spatial_merge_size"),
                                                  v.at("num_position_embeddings"),
                                                  v.at("deepstack_count"));
        text_ = std::make_unique<TextDecoder>(module("text.plan"));
        action_ = std::make_unique<ActionStep>(module("action.plan"));
    }

    StructuredDecisionResult decide(const StructuredDecisionRequest& request) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto record = prepare_driving_request(request, options_, max_batch_);
        const auto samples =
            infer_driving(*processor_, *vision_, *text_, *action_, *prompt_, record.images,
                          record.histories, record.options, record.noise());
        return format_driving_result(record, samples);
    }

  private:
    std::size_t max_batch_{};
    DrivingOptions options_{};
    std::unique_ptr<DrivingPrompt> prompt_;
    std::unique_ptr<VisionPreprocessor> processor_;
    std::unique_ptr<VisionEncoder> vision_;
    std::unique_ptr<TextDecoder> text_;
    std::unique_ptr<ActionStep> action_;
    std::mutex mutex_;
};
} // namespace
} // namespace trtmc::alpamayo

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.find_section("edge_llm.json")) {
        if (context.kv_cache_size_bytes != 0)
            throw std::invalid_argument("alpamayo does not support --kv-cache-size");
#ifdef TRTMC_HAS_EDGE_LLM
        return trtmc::alpamayo::edge_llm::create(context.reader);
#else
        throw std::runtime_error(
            "Alpamayo Edge bundle requires the optional Edge-LLM runtime adapter");
#endif
    }
    return new trtmc::alpamayo::Pipeline(context);
}
