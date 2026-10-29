/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/alpamayo/runtime/edge_llm/adapter.h"

#include "families/alpamayo/runtime/driving_record.h"
#include "families/alpamayo/runtime/edge_llm/contract.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <edgellm/cpp/common/logger.h>
#include <edgellm/cpp/runtime/llmInferenceRuntime.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <sys/utsname.h>

namespace trtmc::alpamayo::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Alpamayo Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Alpamayo Edge runtime host");
    std::ifstream release("/etc/os-release");
    std::string line, os_version;
    while (std::getline(release, line)) {
        if (line.rfind("VERSION_ID=", 0) == 0) {
            os_version = line.substr(11);
            if (os_version.size() >= 2 && os_version.front() == char(34) &&
                os_version.back() == char(34))
                os_version = os_version.substr(1, os_version.size() - 2);
        }
    }
    int device = 0, cuda_version = 0;
    check_cuda(cudaGetDevice(&device));
    check_cuda(cudaRuntimeGetVersion(&cuda_version));
    cudaDeviceProp gpu{};
    check_cuda(cudaGetDeviceProperties(&gpu, device));
    const int trt_version = getInferLibVersion();
    const std::string trt =
        std::to_string(trt_version / 10000) + "." + std::to_string((trt_version % 10000) / 100) +
        "." + std::to_string(trt_version % 100) + "." + std::to_string(getInferLibBuildVersion());
    const std::string cuda =
        std::to_string(cuda_version / 1000) + "." + std::to_string((cuda_version % 1000) / 10);
    if (target.at("os") != "linux" || target.at("os_version") != os_version ||
        target.at("arch") != host.machine || target.at("sm") != gpu.major * 10 + gpu.minor ||
        target.at("cuda_version") != cuda || target.at("tensorrt_version") != trt)
        throw std::runtime_error(
            "Alpamayo Edge bundle requires its build GPU and CUDA/TensorRT stack");
}

/// Own extracted engine/checkpoint files until after the Edge runtime is destroyed.
class Artifacts {
  public:
    explicit Artifacts(const BundleReader& bundle, const nlohmann::json& marker) {
        std::set<std::string> names;
        for (const auto& entry : marker.at("artifacts")) {
            const auto name = entry.get<std::string>();
            if (!safe_artifact_path(name) || !names.insert(name).second ||
                !bundle.find_section(name))
                throw std::runtime_error("Invalid Alpamayo Edge artifact: " + name);
        }
        for (const auto* required :
             {"edge_llm/engine/visual/visual.engine", "edge_llm/engine/visual/config.json",
              "edge_llm/engine/llm.engine", "edge_llm/engine/config.json",
              "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
              "edge_llm/engine/chat_template.jinja", "edge_llm/engine/vocab_map.safetensors",
              "edge_llm/engine/action/action.engine", "edge_llm/engine/action/config.json",
              "edge_llm/checkpoint/config.json",
              "edge_llm/checkpoint/model.safetensors.index.json"})
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(std::string("Required Alpamayo Edge artifact missing: ") +
                                         required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-alpamayo-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Alpamayo Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Alpamayo Edge artifact: " + name);
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Artifacts() { cleanup(); }
    Artifacts(const Artifacts&) = delete;
    Artifacts& operator=(const Artifacts&) = delete;
    std::string engine() const { return (root_ / "edge_llm/engine").string(); }
    std::string checkpoint() const { return (root_ / "edge_llm/checkpoint").string(); }

  private:
    void cleanup() noexcept {
        std::error_code ignored;
        fs::remove_all(root_, ignored);
    }
    fs::path root_;
};

/// Close the plugin handle after runtime destruction; registrations remain mapped.
struct CloseLibrary {
    void operator()(void* handle) const noexcept {
        if (handle)
            dlclose(handle);
    }
};

/// Initialize the CMake-installed adjacent plugin without process-global environment mutation.
std::unique_ptr<void, CloseLibrary> load_plugin() {
    Dl_info location{};
    if (!dladdr(reinterpret_cast<void*>(&create), &location) || !location.dli_fname)
        throw std::runtime_error("Cannot locate Alpamayo family library");
    const auto path =
        fs::absolute(location.dli_fname).parent_path() / "libNvInfer_edgellm_plugin.so";
    std::unique_ptr<void, CloseLibrary> plugin(
        dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE));
    if (!plugin)
        throw std::runtime_error("Cannot load CMake-installed Edge plugin: " +
                                 std::string(dlerror()));
    using Initialize = bool (*)(void*, const char*);
    auto initialize = reinterpret_cast<Initialize>(dlsym(plugin.get(), "initEdgellmPlugins"));
    if (!initialize || !initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
        throw std::runtime_error("Cannot initialize Alpamayo Edge plugin");
    return plugin;
}

/// Stream ownership is independent of construction success and outlives the Edge instance.
class Stream {
  public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return value_; }

  private:
    cudaStream_t value_{nullptr};
};

/// Complete Edge driving execution; physical decoding stays owned by Alpamayo.
class EdgeTask final : public IStructuredDecision {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(artifacts_.engine(), artifacts_.engine(),
                   std::unordered_map<std::string, std::string>{}, stream_.get(),
                   trt_edgellm::rt::ContextCacheConfig{}, artifacts_.checkpoint()),
          capacity_(marker.at("max_sequence_length")), input_limit_(marker.at("max_input_length")) {
        const auto bytes = bundle.read_section("runtime.json");
        const auto config = nlohmann::json::parse(bytes.begin(), bytes.end());
        if (config.at("version") != 1 || config.at("precision") != "fp16" ||
            config.at("execution") != "edge_llm")
            throw std::invalid_argument("Unsupported Alpamayo Edge runtime contract");
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
        const auto tokenizer = bundle.read_section("edge_llm/engine/tokenizer.json");
        prompt_ = std::make_unique<DrivingPrompt>(
            CreateBpeTokenizer(tokenizer.data(), tokenizer.size()),
            HistoryTokenConfig{h.at("points"), h.at("bins"), h.at("token_offset"),
                               h.at("placeholder_token"),
                               h.at("minimum").get<std::array<float, 3>>(),
                               h.at("maximum").get<std::array<float, 3>>()},
            options_.positions.spatial_merge_size, config.at("prompt_capacity"));

        if (max_batch_ != marker.at("max_batch_size").get<std::size_t>())
            throw std::invalid_argument("Alpamayo Edge batch metadata disagrees");
        const auto excluded =
            marker.at("excluded_output_token_interval").get<std::vector<int32_t>>();
        if (excluded != std::vector<int32_t>{options_.reasoning.sampling.suppressed_begin,
                                             options_.reasoning.sampling.suppressed_end})
            throw std::invalid_argument(
                "Alpamayo Edge output mask differs from provider semantics");
    }
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    StructuredDecisionResult decide(const StructuredDecisionRequest& input) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto document = nlohmann::json::parse(input.document);
        if (document.contains("initial_noise"))
            throw std::invalid_argument(
                "Alpamayo Edge does not expose explicit action noise; select native execution");
        auto record = prepare_driving_request(input, options_, max_batch_);
        const auto seed = record.options.reasoning.seed;
        if (seed > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument(
                "Alpamayo Edge action noise seed must fit int32; select native for larger seeds");
        trt_edgellm::rt::LLMGenerationRequest request{};
        request.temperature = record.options.reasoning.sampling.temperature;
        request.topP = record.options.reasoning.sampling.top_p;
        request.topK = record.options.reasoning.sampling.top_k;
        request.maxGenerateLength = record.options.reasoning.max_new_tokens;
        request.samplingSeed = seed;
        request.applyChatTemplate = true;
        request.addGenerationPrompt = true;
        for (std::size_t row = 0; row < record.histories.size(); ++row) {
            auto& target = request.requests.emplace_back();
            target.samplingSeed = seed + row;
            trt_edgellm::rt::Message system{};
            system.role = "system";
            system.contentIsArray = true;
            system.contents.push_back(
                {"text", "You are a driving assistant that generates safe and accurate actions."});
            target.messages.push_back(std::move(system));
            trt_edgellm::rt::Message message{};
            message.role = "user";
            message.contentIsArray = true;
            for (const auto& image : record.images[row]) {
                trt_edgellm::rt::Tensor pixels({1, image.height, image.width, 3},
                                               trt_edgellm::rt::DeviceType::kCPU,
                                               nvinfer1::DataType::kUINT8);
                std::memcpy(pixels.rawPointer(), image.rgb, image.bytes);
                target.imageBuffers.emplace_back(std::move(pixels));
                message.contents.push_back({"image", ""});
            }
            target.pastTrajectory.emplace();
            const auto& xyz = record.histories[row].xyz;
            for (std::size_t index = 0; index < xyz.size(); index += 3)
                target.pastTrajectory->emplace_back(xyz[index], xyz[index + 1], xyz[index + 2]);
            message.contents.push_back({"trajectory", ""});
            message.contents.push_back(
                {"text", "output the chain-of-thought reasoning of the driving process, then "
                         "output the future trajectory."});
            target.messages.push_back(std::move(message));
        }
        runtime_.setActionNoiseSeed(static_cast<int32_t>(seed));
        try {
            trt_edgellm::rt::LLMGenerationResponse response{};
            if (!runtime_.handleRequest(request, response, stream_.get()))
                throw std::runtime_error("Alpamayo Edge driving generation failed");
            check_cuda(cudaStreamSynchronize(stream_.get()));
            const auto rows = record.histories.size();
            if (response.outputIds.size() != rows || response.outputTrajectories.size() != rows ||
                response.inputTokenCounts.size() != rows || response.finishReasons.size() != rows)
                throw std::runtime_error("Alpamayo Edge returned an incomplete driving batch");
            std::vector<DrivingSample> samples;
            for (std::size_t row = 0; row < rows; ++row) {
                const auto tokens = response.inputTokenCounts[row];
                const auto reason = response.finishReasons[row];
                if (tokens <= 0 || tokens > input_limit_ ||
                    request.maxGenerateLength > capacity_ - tokens ||
                    response.outputIds[row].empty() ||
                    response.outputIds[row].size() >
                        static_cast<std::size_t>(request.maxGenerateLength) ||
                    response.outputTrajectories[row].size() != options_.action_space.waypoints ||
                    (reason != trt_edgellm::rt::FinishReason::kEndId &&
                     reason != trt_edgellm::rt::FinishReason::kLength))
                    throw std::runtime_error("Alpamayo Edge returned an invalid driving response");
                DrivingSample sample;
                sample.generated_tokens = std::move(response.outputIds[row]);
                sample.text = prompt_->extract_text(sample.generated_tokens);
                for (const auto& point : response.outputTrajectories[row]) {
                    if (!std::isfinite(point.first) || !std::isfinite(point.second))
                        throw std::runtime_error("Alpamayo Edge returned non-finite actions");
                    sample.normalized_actions.push_back(point.first);
                    sample.normalized_actions.push_back(point.second);
                }
                TrajectoryDecoder decoder(options_.action_space);
                decoder.set_history(record.histories[row].xyz, record.histories[row].rotations);
                sample.trajectory = decoder.decode(sample.normalized_actions);
                samples.push_back(std::move(sample));
            }
            return format_driving_result(record, samples);
        } catch (...) {
            // Keep request-owned pinned pixels alive until asynchronous work drains.
            cudaStreamSynchronize(stream_.get());
            throw;
        }
    }

  private:
    Artifacts artifacts_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    trt_edgellm::rt::LLMInferenceRuntime runtime_;
    int64_t capacity_, input_limit_;
    std::size_t max_batch_{};
    DrivingOptions options_{};
    std::unique_ptr<DrivingPrompt> prompt_;
    std::mutex mutex_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    const auto batch = marker.at("max_batch_size").get<int>();
    if (marker.at("version") != 1 || marker.at("edge_revision") != kRevision ||
        marker.at("max_sequence_length").get<int>() <= 1 ||
        marker.at("max_input_length").get<int>() <= 0 ||
        marker.at("max_input_length").get<int>() > marker.at("max_sequence_length").get<int>() ||
        (batch != 1 && batch != 6) || marker.at("precision") != "fp16" ||
        !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid Alpamayo Edge bundle contract");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}
} // namespace trtmc::alpamayo::edge_llm
