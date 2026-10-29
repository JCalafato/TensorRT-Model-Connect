/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "driving_record.h"

#include <array>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <stdexcept>

namespace trtmc::alpamayo {
namespace {
using Json = nlohmann::json;

void require(bool valid, const char* message) {
    if (!valid)
        throw std::invalid_argument(std::string("Alpamayo driving record: ") + message);
}

void keys(const Json& value, std::initializer_list<const char*> allowed) {
    require(value.is_object(), "expected a JSON object");
    const std::set<std::string> names(allowed.begin(), allowed.end());
    for (const auto& entry : value.items())
        require(names.count(entry.key()), "unknown record field");
}

int64_t integer(const Json& value) {
    require(value.is_number_integer(), "expected an integer");
    if (value.is_number_unsigned())
        require(value.get<uint64_t>() <= static_cast<uint64_t>(INT64_MAX), "integer overflow");
    return value.get<int64_t>();
}

float number(const Json& value) {
    require(value.is_number(), "expected a finite number");
    const auto result = value.get<float>();
    require(std::isfinite(result), "expected a finite number");
    return result;
}

void flatten(const Json& value, const std::vector<std::size_t>& shape, std::size_t axis,
             std::vector<float>& output) {
    if (axis == shape.size()) {
        output.push_back(number(value));
        return;
    }
    require(value.is_array() && value.size() == shape[axis], "invalid tensor shape");
    for (const auto& part : value)
        flatten(part, shape, axis + 1, output);
}

std::vector<int64_t> timestamps(const Json& value, std::size_t count) {
    require(value.is_array() && value.size() == count, "invalid timestamp count");
    std::vector<int64_t> result;
    for (const auto& time : value) {
        const auto current = integer(time);
        require(current >= 0 && (result.empty() || current > result.back()),
                "timestamps must be nonnegative and strictly increasing");
        result.push_back(current);
    }
    return result;
}

Json rows(const std::vector<float>& values, std::size_t width) {
    Json result = Json::array();
    for (std::size_t offset = 0; offset < values.size(); offset += width) {
        Json row = Json::array();
        for (std::size_t j = 0; j < width; ++j) {
            require(std::isfinite(values.at(offset + j)), "non-finite result");
            row.push_back(values.at(offset + j));
        }
        result.push_back(std::move(row));
    }
    return result;
}

} // namespace

Tensor DrivingRecord::noise() {
    return {initial_noise.data(),
            {static_cast<int64_t>(histories.size()),
             static_cast<int64_t>(options.action_space.waypoints), 2},
            DType::kFloat32};
}

DrivingRecord prepare_driving_request(const StructuredDecisionRequest& request,
                                      const DrivingOptions& defaults, std::size_t max_batch) {
    require(request.images.empty(), "use four camera videos, not unordered still images");
    require(request.max_state_tokens == -1, "use sampling.max_new_tokens for reasoning capacity");
    require(max_batch == 1 || max_batch == 6, "documented batch capacity must be one or six");
    require(defaults.action_space.history_points == 16 && defaults.action_space.waypoints == 64,
            "incompatible trained history or trajectory dimensions");
    require(std::isfinite(defaults.action_space.dt) && defaults.action_space.dt > 0 &&
                defaults.action_space.dt < static_cast<double>(INT64_MAX / 64) / 1000000.0,
            "invalid trajectory interval");
    const auto interval = std::llround(defaults.action_space.dt * 1000000.0);
    require(interval > 0 && interval <= INT64_MAX / 64, "invalid trajectory interval");
    const auto document = Json::parse(request.document);
    keys(document, {"schema", "coordinate_frame", "observations", "num_samples", "sampling",
                    "initial_noise"});
    require(document.at("schema") == "alpamayo.driving.v1", "unsupported schema");
    require(document.at("coordinate_frame") == "current_ego", "expected current ego frame");
    const auto& observations = document.at("observations");
    require(observations.is_array() && !observations.empty(), "observations must not be empty");
    const auto count = integer(document.value("num_samples", Json(1)));
    require((count == 1 || count == 6) && observations.size() <= max_batch / count,
            "requested trajectories exceed batch capacity");
    require(request.videos.size() == observations.size() * 4,
            "expected four cameras per observation");
    DrivingRecord record;
    record.options = defaults;
    const auto sampling = document.value("sampling", Json::object());
    keys(sampling, {"seed", "temperature", "top_p", "top_k", "max_new_tokens", "denoising_steps"});
    auto& options = record.options.reasoning;
    const auto seed = integer(sampling.value("seed", Json(0)));
    require(seed >= 0, "seed must be nonnegative");
    options.seed = static_cast<uint64_t>(seed);
    options.sampling.temperature =
        number(sampling.value("temperature", Json(options.sampling.temperature)));
    options.sampling.top_p = number(sampling.value("top_p", Json(options.sampling.top_p)));
    options.sampling.top_k = integer(sampling.value("top_k", Json(options.sampling.top_k)));
    options.max_new_tokens =
        integer(sampling.value("max_new_tokens", Json(options.max_new_tokens)));
    require(options.sampling.temperature >= 0 && options.sampling.top_p > 0 &&
                options.sampling.top_p <= 1 && options.sampling.top_k >= 0 &&
                options.max_new_tokens > 0 &&
                options.max_new_tokens <= defaults.reasoning.max_new_tokens,
            "invalid or over-capacity reasoning options");
    // Only the documented ten-step Euler flow is exposed in this adapter.
    require(integer(sampling.value("denoising_steps", Json(10))) == 10,
            "the documented action flow uses ten Euler steps");
    record.options.denoising_steps = 10;
    record.pixels.reserve(request.videos.size());
    for (const auto& video : request.videos) {
        require(video.height > 0 && video.width > 0 && video.channels == 3 && video.num_frames == 4,
                "each camera requires four RGB HWC frames");
        const auto height = static_cast<std::size_t>(video.height);
        const auto width = static_cast<std::size_t>(video.width);
        require(height <= std::numeric_limits<std::size_t>::max() / 12 / width,
                "camera dimensions overflow");
        require(video.pixels.size() == height * width * 12, "camera pixel count mismatch");
        auto& pixels = record.pixels.emplace_back();
        pixels.reserve(video.pixels.size());
        for (const auto pixel : video.pixels) {
            require(std::isfinite(pixel) && pixel >= 0 && pixel <= 255 &&
                        pixel == std::floor(pixel),
                    "camera pixels must be exact RGB bytes in [0,255]");
            pixels.push_back(static_cast<uint8_t>(pixel));
        }
    }
    std::set<std::string> ids;
    std::set<std::size_t> video_indexes;
    const std::array<int64_t, 4> cameras{0, 1, 2, 6};
    for (const auto& observation : observations) {
        keys(observation, {"id", "cameras", "history"});
        const auto id = observation.at("id").get<std::string>();
        require(!id.empty() && ids.insert(id).second,
                "observation IDs must be nonempty and unique");
        const auto& history = observation.at("history");
        keys(history, {"xyz", "rotations", "timestamps_us"});
        DrivingHistoryInput state;
        flatten(history.at("xyz"), {16, 3}, 0, state.xyz);
        flatten(history.at("rotations"), {16, 3, 3}, 0, state.rotations);
        const auto times = timestamps(history.at("timestamps_us"), 16);
        for (std::size_t i = 1; i < times.size(); ++i)
            require(times[i] - times[i - 1] == interval,
                    "history timestamps must use the trained interval");
        require(times.back() <= INT64_MAX - 64 * interval, "future timestamps overflow");
        // This also checks all history values using the same physical trajectory
        // contract as the actual action decoder, before any engine execution.
        TrajectoryDecoder history_check(defaults.action_space);
        history_check.set_history(state.xyz, state.rotations);
        const auto& inputs = observation.at("cameras");
        require(inputs.is_array() && inputs.size() == cameras.size(),
                "expected four ordered cameras");
        std::vector<DrivingImageInput> frames;
        for (std::size_t camera = 0; camera < cameras.size(); ++camera) {
            const auto& input = inputs[camera];
            keys(input, {"camera_index", "video_index", "timestamps_us"});
            require(integer(input.at("camera_index")) == cameras[camera],
                    "camera order must be [0,1,2,6]");
            const auto index = integer(input.at("video_index"));
            require(index >= 0 && static_cast<uint64_t>(index) < request.videos.size() &&
                        video_indexes.insert(static_cast<std::size_t>(index)).second,
                    "invalid or repeated camera video index");
            const auto frame_times = timestamps(input.at("timestamps_us"), 4);
            require(frame_times.back() <= times.back(),
                    "camera observations cannot follow the current pose");
            const auto& video = request.videos[index];
            const auto bytes = record.pixels[index].size() / 4;
            for (std::size_t frame = 0; frame < 4; ++frame)
                frames.push_back({record.pixels[index].data() + frame * bytes, bytes, video.height,
                                  video.width});
        }
        for (int64_t sample = 0; sample < count; ++sample) {
            record.ids.push_back(id);
            record.sample_indices.push_back(static_cast<std::size_t>(sample));
            record.history_end_us.push_back(times.back());
            record.images.push_back(frames);
            record.histories.push_back(state);
        }
    }
    if (document.contains("initial_noise")) {
        flatten(document.at("initial_noise"), {record.histories.size(), 64, 2}, 0,
                record.initial_noise);
    } else {
        // Independent native RNG, not a claim of PyTorch seed identity. Explicit
        // initial_noise supports matched-noise numerical replay across providers.
        std::mt19937_64 random(options.seed);
        std::normal_distribution<float> gaussian(0.0F, 1.0F);
        record.initial_noise.resize(record.histories.size() * 64 * 2);
        for (auto& value : record.initial_noise)
            value = gaussian(random);
    }
    return record;
}

StructuredDecisionResult format_driving_result(const DrivingRecord& record,
                                               const std::vector<DrivingSample>& samples) {
    require(samples.size() == record.histories.size() && samples.size() == record.ids.size() &&
                samples.size() == record.sample_indices.size() &&
                samples.size() == record.history_end_us.size(),
            "result batch does not match request");
    const auto interval = std::llround(record.options.action_space.dt * 1000000.0);
    Json output = Json::array();
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto& sample = samples[index];
        require(sample.normalized_actions.size() == 128 && sample.trajectory.xyz.size() == 192 &&
                    sample.trajectory.rotations.size() == 576 &&
                    std::isfinite(sample.trajectory.initial_velocity),
                "incomplete physical trajectory output");
        Json rotations = Json::array();
        const auto matrix_rows = rows(sample.trajectory.rotations, 3);
        Json times = Json::array();
        for (std::size_t i = 0; i < 64; ++i) {
            rotations.push_back(
                {matrix_rows[3 * i], matrix_rows[3 * i + 1], matrix_rows[3 * i + 2]});
            times.push_back(record.history_end_us[index] + static_cast<int64_t>(i + 1) * interval);
        }
        output.push_back({{"id", record.ids[index]},
                          {"sample_index", record.sample_indices[index]},
                          {"tokens", sample.generated_tokens},
                          {"reasoning",
                           {{"cot", sample.text.cot},
                            {"meta_action", sample.text.meta_action},
                            {"answer", sample.text.answer}}},
                          {"normalized_acceleration_curvature", rows(sample.normalized_actions, 2)},
                          {"trajectory",
                           {{"xyz", rows(sample.trajectory.xyz, 3)},
                            {"rotations", rotations},
                            {"timestamps_us", times},
                            {"initial_velocity_mps", sample.trajectory.initial_velocity}}}});
    }
    StructuredDecisionResult result;
    // Byte-level generation may end inside a UTF-8 sequence. Match the provider
    // decoder replacement policy without changing token IDs or physical outputs.
    result.document = Json{
        {"schema", "alpamayo.driving.result.v1"},
        {"coordinate_frame", "current_ego"},
        {"samples",
         output}}.dump(-1, ' ', false, Json::error_handler_t::replace);
    return result;
}
} // namespace trtmc::alpamayo
