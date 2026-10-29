/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "driving_record.h"

#include "action_step.h"

#include <functional>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using namespace trtmc;
using namespace trtmc::alpamayo;

void check(bool valid, const char* message) {
    if (!valid)
        throw std::runtime_error(message);
}

DrivingOptions defaults() {
    DrivingOptions result{};
    result.action_space = {64, 16, 0.1, 1e-6, 1e-4, 0, 1, 0, 1};
    result.reasoning.max_new_tokens = 128;
    return result;
}

StructuredDecisionRequest request() {
    StructuredDecisionRequest result;
    Json xyz = Json::array(), rotations = Json::array(), times = Json::array();
    for (int i = 0; i < 16; ++i) {
        xyz.push_back({-1.5F + i * 0.1F, 0.0F, 0.0F});
        rotations.push_back({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        times.push_back(100000 * i);
    }
    Json cameras = Json::array();
    for (const auto index : {0, 1, 2, 6}) {
        ImageResult video;
        video.height = video.width = 2;
        video.num_frames = 4;
        for (int frame = 0; frame < 4; ++frame)
            video.pixels.insert(video.pixels.end(), 12, static_cast<float>(index * 10 + frame));
        cameras.push_back({{"camera_index", index},
                           {"video_index", result.videos.size()},
                           {"timestamps_us", {1200000, 1300000, 1400000, 1500000}}});
        result.videos.push_back(std::move(video));
    }
    result.document =
        Json{{"schema", "alpamayo.driving.v1"},
             {"coordinate_frame", "current_ego"},
             {"observations",
              {{{"id", "clip"},
                {"cameras", cameras},
                {"history", {{"xyz", xyz}, {"rotations", rotations}, {"timestamps_us", times}}}}}}}
            .dump();
    return result;
}

void modify(StructuredDecisionRequest& value, const std::function<void(Json&)>& edit) {
    auto document = Json::parse(value.document);
    edit(document);
    value.document = document.dump();
}

void rejects(const std::function<void(StructuredDecisionRequest&)>& edit) {
    auto invalid = request();
    edit(invalid);
    bool rejected = false;
    try {
        auto ignored = prepare_driving_request(invalid, defaults(), 6);
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "invalid request was not rejected");
    check(prepare_driving_request(request(), defaults(), 6).histories.size() == 1,
          "invalid request poisoned later admission");
}
} // namespace

int main() {
    try {
        const auto blocked = std::numeric_limits<float>::lowest();
        const auto mask =
            make_action_prefix_mask({{blocked, blocked, 0}, {0, -7, 0}}, {4, 6}, 5, 2);
        check(mask ==
                  std::vector<float>({blocked, blocked, 0,  0, blocked, 0,  0, blocked, blocked, 0,
                                      0,       blocked, 0,  0, 0,       -7, 0, 0,       0,       0,
                                      0,       0,       -7, 0, 0,       0,  0, 0}),
              "action handoff lost padding, custom exclusions, or future-token visibility");
        for (const auto& invalid : std::vector<std::function<void()>>{
                 [] { make_action_prefix_mask({}, {3}, 3, 2); },
                 [] { make_action_prefix_mask({{}}, {3}, 3, 2); },
                 [] { make_action_prefix_mask({{0, 0, 0, 0}}, {3}, 3, 2); },
                 [] { make_action_prefix_mask({{0, 0}}, {1}, 3, 2); },
                 [] { make_action_prefix_mask({{0}}, {0}, 3, 2); },
                 [] { make_action_prefix_mask({{0}}, {5}, 3, 2); },
                 [] { make_action_prefix_mask({{0}, {0, 0}}, {3, 3}, 3, 2); },
                 [] { make_action_prefix_mask({{1}}, {3}, 3, 2); },
                 [] {
                     make_action_prefix_mask({{std::numeric_limits<float>::quiet_NaN()}}, {3}, 3,
                                             2);
                 },
                 [] { make_action_prefix_mask({{0}}, {3}, INT64_MAX, 2); }}) {
            bool rejected = false;
            try {
                invalid();
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected, "malformed action mask handoff was accepted");
        }
        auto input = request();
        auto record = prepare_driving_request(input, defaults(), 1);
        check(record.histories.size() == 1 && record.images[0].size() == 16,
              "wrong modality count");
        for (std::size_t camera = 0; camera < 4; ++camera) {
            const auto index = std::array<int, 4>{0, 1, 2, 6}[camera];
            for (std::size_t frame = 0; frame < 4; ++frame)
                check(record.images[0][camera * 4 + frame].rgb[0] == index * 10 + frame,
                      "camera or temporal order changed");
        }
        const auto noise = record.initial_noise;
        auto moved = std::move(record);
        check(moved.images[0][15].rgb[0] == 63, "move invalidated owned camera pixels");
        check(moved.noise().shape == std::vector<int64_t>{1, 64, 2}, "invalid noise shape");
        check(prepare_driving_request(input, defaults(), 1).initial_noise == noise,
              "seeded replay is not deterministic");
        modify(input, [](Json& document) { document["num_samples"] = 6; });
        auto six = prepare_driving_request(input, defaults(), 6);
        check(six.histories.size() == 6 && six.ids[5] == "clip" && six.sample_indices[5] == 5,
              "six trajectories lost observation identity");
        modify(input, [](Json& document) {
            document["initial_noise"] = Json::array();
            for (int row = 0; row < 6; ++row) {
                Json values = Json::array();
                for (int i = 0; i < 64; ++i)
                    values.push_back({0.25F, -0.5F});
                document["initial_noise"].push_back(values);
            }
        });
        auto explicit_noise = prepare_driving_request(input, defaults(), 6);
        check(explicit_noise.initial_noise[766] == 0.25F &&
                  explicit_noise.initial_noise[767] == -0.5F,
              "explicit replay noise changed");
        std::vector<DrivingSample> samples(6);
        for (auto& sample : samples) {
            sample.generated_tokens = {7, 11};
            sample.text = {"Keep clear of the work zone.", "", ""};
            sample.normalized_actions.assign(128, 0.125F);
            sample.trajectory.xyz.assign(192, 0.75F);
            sample.trajectory.rotations.resize(576);
            for (int i = 0; i < 64; ++i)
                for (int j = 0; j < 3; ++j)
                    sample.trajectory.rotations[9 * i + 4 * j] = 1;
            sample.trajectory.initial_velocity = 1.0F;
        }
        const auto output = Json::parse(format_driving_result(six, samples).document);
        const auto& last = output.at("samples").at(5);
        check(last.at("trajectory").at("xyz").size() == 64 &&
                  last.at("trajectory").at("rotations").at(63).at(2).at(2) == 1 &&
                  last.at("trajectory").at("timestamps_us").at(63) == 7900000 &&
                  last.at("normalized_acceleration_curvature").at(0).at(0) == 0.125F &&
                  last.at("tokens") == Json({7, 11}),
              "result lost typed physical or text output");
        // Generated byte-BPE text may end inside a Unicode code point. These
        // expectations match tokenizers.decoders.ByteLevel replacement behavior.
        for (const auto& [bytes, expected] :
             std::vector<std::pair<std::string, std::string>>{{"\xe6", "\xef\xbf\xbd"},
                                                              {"\xe6\xb5", "\xef\xbf\xbd"},
                                                              {"\xe6X", "\xef\xbf\xbdX"},
                                                              {"\xe6\xb5\x8b", "\xe6\xb5\x8b"}}) {
            samples[0].text.cot = bytes;
            const auto serialized = Json::parse(format_driving_result(six, samples).document);
            check(serialized["samples"][0]["reasoning"]["cot"] == expected &&
                      serialized["samples"][0]["tokens"] == Json({7, 11}) &&
                      serialized["samples"][0]["trajectory"] == output["samples"][0]["trajectory"],
                  "UTF-8 replacement changed tokens or physical output");
        }
        rejects([](auto& r) { r.images.push_back({}); });
        rejects([](auto& r) { r.max_state_tokens = 10; });
        rejects([](auto& r) { r.videos[0].num_frames = 3; });
        rejects([](auto& r) { r.videos[0].pixels[0] = 0.5F; });
        rejects([](auto& r) { r.videos[0].pixels[0] = std::numeric_limits<float>::quiet_NaN(); });
        rejects([](auto& r) { r.videos[0].height = 0; });
        rejects([](auto& r) { r.videos.pop_back(); });
        for (const auto& edit : std::vector<std::function<void(Json&)>>{
                 [](Json& j) { j["schema"] = "unknown"; },
                 [](Json& j) { j["coordinate_frame"] = "world"; },
                 [](Json& j) { j["num_samples"] = 7; }, [](Json& j) { j["num_samples"] = 1.5; },
                 [](Json& j) { j["unknown"] = true; },
                 [](Json& j) { j["observations"] = Json::array(); },
                 [](Json& j) { j["observations"][0]["cameras"][0]["camera_index"] = 6; },
                 [](Json& j) { j["observations"][0]["cameras"][1]["video_index"] = 0; },
                 [](Json& j) { j["observations"][0]["history"]["xyz"][0] = {1, 2}; },
                 [](Json& j) { j["observations"][0]["history"]["timestamps_us"][0] = 100001; },
                 [](Json& j) { j["sampling"] = {{"seed", -1}}; },
                 [](Json& j) { j["sampling"] = {{"top_p", 0}}; },
                 [](Json& j) { j["sampling"] = {{"max_new_tokens", 129}}; },
                 [](Json& j) { j["sampling"] = {{"denoising_steps", 1}}; },
                 [](Json& j) { j["initial_noise"] = {0}; }})
            rejects([&](auto& r) { modify(r, edit); });
        std::cout << "Driving record: B1/B6, media order, owned moves, seeded/explicit noise, "
                     "physical poses, 22 invalid requests and recovery passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
