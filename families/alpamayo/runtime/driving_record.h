/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "driving_pipeline.h"
#include "trtmc/task.h"

#include <string>
#include <vector>

namespace trtmc::alpamayo {

// The existing structured-decision Task carries the family's versioned record
// and decoded cameras. No model state is hidden in a generic robot action vector.
struct DrivingRecord {
    DrivingRecord() = default;
    DrivingRecord(const DrivingRecord&) = delete;
    DrivingRecord& operator=(const DrivingRecord&) = delete;
    DrivingRecord(DrivingRecord&&) = default;
    DrivingRecord& operator=(DrivingRecord&&) = default;
    std::vector<std::string> ids;
    std::vector<std::size_t> sample_indices;
    std::vector<int64_t> history_end_us;
    std::vector<std::vector<uint8_t>> pixels;
    std::vector<std::vector<DrivingImageInput>> images;
    std::vector<DrivingHistoryInput> histories;
    std::vector<float> initial_noise;
    DrivingOptions options;

    Tensor noise();
};

// Entire request admission is CPU-only. Reject invalid records before mutating
// any engine state. Pixel ownership remains valid after moving the record.
DrivingRecord prepare_driving_request(const StructuredDecisionRequest& request,
                                      const DrivingOptions& defaults, std::size_t max_batch);
StructuredDecisionResult format_driving_result(const DrivingRecord& record,
                                               const std::vector<DrivingSample>& samples);

} // namespace trtmc::alpamayo
