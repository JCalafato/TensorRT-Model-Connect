/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tokenizer.h"

namespace trtmc::diffusion_gemma {
std::string decode_response(const ITokenizer& tokenizer, const std::vector<std::int32_t>& ids);
} // namespace trtmc::diffusion_gemma
