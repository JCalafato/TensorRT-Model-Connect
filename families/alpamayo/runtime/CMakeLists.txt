# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Native components and the existing structured-decision Task adapter.
add_library(trtmc_alpamayo_action STATIC action_step.cpp trajectory.cpp vision_encoder.cpp vision_preprocessor.cpp history_tokenizer.cpp text_decoder.cpp bpe_tokenizer.cpp driving_pipeline.cpp driving_record.cpp)
target_link_libraries(trtmc_alpamayo_action PUBLIC trtmc_core PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(trtmc_alpamayo_action PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(trtmc_alpamayo_action PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(trtmc_alpamayo_action PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(trtmc_model_alpamayo SHARED plugin.cpp)
target_link_libraries(trtmc_model_alpamayo PRIVATE trtmc_alpamayo_action nlohmann_json::nlohmann_json)
target_compile_options(trtmc_model_alpamayo PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_alpamayo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
install(TARGETS trtmc_model_alpamayo LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_alpamayo_driving_record_test ../tests/cpp/driving_record.cpp)
  target_link_libraries(test_alpamayo_driving_record_test PRIVATE
    trtmc_alpamayo_action nlohmann_json::nlohmann_json)
  target_compile_options(test_alpamayo_driving_record_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME alpamayo_driving_record_test COMMAND test_alpamayo_driving_record_test)
  add_executable(alpamayo_action_step_test ../tests/cpp/action_step.cpp)
  target_link_libraries(alpamayo_action_step_test PRIVATE
    trtmc_alpamayo_action trtmc_backend_trt nlohmann_json::nlohmann_json)
  target_compile_options(alpamayo_action_step_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/edge_llm/Adapter.cmake")
