# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_cosmos3_edge SHARED plugin.cpp tokenizer.cpp media.cpp sampler.cpp)
target_include_directories(trtmc_model_cosmos3_edge PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_cosmos3_edge SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_cosmos3_edge PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_cosmos3_edge PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_cosmos3_edge PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_cosmos3_edge LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

include("${CMAKE_CURRENT_LIST_DIR}/edge_llm/Adapter.cmake")

if(TRTMC_BUILD_TESTS)
  add_executable(cosmos3_edge_inference_test ../tests/cpp/run_model.cpp)
  target_link_libraries(cosmos3_edge_inference_test PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  # CI builds the owning family target, not every test executable in the tree.
  add_dependencies(trtmc_model_cosmos3_edge cosmos3_edge_inference_test)
  target_include_directories(cosmos3_edge_inference_test PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  add_executable(test_cosmos3_edge_media
    ../tests/cpp/test_media.cpp media.cpp)
  target_include_directories(test_cosmos3_edge_media PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  add_test(NAME test_cosmos3_edge_media COMMAND test_cosmos3_edge_media)
endif()
