# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_diffusion_gemma SHARED
  bpe_tokenizer.cpp
  image.cpp
  native_runner.cpp
  pipeline.cpp
  plugin.cpp
  request.cpp
  response.cpp
  sampler.cpp
)
target_include_directories(trtmc_model_diffusion_gemma PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_diffusion_gemma SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_diffusion_gemma PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_diffusion_gemma PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_diffusion_gemma PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_diffusion_gemma LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(diffusion_gemma_inference_test ../tests/cpp/run_model.cpp)
  target_link_libraries(diffusion_gemma_inference_test PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  target_include_directories(diffusion_gemma_inference_test PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  add_dependencies(trtmc_model_diffusion_gemma diffusion_gemma_inference_test)
  add_executable(test_diffusion_gemma_response
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/test_response.cpp
    response.cpp bpe_tokenizer.cpp)
  target_include_directories(test_diffusion_gemma_response PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_diffusion_gemma_response PRIVATE
    trtmc_core nlohmann_json::nlohmann_json)
  add_test(NAME test_diffusion_gemma_response COMMAND test_diffusion_gemma_response)
  add_executable(test_diffusion_gemma_request
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/test_request.cpp
    request.cpp
  )
  target_include_directories(test_diffusion_gemma_request PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_diffusion_gemma_request PRIVATE nlohmann_json::nlohmann_json)
  add_test(NAME test_diffusion_gemma_request COMMAND test_diffusion_gemma_request)
  add_executable(test_diffusion_gemma_sampler
    ${PROJECT_SOURCE_DIR}/families/diffusion_gemma/tests/cpp/test_sampler.cpp
    sampler.cpp
  )
  target_include_directories(test_diffusion_gemma_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_diffusion_gemma_sampler COMMAND test_diffusion_gemma_sampler)
endif()
