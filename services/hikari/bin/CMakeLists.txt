# Developer executables are built only when KADAN_BUILD_TOOLS is enabled.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

if(TARGET kadan_checkpoint)
  add_executable(kadan-checkpoint-inspect cpp/checkpoint_inspect.cpp)
  target_link_libraries(kadan-checkpoint-inspect PRIVATE kadan_checkpoint)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kadan-checkpoint-inspect PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  list(APPEND KADAN_TOOL_TARGETS kadan-checkpoint-inspect)
endif()

add_executable(kadan-prepare-assets cpp/prepare_assets.cpp)
target_link_libraries(kadan-prepare-assets PRIVATE kadan_preparation)
list(APPEND KADAN_TOOL_TARGETS kadan-prepare-assets)
add_custom_target(kadan-tools DEPENDS ${KADAN_TOOL_TARGETS})
