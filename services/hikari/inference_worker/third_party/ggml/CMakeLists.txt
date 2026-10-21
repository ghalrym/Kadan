# Vendored subset of ggml from llama.cpp 443fc4f6d733d5cd52d151e102390c61f25547eb:
# the GGUF reader, type traits and row dequantization (ggml-base), plus the
# x86 quantized dot kernels and row quantizers (ggml-cpu). Upstream sources are
# unmodified except src/ggml-cpu/ggml-cpu.c, which keeps only the CPU type
# traits and their lookup tables. The kernels target the AVX2, FMA and F16C
# baseline.
add_library(ggml-base STATIC src/ggml.c src/gguf.cpp src/ggml-quants.c
  src/ggml-threading.cpp src/ggml-backend-stubs.c)
target_include_directories(ggml-base PUBLIC include PRIVATE src)
target_compile_definitions(ggml-base PRIVATE _GNU_SOURCE _XOPEN_SOURCE=600 GGML_SCHED_MAX_COPIES=4)
set_target_properties(ggml-base PROPERTIES C_STANDARD 11 C_EXTENSIONS ON CXX_STANDARD 17
  CXX_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ggml-base PRIVATE Threads::Threads m)

add_library(ggml-cpu STATIC src/ggml-cpu/ggml-cpu.c src/ggml-cpu/quants.c
  src/ggml-cpu/arch/x86/quants.c src/ggml-cpu/vec.cpp)
target_include_directories(ggml-cpu PUBLIC include PRIVATE src src/ggml-cpu)
target_compile_definitions(ggml-cpu PRIVATE _GNU_SOURCE _XOPEN_SOURCE=600 GGML_SCHED_MAX_COPIES=4
  GGML_SSE42 GGML_AVX GGML_AVX2 GGML_BMI2 GGML_F16C GGML_FMA)
target_compile_options(ggml-cpu PRIVATE -msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2)
set_target_properties(ggml-cpu PROPERTIES C_STANDARD 11 C_EXTENSIONS ON CXX_STANDARD 17
  CXX_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ggml-cpu PUBLIC ggml-base PRIVATE m)
