#pragma once
#include <cfloat>

namespace kadan::cuda::detail {
// For admitted positive finite E4M3 scales, |FP4 * scale| <= 6*448.
// This conservative power-of-two bound proves the exact product fits FP32,
// without executing double arithmetic per weight. NaN/Inf fail the comparisons.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr bool bounded_nvfp4_global(float global) {
    return global <= FLT_MAX / 4096.0F && global >= -FLT_MAX / 4096.0F;
}
// Exact check before rounding: values beyond FLT_MAX can round to FLT_MAX.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr bool weight_overflows(float local, float global) {
    const double exact = static_cast<double>(local) * static_cast<double>(global);
    return exact > static_cast<double>(FLT_MAX) || exact < -static_cast<double>(FLT_MAX);
}
}
