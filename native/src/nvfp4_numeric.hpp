#pragma once
#include <cfloat>

namespace kadan::cuda::detail {
// local = FP4 * E4M3 is exact in FP32. Multiplying its at most seven
// significant bits by the FP32 global scale is exact in double. Check before
// rounding: an out-of-range exact value can still round to finite FLT_MAX.
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr bool weight_overflows(float local, float global) {
    const double exact = static_cast<double>(local) * static_cast<double>(global);
    return exact > static_cast<double>(FLT_MAX) || exact < -static_cast<double>(FLT_MAX);
}
}
