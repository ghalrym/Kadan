#pragma once

#include <cmath>

namespace kadan::image::detail {
#ifdef __CUDACC__
__host__ __device__
#endif
    inline float silu(float x) {
    double v = x;
    if (v >= 0)
        return float(v / (1 + ::exp(-v)));
    // Keep the negative-tail formula and double intermediates; only reuse exp(v).
    const double exponential = ::exp(v);
    return float(v * exponential / (1 + exponential));
}

#ifdef __CUDACC__
__host__ __device__
#endif
    inline float silu_product(float gate, float up) {
    // Round the double SiLU result before multiplying the FP32 up projection.
    const float activated = silu(gate);
    return activated * up;
}
}  // namespace kadan::image::detail
