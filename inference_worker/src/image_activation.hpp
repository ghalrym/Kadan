#pragma once

#include <cmath>

namespace kadan::image::detail {
inline float silu(float x) {
    double v = x;
    if (v >= 0)
        return float(v / (1 + std::exp(-v)));
    // Keep the negative-tail formula and double intermediates; only reuse exp(v).
    const double exponential = std::exp(v);
    return float(v * exponential / (1 + exponential));
}
}  // namespace kadan::image::detail
