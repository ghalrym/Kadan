#pragma once
#include <cmath>

namespace kadan::gpt_oss {
// The four selected logits are sorted highest first. Preserve their maximum
// while replacing them with FP32 softmax weights, on either CPU or CUDA.
#ifdef __CUDACC__
__host__ __device__
#endif
    inline void routing_weights(float (&values)[4]) {
    const float maximum = values[0];
    float total = 0;
    for (float& value : values) {
        value = expf(value - maximum);
        total += value;
    }
    for (float& value : values)
        value /= total;
}
}  // namespace kadan::gpt_oss
