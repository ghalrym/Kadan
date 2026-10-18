#pragma once
#include <span>

#include "checkpoint_pickle.hpp"

namespace kadan::serving {
// Numerical checkpoint conversion only; no model execution.
void fold_codec_weight_norm(std::span<float> row, float gain);
}  // namespace kadan::serving
