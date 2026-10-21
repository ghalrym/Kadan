#pragma once
#include <variant>
#include <vector>

namespace kadan::text {
struct AttentionState {
    std::vector<float> keys, values;
};

struct RecurrentState {
    std::vector<float> convolution, recurrent;
    bool device = false;
};

struct QsaState {
    AttentionState attention;
    std::vector<float> raw_keys, pooled;
};

struct ModelState {
    std::vector<std::variant<AttentionState, RecurrentState, QsaState>> layers;
    std::vector<float> ple;
    std::vector<unsigned> recent;
};
}  // namespace kadan::text
