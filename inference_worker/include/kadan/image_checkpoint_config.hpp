#pragma once
#include <array>
#include <string>

namespace kadan::image {
struct ImageCheckpointConfig {
    std::array<float, 64> mean, stddev;
};

ImageCheckpointConfig validate_checkpoint_config(const std::string& root);
}  // namespace kadan::image
