#pragma once
#include <array>
#include <string>
#include <vector>

#include "kadan/image_schedule.hpp"

namespace kadan::image {
struct ImageCheckpointConfig {
    std::array<float, 64> mean, stddev;
    std::vector<std::string> text_files;
    Schedule sample_schedule;

    Schedule sampling_schedule(std::size_t tokens, std::size_t steps) const {
        return sample_schedule.steps ? sample_schedule : schedule(tokens, steps);
    }
};

ImageCheckpointConfig validate_checkpoint_config(const std::string& root);
}  // namespace kadan::image
