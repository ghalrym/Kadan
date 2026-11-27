#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "kadan/image_schedule.hpp"

namespace kadan::image {
struct ImageCheckpointConfig {
    std::vector<std::string> text_files;
    std::vector<std::string> denoiser_files;
    float batch_norm_eps = 1e-4f;
    std::size_t default_steps = 4;

    Schedule sampling_schedule(std::size_t tokens, std::size_t steps) const {
        return schedule(tokens, steps ? steps : default_steps);
    }
};

ImageCheckpointConfig validate_checkpoint_config(const std::string& root);
}  // namespace kadan::image
