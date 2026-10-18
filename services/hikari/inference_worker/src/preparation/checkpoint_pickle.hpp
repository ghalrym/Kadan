#pragma once

#include "kadan/preparation.hpp"

namespace kadan::serving {
// Reads data-only torch.save metadata. Never imports or invokes pickle globals.
Json read_checkpoint_pickle(std::string bytes, const std::atomic_bool& cancel);
}  // namespace kadan::serving
