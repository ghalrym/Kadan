#pragma once

#include "kadan/inference_request.hpp"

namespace kadan::serving {
// One request may span several denoising batches. Completed output files belong
// to that request even if a later batch fails for another request.
struct ImageBatchOutput {
    std::size_t count;
    std::vector<std::filesystem::path> paths;
    Json failure;

    void fail_if_incomplete(const Json& error) {
        if (paths.size() != count && failure.is_null())
            failure = error;
    }
};
}  // namespace kadan::serving
