#pragma once
#include "kadan/resources.hpp"

namespace kadan::serving {
// Detached reusable state; contains no model weights, device allocations or files.
class InferenceCache {
public:
    virtual ~InferenceCache() = default;
    virtual Bytes bytes() const = 0;
};
}  // namespace kadan::serving
