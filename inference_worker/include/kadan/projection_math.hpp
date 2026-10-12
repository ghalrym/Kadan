#pragma once
#include "kadan/cleanup_error.hpp"
#include "kadan/resources.hpp"

namespace kadan {
enum class ProjectionPolicy {
    standard,
    image_feed_forward
};

inline bool projection_uses_tf32(
    ProjectionPolicy policy, Workload workload, int compute_major, bool precise
) {
    return policy == ProjectionPolicy::image_feed_forward && workload == Workload::image
           && compute_major >= 8 && !precise;
}

// Math mode belongs to the persistent device-thread handle. Restore it before
// checking cancellation or using that handle for attention/another projection.
// Callbacks keep this state transition testable without CUDA or model execution.
template <class Mode, class Get, class Set, class Run>
void with_projection_math(bool tf32, Mode mode, Get get, Set set, Run run) {
    if (!tf32) {
        run();
        return;
    }
    const auto previous = get();
    try {
        set(mode);
        run();
    } catch (...) {
        rethrow_after_cleanup(std::current_exception(), [&] { set(previous); });
    }
    set(previous);
}
}  // namespace kadan
