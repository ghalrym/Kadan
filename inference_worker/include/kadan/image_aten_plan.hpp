#pragma once
#include <algorithm>

#include "kadan/resources.hpp"

namespace kadan::image {
enum class AtenImageStage {
    text,
    denoiser,
    vae
};

// Select one whole-component owner from the worker's admitted devices. No fixed
// device ordering, cross-device weight sharding, or implicit CPU fallback.
inline int aten_stage_device(
    const Snapshot& snapshot, const std::vector<int>& devices, Bytes bytes
) {
    int selected = -1;
    Bytes largest = 0;
    for (int device : devices) {
        if (device < 0 || (std::size_t(device) + 1) >= snapshot.capacity.size())
            throw std::invalid_argument("image_aten_device");
        const auto free = snapshot.capacity[device + 1] - snapshot.used[device + 1];
        if (free >= bytes && (selected < 0 || free > largest)) {
            selected = device;
            largest = free;
        }
    }
    if (selected < 0)
        throw ResourceExhausted("image_aten_component_headroom");
    return selected;
}

inline Bytes aten_stage_workspace(AtenImageStage stage, Bytes points, Bytes tokens) {
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    if (stage == AtenImageStage::text)
        return 2 * gib + tokens * 4096 * 4 * 16;
    if (stage == AtenImageStage::denoiser)
        return 2 * gib + (points + tokens) * 4096 * 2 * 20;
    if (stage == AtenImageStage::vae)
        return 2 * gib + points * 256 * 144 * 4 * 5;
    throw std::invalid_argument("image_aten_stage");
}
}  // namespace kadan::image
