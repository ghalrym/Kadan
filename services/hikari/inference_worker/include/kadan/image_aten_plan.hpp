#pragma once
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kadan/image_vae_tiles.hpp"
#include "kadan/resources.hpp"

namespace kadan::image {
enum class AtenImageStage {
    text,
    denoiser,
    vae,
    vae_encoder
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

struct AtenStagePlan {
    int device;
    bool stream_weights;
    bool tile_vae;
    Bytes host_bytes;
    Bytes device_bytes;
};

// Keep a whole component resident when it fits. Otherwise stream one operator's
// weights; two largest tensors cover a weight plus bias or dtype conversion.
inline AtenStagePlan aten_stage_plan(
    const Snapshot& snapshot,
    const std::vector<int>& devices,
    Bytes weights,
    Bytes largest,
    Bytes workspace,
    Bytes tiled_workspace = 0
) {
    constexpr Bytes mib = 1024ULL * 1024;
    const auto host = 2 * largest + 256 * mib;
    if (host > snapshot.capacity[0] - snapshot.used[0])
        throw ResourceExhausted("image_aten_host_headroom");
    const auto fit = [&](Bytes scratch, bool tiled) -> AtenStagePlan {
        try {
            return {
                aten_stage_device(snapshot, devices, weights + scratch),
                false,
                tiled,
                host,
                weights + scratch
            };
        } catch (const ResourceExhausted&) {
            return {
                aten_stage_device(snapshot, devices, 2 * largest + scratch),
                true,
                tiled,
                host,
                2 * largest + scratch
            };
        }
    };
    try {
        return fit(workspace, false);
    } catch (const ResourceExhausted&) {
        if (!tiled_workspace)
            throw;
        return fit(tiled_workspace, true);
    }
}

// CPU tensors retained across sequential stages: encoded reference latents
// (128 FP32 channels per cell) and Qwen3 stacked hidden states (7680 FP32).
inline Bytes aten_reference_host_bytes(Bytes reference_points, Bytes text_tokens) {
    return reference_points * 128 * 4 + text_tokens * 7680 * 4;
}

inline Bytes aten_stage_workspace(
    AtenImageStage stage,
    Bytes points,
    Bytes tokens,
    bool editing = false,
    Bytes reference_points = 0
) {
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    if (stage == AtenImageStage::text)
        return 2 * gib + tokens * 2560 * 4 * 16;
    if (stage == AtenImageStage::denoiser) {
        const auto edit_buffers = editing ? points * (8 * 128 + 2) * 4 : 0;
        const auto sequence = points + reference_points + tokens;
        const auto reference_latent = reference_points * 128 * 4;
        // Double-stream and single-stream activations (dim=3072, inner=9216,
        // fused QKV+MLP=27648) plus cuBLAS/SDPA scratch.
        return 2 * gib + sequence * 3072 * 2 * 24 + edit_buffers + reference_latent;
    }
    if (stage == AtenImageStage::vae_encoder) {
        // AutoencoderKLFlux2 2D encoder: 128 channels at full image resolution.
        const auto largest_activation = points * 256 * 128 * 4;
        return 2 * gib + 6 * largest_activation;
    }
    if (stage == AtenImageStage::vae) {
        // AutoencoderKLFlux2 2D decoder: 128 channels at full image resolution.
        const auto largest_activation = points * 256 * 128 * 4;
        return 2 * gib + 4 * largest_activation;
    }
    throw std::invalid_argument("image_aten_stage");
}

// Components remain sequential. Only denoising retains a batch of latent
// images and activations; fixed library scratch and loaded weights are shared.
inline Bytes aten_batch_workspace(
    Bytes points, Bytes tokens, Bytes reference_points, std::size_t images
) {
    if (!images)
        throw std::invalid_argument("image_batch_empty");
    constexpr Bytes fixed = 2ULL * 1024 * 1024 * 1024;
    const auto one = aten_stage_workspace(
        AtenImageStage::denoiser, points, tokens, false, reference_points
    );
    if (images > (std::numeric_limits<Bytes>::max() - fixed) / (one - fixed))
        throw std::overflow_error("image_batch_workspace");
    return fixed + images * (one - fixed);
}

struct AtenDenoiserLane {
    std::size_t begin;
    std::size_t images;
    AtenStagePlan stage;
};

// Independent replicas split images, never one image's weights or device budget.
// Reference/text encoding and decoding stay sequential outside these lanes.
inline std::vector<AtenDenoiserLane> aten_denoiser_plan(
    const Snapshot& snapshot,
    const std::vector<int>& devices,
    Bytes weights,
    Bytes largest,
    Bytes points,
    Bytes tokens,
    Bytes reference_points,
    std::size_t images
) {
    const auto workspace = [&](std::size_t count) {
        return aten_batch_workspace(points, tokens, reference_points, count);
    };
    const auto single = [&] {
        return std::vector<AtenDenoiserLane>{
            {0,
             images,
             aten_stage_plan(snapshot, devices, weights, largest, workspace(images))}
        };
    };
    if (images <= 1)
        return single();
    constexpr Bytes mib = 1024ULL * 1024;
    const auto host = 2 * largest + 256 * mib;
    const auto host_lanes = (snapshot.capacity[0] - snapshot.used[0]) / host;

    struct Candidate {
        int device;
        Bytes free;
        std::size_t capacity;
        std::size_t images = 0;
    };

    std::vector<Candidate> candidates;
    std::vector<int> seen;
    for (int device : devices) {
        if (device < 0 || std::size_t(device) + 1 >= snapshot.capacity.size()
            || std::find(seen.begin(), seen.end(), device) != seen.end())
            throw std::invalid_argument("image_aten_device");
        seen.push_back(device);
        const auto free = snapshot.capacity[device + 1] - snapshot.used[device + 1];
        std::size_t capacity = 0;
        for (std::size_t count = 1; count <= images; ++count) {
            if (weights > free || workspace(count) > free - weights)
                break;
            capacity = count;
        }
        if (capacity)
            candidates.push_back({device, free, capacity});
    }
    std::stable_sort(
        candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.capacity != b.capacity ? a.capacity > b.capacity : a.free > b.free;
        }
    );
    candidates.resize(
        std::min<std::size_t>(candidates.size(), std::min<Bytes>(images, host_lanes))
    );
    for (std::size_t assigned = 0; assigned < images; ++assigned) {
        auto selected = candidates.end();
        for (auto lane = candidates.begin(); lane != candidates.end(); ++lane)
            if (lane->images < lane->capacity
                && (selected == candidates.end() || lane->images < selected->images))
                selected = lane;
        if (selected == candidates.end())
            return single();
        ++selected->images;
    }
    std::vector<AtenDenoiserLane> lanes;
    lanes.reserve(candidates.size());
    std::size_t begin = 0;
    for (const auto& lane : candidates) {
        lanes.push_back(
            {begin,
             lane.images,
             {lane.device, false, false, host, weights + workspace(lane.images)}}
        );
        begin += lane.images;
    }
    return lanes;
}

// Tiling bounds convolutional activations. Global bottleneck attention still
// runs on the full latent grid (512 channels at 2x latent resolution).
inline Bytes aten_tiled_vae_workspace(AtenImageStage stage, Bytes points) {
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    if (stage == AtenImageStage::vae) {
        constexpr Bytes side = vae_tile_core + 2 * vae_decode_halo;
        const auto tile_points = std::min(points, side * side);
        return 2 * gib + 4 * tile_points * 256 * 128 * 4 + 10 * points * 4 * 512 * 4;
    }
    if (stage == AtenImageStage::vae_encoder) {
        constexpr Bytes side = vae_tile_core + 2 * vae_encode_halo;
        const auto tile_points = std::min(points, side * side);
        return 2 * gib + 6 * tile_points * 256 * 128 * 4 + 10 * points * 4 * 512 * 4;
    }
    return 0;
}

}  // namespace kadan::image
