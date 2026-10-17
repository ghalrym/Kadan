#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "kadan/gguf_quantization.hpp"
#include "kadan/image_convolution.hpp"
#include "kadan/projection_math.hpp"
#include "kadan/resources.hpp"
#include "kadan/weight_identity.hpp"

namespace kadan {
#ifdef __CUDACC__
__host__ __device__
#endif
    inline bool attention_visible(
        std::size_t query,
        std::size_t key,
        std::size_t causal_queries,
        std::size_t window,
        std::size_t offset
    ) {
    return (query >= causal_queries || key <= query + offset)
           && (!window || key + window > query + offset);
}

inline bool projection_split_columns(std::size_t rows, std::size_t devices) {
    return rows < devices;
}

struct DeviceWeightPlacement {
    int device;
    Bytes planned_bytes, allocated_bytes;
};

// Per-call result: finite certifies every stored post-bias output, after all
// partitions synchronize. Unsupported leaves the caller's fallback responsible.
enum class ProjectionResult {
    unsupported,
    unvalidated,
    finite
};

inline bool
projection_output_finite(ProjectionResult result, std::span<const float> output) {
    return result == ProjectionResult::finite
           || std::all_of(output.begin(), output.end(), [](float v) {
                  return std::isfinite(v);
              });
}

// Synchronous projection boundary. Caller owns admitted host spans until return;
// implementations admit staging/device memory and synchronize before returning.
class DenseCompute {
public:
    virtual ~DenseCompute() = default;

    // Request boundary: release reusable scratch before reporting completion.
    virtual void release_scratch() {
    }

    // Dedicated worker only: synchronize and destroy its process-local contexts.
    virtual void release_devices() {
    }

    // Retire only these immutable tensor identities at a synchronous stage boundary.
    // Keep device contexts, execution threads and other retained weights.
    virtual Bytes release_weights(std::span<const WeightIdentity>) {
        return 0;
    }

    virtual Footprint retained_weights() const {
        return {};
    }

    // Weight-bank limits and live allocation bytes, excluding context and scratch.
    virtual std::vector<DeviceWeightPlacement> weight_placement() const {
        return {};
    }

    // Header-derived execution bytes, before any weight placement. True means the
    // conservative full inventory fits; false selects bounded RAM-backed streaming.
    virtual bool prepare_weights(Bytes) {
        return false;
    }

    // Stable serialized identity opts immutable weights into bounded retention.
    virtual void dense_weight(
        const WeightIdentity&,
        std::span<const float> weights,
        std::span<const float> bias,
        std::span<const float> input,
        std::size_t in,
        std::size_t out,
        std::span<float> output,
        const std::atomic_bool& cancel,
        bool precise = false
    ) {
        dense(weights, bias, input, in, out, output, cancel, precise);
    }

    // The callback reads only a missing partition; it must preserve exact F32 values.
    virtual bool supports_weight_sources() const {
        return false;
    }

    virtual bool supports_bf16_weights() const {
        return false;
    }

    using Bf16WeightSource = std::function<void(std::size_t, std::span<std::uint16_t>)>;

    // Synchronous planar [channel,point] image VAE normalization and optional SiLU.
    // False means unsupported before changing output. Failures throw; callers
    // must not retry a partially completed operation through the CPU fallback.
    virtual bool image_vae_normalize(
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::span<float>,
        bool,
        const std::atomic_bool&
    ) {
        return false;
    }

    // Synchronous bias-free image gate/up pair. False means unsupported before
    // reading either source or changing output. Execution failures throw; callers
    // must never retry a partially completed operation through the fallback.
    virtual bool image_silu_pair_bf16_source(
        const WeightIdentity&,
        const Bf16WeightSource&,
        const WeightIdentity&,
        const Bf16WeightSource&,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&
    ) {
        return false;
    }

    virtual ProjectionResult dense_bf16_source(
        const WeightIdentity&,
        const Bf16WeightSource&,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&
    ) {
        return ProjectionResult::unsupported;
    }

    // Whole packed rows; first is a matrix row, not an element offset. Each
    // device may invoke this concurrently with an independently admitted buffer.
    using GgufWeightSource = std::function<void(std::size_t, std::span<std::byte>)>;

    // Unsupported precedes source reads or output writes. Packed weights retain
    // their serialized identity; implementations validate decoded values and
    // synchronize before return. Execution failures throw and must not retry.
    // Disabling miss retention keeps existing hits and streams uncached tiles.
    // Source callbacks may run concurrently on disjoint whole rows. All calls
    // finish before this operation returns or throws.
    virtual ProjectionResult dense_gguf_source(
        const WeightIdentity&,
        gguf::Encoding,
        const GgufWeightSource&,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool = false,
        bool = true
    ) {
        return ProjectionResult::unsupported;
    }

    using FloatWeightSource = std::function<void(std::size_t, std::span<float>)>;

    // F32 VAE stride-one CHW convolution. False precedes source reads/output
    // writes; execution errors throw and must never retry through CPU packing.
    virtual bool image_vae_convolution_source(
        const WeightIdentity&,
        const FloatWeightSource&,
        std::span<const float>,
        std::span<const float>,
        image::VaeConvolution,
        std::span<float>,
        const std::atomic_bool&
    ) {
        return false;
    }

    virtual ProjectionResult dense_source(
        const WeightIdentity&,
        const FloatWeightSource&,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool = false,
        ProjectionPolicy = ProjectionPolicy::standard
    ) {
        return ProjectionResult::unsupported;
    }

    // Return false to retain the model's CPU attention. Q/K/V use [token,head,dim].
    // Only the first causal_queries query rows are causal. window=0 is unbounded;
    // query_offset addresses cached decoding where queries are a suffix of keys.
    virtual bool attend(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&
    ) {
        return false;
    }

    virtual void dense(
        std::span<const float> weights,
        std::span<const float> bias,
        std::span<const float> input,
        std::size_t in,
        std::size_t out,
        std::span<float> output,
        const std::atomic_bool& cancel,
        bool precise = false
    ) = 0;
};

std::shared_ptr<DenseCompute>
    cuda_dense_compute(std::shared_ptr<Resources>, std::vector<int>, Workload);
}  // namespace kadan
