#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "kadan/gguf_quantization.hpp"
#include "kadan/image_convolution.hpp"
#include "kadan/projection_math.hpp"
#include "kadan/resources.hpp"
#include "kadan/weight_identity.hpp"

namespace kadan {
// Conservative envelope for native CUDA context and library-handle overhead.
inline constexpr Bytes compute_context_bytes = 384ULL * 1024 * 1024;

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

// KV mirrors grow geometrically while the previous buffer is still alive.
// Three times the declared state covers the old and new allocation together.
inline Bytes
native_device_execution_bytes(Bytes limit, Bytes weights, Bytes scratch, Bytes state) {
    auto bytes = std::min(limit, weights);
    bytes += std::min(limit - bytes, scratch);
    for (int allocation = 0; allocation < 3; ++allocation)
        bytes += std::min(limit - bytes, state);
    return bytes;
}

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

    // Future device allocations, including bounded caches and lazy workspace,
    // excluding the caller's CUDA context. Empty means the bound is unknown.
    virtual Footprint execution_memory() const {
        return {};
    }

    // Device bytes per device kept out of the weight bank for request state such
    // as attend_cached mirrors; it applies from the next prepare_weights.
    virtual void reserve_device_state(Bytes) {
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

    // Input rows per job that dense_gguf_batch accepts.
    static constexpr std::size_t gguf_batch_rows = 8;

    // One small-row product of a packed GGUF matrix; no encoding means F32 rows.
    // Bank keys match dense_gguf_source/dense_source, so residency is shared.
    struct GgufBatchJob {
        const WeightIdentity* identity;
        std::optional<gguf::Encoding> encoding;
        const GgufWeightSource* source;
        std::span<const float> input;
        std::size_t in, out;
        std::span<float> output;
        bool retain_misses;
    };

    // Token decoding: every job runs in one synchronous device submission with
    // one input upload and one output download. Unsupported precedes source
    // reads and output writes; callers then project each job separately.
    virtual ProjectionResult
    dense_gguf_batch(std::span<const GgufBatchJob>, const std::atomic_bool&) {
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

    // Causal attention over a host KV cache that the device mirrors per `layer`
    // until release_scratch. Rows are queries at positions position.. whose keys
    // and values are already in keys/values ([position, kv_head, dim], at least
    // position + rows positions). Earlier mirrored positions are reused and
    // rewritten positions are uploaded again. With `selected`, row r attends
    // only positions selected[offsets[r] .. offsets[r + 1]) (none after its own
    // position). False precedes any output write and leaves the caller's
    // attention responsible.
    virtual bool attend_cached(
        std::uint64_t /*layer*/,
        std::span<const float> /*q*/,
        std::span<const float> /*keys*/,
        std::span<const float> /*values*/,
        std::size_t /*rows*/,
        std::size_t /*position*/,
        std::size_t /*heads*/,
        std::size_t /*kv_heads*/,
        std::size_t /*dim*/,
        float /*scale*/,
        std::span<float> /*output*/,
        const std::atomic_bool&,
        std::span<const std::uint32_t> /*selected*/ = {},
        std::span<const std::uint32_t> /*offsets*/ = {}
    ) {
        return false;
    }

    // One gated delta-net step over `rows` in order (llama.cpp delta-net-base):
    // causal depthwise convolution with SiLU over qkv, L2-normalized q and k,
    // then per value head the decayed delta rule, RMSNorm by `norm` and the
    // sigmoid or SiLU gate of z. The convolution history and recurrent state of
    // `layer` stay on the device until release_scratch. `fresh` starts them at
    // zero. False on a fresh step precedes any output write and leaves the
    // caller's host path responsible for the request; later steps throw rather
    // than lose device state.
    struct DeltaStep {
        std::uint64_t layer;
        bool fresh;
        std::span<const float> qkv, z, alpha, beta;
        std::span<const float> conv, dt, a, norm;
        std::size_t rows, key_heads, value_heads, dim, kernel;
        float eps;
        bool sigmoid_gate;
        std::span<float> output;
    };

    virtual bool delta_step(const DeltaStep&, const std::atomic_bool&) {
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
