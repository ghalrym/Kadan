#pragma once
#include "kadan/cuda_plan.hpp"
#include "kadan/cuda_error.hpp"
#include "kadan/resources.hpp"
#include <memory>
#include <span>

namespace kadan::cuda {
// Explicit GPU-executing API; construction uploads weights. Never instantiate
// during CPU-only validation. Caller selects the device first. All methods and
// destruction belong to the creating thread with that same current device.
// This first path uses the legacy default stream and blocks at each boundary.
class Nvfp4Projection {
public:
    Nvfp4Projection(const quantization::Matrix& host, int device,
                    std::shared_ptr<Resources> admitted_resources);
    ~Nvfp4Projection();
    Nvfp4Projection(const Nvfp4Projection&) = delete;
    Nvfp4Projection& operator=(const Nvfp4Projection&) = delete;
    Nvfp4Projection(Nvfp4Projection&&) = delete;
    Nvfp4Projection& operator=(Nvfp4Projection&&) = delete;
    const Nvfp4Plan& plan() const;
    // Caller-owned RAM spans; no hidden host or dense-weight allocation.
    // Output must be discarded if an exception is thrown. Input/output may alias
    // because input upload and kernel completion precede the output copy.
    void matvec(std::span<const float> input, std::span<float> output);
    // Explicit close reports cleanup failures. Failed/uncertain cleanup retains
    // the resource reservation; no automatic retry or device reset is attempted.
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Same ownership/runtime contract as Nvfp4Projection. FP8 scalar/per-row scales
// and all decoded-weight range checks are validated once at admission. Host
// spans must remain immutable through construction, then may be released.
class Fp8Projection {
public:
    Fp8Projection(const quantization::Matrix& host, int device,
                    std::shared_ptr<Resources> admitted_resources);
    ~Fp8Projection();
    Fp8Projection(const Fp8Projection&) = delete;
    Fp8Projection& operator=(const Fp8Projection&) = delete;
    Fp8Projection(Fp8Projection&&) = delete;
    Fp8Projection& operator=(Fp8Projection&&) = delete;
    const Fp8Plan& plan() const;
    // Caller-owned RAM spans; no hidden host or dense-weight allocation.
    // Output must be discarded if an exception is thrown. Input/output may alias
    // because input upload and kernel completion precede the output copy.
    void matvec(std::span<const float> input, std::span<float> output);
    // Explicit close reports cleanup failures. Failed/uncertain cleanup retains
    // the resource reservation; no automatic retry or device reset is attempted.
    // Caller-admitted device buffers; activations never pass through host RAM.
    // Blocking legacy-stream boundary, including status verification. Caller pins
    // both spans until return, supplies disjoint aligned buffers on this device,
    // and discards output after errors. DeviceBufferQuarantine overrides the
    // return boundary: retain borrowed buffers until close succeeds/context ends.
    // Other errors return only after quiescence or before work was enqueued.
    // No allocation/provenance inference.
    void matvec_device(std::span<const float> input,std::span<float> output);
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
