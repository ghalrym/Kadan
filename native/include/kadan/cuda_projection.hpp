#pragma once
#include "kadan/cuda_plan.hpp"
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
} // namespace kadan::cuda
