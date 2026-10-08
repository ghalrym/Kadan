#pragma once
#include "kadan/moe.hpp"
#include "kadan/cuda_error.hpp"
#include "kadan/resources.hpp"
namespace kadan::cuda {
// Resident layer arena: ONE ledger handle and ONE device allocation, independent
// of expert count. Shared scratch; no projection owners or per-forward allocation.
// Construction uploads caller-supplied weights; never opens checkpoint files.
class Moe {
public:
    Moe(moe::Config config,const moe::Weights& weights,int device,std::shared_ptr<Resources> resources);
    ~Moe();Moe(const Moe&)=delete;Moe& operator=(const Moe&)=delete;
    static std::size_t host_metadata_bytes(); // charged RAM for the fixed owner/descriptor table
    // Creating-thread/current-SM86 device, blocking legacy stream. Caller pins
    // admitted BF16-in-float input/output; exact alias allowed. Discard output on
    // failure. Quarantine requires retention until successful close/context end.
    void forward_device(std::span<const float> input,std::span<float> output);
    void reset(); // zero scratch after numerical/input failure; runtime poison requires close
    bool valid() const;
    void read_routes(std::span<unsigned> selected,std::span<float> logits,std::span<float> probabilities,std::span<float> top_weights);
    void read_outputs(std::span<float> routed,std::span<float> shared,std::span<float> result);
    // Success frees the charged descriptor object; idempotent afterward, valid()
    // is false and execution/diagnostics reject. Uncertain cleanup retains the
    // object and RAM/VRAM reservation without automatic retry.
    void close();
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
