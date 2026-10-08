#pragma once
#include "kadan/model.hpp"
#include "kadan/resources.hpp"
#include "kadan/stack.hpp"
#include "kadan/cuda_error.hpp"
namespace kadan::cuda {
struct ModelOptions {
    std::size_t capacity=128,metadata_bytes=256*1024*1024,staging_bytes=1024*1024;
    std::size_t device_headroom=512*1024*1024;
};
// Generalized checkpoint-backed successor to the four-layer synthetic Stack.
// One host+device reservation, one arena and one cursor across 1..256 layers.
// All experts resident; no cache, cross-device execution or production IPC yet.
class Model {
public:
    Model(const char* root,ModelOptions,int device,std::shared_ptr<Resources>,const std::atomic_bool* cancelled=nullptr);
    ~Model();Model(const Model&)=delete;Model&operator=(const Model&)=delete;
    // Conservative fixed control/fd/allocator overhead separate from PMR quotas.
    static constexpr std::size_t control_headroom=1024*1024;
    static std::size_t host_bytes(ModelOptions);
    stack::Selection step(unsigned token,bool stop_on_eos=true,const std::atomic_bool* cancelled=nullptr);
    void reset();void close();bool valid()const;bool finished()const;std::size_t tokens()const;
    std::size_t vocabulary()const;std::size_t device_bytes()const;
    // Diagnostic copy of the last committed logits. Failure invalidates reuse;
    // caller must discard the entire destination on exception.
    void read_logits(std::span<float>);
    void read_state(std::size_t layer,std::span<std::uint8_t>,std::span<std::uint8_t>);
private:struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
