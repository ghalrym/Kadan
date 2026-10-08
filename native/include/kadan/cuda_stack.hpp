#pragma once
#include "kadan/stack.hpp"
#include "kadan/resources.hpp"
#include "kadan/cuda_error.hpp"
namespace kadan::cuda {
class Stack {
public:
    Stack(stack::Config,const stack::Weights&,int device,std::shared_ptr<Resources>);~Stack();
    Stack(const Stack&)=delete;Stack&operator=(const Stack&)=delete;
    static std::size_t host_metadata_bytes();
    // One model-owned arena; no borrowed device activations or caller output to
    // consume on failure. Selection returns only after whole-token publication.
    // Same creating-thread/current-SM86, blocking legacy-stream contract.
    stack::Selection step(unsigned input,bool stop_on_eos=true,const std::atomic_bool* cancelled=nullptr);
    void reset();bool valid()const;bool finished()const;std::size_t tokens()const;
    void read_layer(std::size_t,std::span<float>);
    void read_state(std::size_t,std::span<std::uint8_t>,std::span<std::uint8_t>);
    void read_output(std::span<float> normalized,std::span<float> logits);
    // Uncertain recovery quarantines the entire pinned model arena. Failed close
    // retains reservation and never retries automatically; success frees metadata.
    void close();
private:struct Impl;std::unique_ptr<Impl> impl_;
};
}
