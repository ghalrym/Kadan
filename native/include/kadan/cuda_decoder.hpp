#pragma once
#include "kadan/decoder.hpp"
#include "kadan/resources.hpp"
#include "kadan/cuda_error.hpp"
namespace kadan::cuda {
// One resident decoder arena, one admission, one publication authority. Borrowed
// internal producers never reserve/free/commit. No checkpoint loading.
class Decoder {
public:
    Decoder(decoder::Config,const decoder::Weights&,int device,std::shared_ptr<Resources>);
    ~Decoder();Decoder(const Decoder&)=delete;Decoder& operator=(const Decoder&)=delete;
    static std::size_t host_metadata_bytes();
    // Creating thread/current SM86 device, legacy stream. Caller pins admitted
    // exact BF16-in-float spans until return; exact alias is allowed. Cancellation
    // flag must remain alive. Discard output on ANY failure, even a final copy.
    // Quarantine requires keeping both spans until successful close/context end.
    void step_device(std::span<const float>,std::span<float>,const std::atomic_bool* cancelled=nullptr);
    bool valid()const;std::size_t tokens()const;
    void reset(); // numerical/cancellation failure only; runtime poison needs close
    void read_intermediates(std::span<float> attention,std::span<float> normalized,std::span<float> mixture);
    void read_state(std::span<std::uint8_t> first,std::span<std::uint8_t> second);
    void close(); // success frees charged host object immediately; failure retains charge
private:struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::cuda
