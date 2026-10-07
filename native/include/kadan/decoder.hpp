#pragma once
#include "kadan/full_attention.hpp"
#include "kadan/linear_attention.hpp"
#include "kadan/moe.hpp"
#include <atomic>
namespace kadan::decoder {
enum class Attention { linear, full };
struct Config { Attention attention; linear::Config linear{}; full::Config full{}; moe::Config moe{}; };
struct Weights { linear::Weights linear{}; full::Weights full{}; moe::Weights moe{}; std::span<const float> post_norm; };
struct ProjectionLayout { std::size_t weights,scale,rows,columns; };
struct Plan {
    std::size_t hidden,capacity;float epsilon;
    linear::Plan linear{};full::Plan full{};moe::Plan moe{};
    std::array<ProjectionLayout,4> projections{};std::size_t projection_count;
    std::size_t auxiliary,frequencies,post_norm,moe_offset,state_first,state_second,state_first_bytes,state_second_bytes;
    std::size_t attention_scratch,workspace,status,device_bytes,host_numeric_bytes;
};
Plan plan(Config);
void validate_weights(Config,const Weights&);
// CPU composition oracle. Caller retains immutable weights; budget covers numeric
// state/scratch, not the oracle's control objects. No per-step allocation.
// Only the coordinator's token count is public; child progress is private.
class Reference {
public:
    Reference(Config,Weights,std::size_t numeric_budget);
    ~Reference();Reference(const Reference&)=delete;Reference& operator=(const Reference&)=delete;
    void step(std::span<const float>,std::span<float>,const std::atomic_bool* cancelled=nullptr);
    void reset();bool valid()const;std::size_t tokens()const;
    void read_state(std::span<std::uint8_t> first,std::span<std::uint8_t> second)const;
    // Last successful complete decoder step only.
    void read_intermediates(std::span<float> attention,std::span<float> normalized,std::span<float> mixture)const;
private:struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace kadan::decoder
