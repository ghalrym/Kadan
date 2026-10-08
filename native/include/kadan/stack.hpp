#pragma once
#include "kadan/decoder.hpp"
namespace kadan::stack {
inline constexpr std::size_t layers=4;
struct Config {std::array<decoder::Config,layers> layer;std::size_t vocabulary;unsigned eos;float epsilon;};
struct Weights {std::array<decoder::Weights,layers> layer;std::span<const float> embedding,final_norm;quantization::Matrix head;};
struct Selection {unsigned token=0;bool eos=false;bool operator==(const Selection&)const=default;};
struct Plan {
    std::size_t hidden,capacity;std::array<decoder::Plan,layers> layer;std::array<std::size_t,layers> offset;
    std::size_t embedding,final_norm,head_weights,head_scales,embedded,normalized,logits,selected,status,device_bytes,host_numeric_bytes;
};
Plan plan(Config);
void validate_weights(Config,const Weights&);
// Synthetic/native caller-supplied weights only; no tokenizer/checkpoint IO.
// CPU oracle: budget covers numeric storage, caller retains immutable weights.
class Reference {
public:
    Reference(Config,Weights,std::size_t numeric_budget);~Reference();
    Reference(const Reference&)=delete;Reference& operator=(const Reference&)=delete;
    // Prompt feeding may ignore predicted EOS with stop_on_eos=false; all steps
    // still compute selection. A true EOS terminates until reset. Returned EOS
    // describes the prediction regardless of stop_on_eos. Input is a token ID.
    Selection step(unsigned input,bool stop_on_eos=true,const std::atomic_bool* cancelled=nullptr);
    void reset();bool valid()const;bool finished()const;std::size_t tokens()const;
    void read_layer(std::size_t,std::span<float> output)const;
    void read_state(std::size_t,std::span<std::uint8_t>,std::span<std::uint8_t>)const;
    void read_output(std::span<float> normalized,std::span<float> logits)const;
private:struct Impl;std::unique_ptr<Impl> impl_;
};
}
