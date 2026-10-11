#pragma once
#include "kadan/checkpoint.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <functional>

namespace kadan::checkpoint {
inline constexpr std::size_t float_read_buffer_bytes=1024*1024;
// Bounded storage decoding only; no inference and no precision reduction. Both
// scratch and output are admitted/owned by the caller. Notify after each chunk.
template<class Read, class Progress>
void read_floats(Dtype dtype, std::span<float> output, std::span<std::uint8_t> scratch,
                 const std::atomic_bool& cancel, Read read, Progress progress) {
    if(dtype!=Dtype::bf16 && dtype!=Dtype::fp32)throw std::invalid_argument("checkpoint_float_dtype");
    const std::size_t width=dtype==Dtype::bf16?2:4;
    if(scratch.size()<width || scratch.size()>float_read_buffer_bytes)throw std::invalid_argument("checkpoint_float_scratch");
    for(std::size_t at=0;at<output.size();) {
        if(cancel.load())throw std::runtime_error("checkpoint_read_cancelled");
        const auto count=std::min(scratch.size()/width,output.size()-at);
        read(at*width,scratch.first(count*width));
        if(cancel.load())throw std::runtime_error("checkpoint_read_cancelled");
        for(std::size_t i=0;i<count;++i) {
            std::uint32_t bits=0;
            for(std::size_t j=0;j<width;++j)bits|=std::uint32_t(scratch[i*width+j])<<(j*8);
            const auto value=std::bit_cast<float>(width==2?bits<<16:bits);
            if(!std::isfinite(value))throw std::runtime_error("checkpoint_nonfinite_weight");
            output[at+i]=value;
        }
        at+=count;progress(at*width);
    }
}
}
