#pragma once
#include "kadan/checkpoint.hpp"
#include <limits>
#include <stdexcept>

namespace kadan::checkpoint {
inline std::uint64_t weight_bytes_add(std::uint64_t a,std::uint64_t b){
    if(b>std::numeric_limits<std::uint64_t>::max()-a)throw std::overflow_error("weight_inventory_overflow");
    return a+b;
}
inline std::uint64_t weight_bytes_multiply(std::uint64_t a,std::uint64_t b){
    if(b&&a>std::numeric_limits<std::uint64_t>::max()/b)throw std::overflow_error("weight_inventory_overflow");
    return a*b;
}
inline std::size_t storage_width(Dtype dtype){
    switch(dtype){
        case Dtype::u8:case Dtype::i8:case Dtype::fp8:return 1;
        case Dtype::bf16:case Dtype::fp16:return 2;
        case Dtype::fp32:return 4;
    }
    throw std::invalid_argument("weight_inventory_dtype");
}
inline std::string_view storage_name(Dtype dtype){
    switch(dtype){
        case Dtype::u8:return "U8";case Dtype::i8:return "I8";case Dtype::fp8:return "F8_E4M3";
        case Dtype::bf16:return "BF16";case Dtype::fp16:return "F16";case Dtype::fp32:return "F32";
    }
    throw std::invalid_argument("weight_inventory_dtype");
}
// The caller specifies the loader representation, never infers quantization from
// file size. F32 is the existing image/H3 floating-weight loader representation.
// Raw preserves packed/quantized formats including their separate scale tensors.
enum class WeightRepresentation { raw, f32 };
inline std::uint64_t execution_weight_bytes(const TensorInfo& tensor,WeightRepresentation representation){
    if(tensor.rank>tensor.shape.size())throw std::invalid_argument("weight_inventory_rank");
    std::uint64_t elements=1;
    for(std::size_t i=0;i<tensor.rank;++i)elements=weight_bytes_multiply(elements,tensor.shape[i]);
    if(weight_bytes_multiply(elements,storage_width(tensor.dtype))!=tensor.bytes)
        throw std::invalid_argument("weight_inventory_shape");
    if(representation==WeightRepresentation::raw)return tensor.bytes;
    if(tensor.dtype!=Dtype::fp32&&tensor.dtype!=Dtype::bf16&&tensor.dtype!=Dtype::fp16)
        throw std::invalid_argument("weight_inventory_float_representation");
    return weight_bytes_multiply(elements,4);
}
struct WeightInventory {
    struct Group {std::uint64_t tensors=0,stored_bytes=0;};
    std::array<Group,6> precision{};
    std::uint64_t stored_bytes=0,floating_f32_bytes=0,raw_nonfloating_bytes=0;
    void add(const TensorInfo& tensor){
        const auto raw=execution_weight_bytes(tensor,WeightRepresentation::raw);
        const auto index=static_cast<std::size_t>(tensor.dtype);
        auto next=*this;
        next.precision.at(index).tensors=weight_bytes_add(next.precision.at(index).tensors,1);
        next.precision.at(index).stored_bytes=weight_bytes_add(next.precision.at(index).stored_bytes,raw);
        next.stored_bytes=weight_bytes_add(next.stored_bytes,raw);
        if(tensor.dtype==Dtype::fp32||tensor.dtype==Dtype::bf16||tensor.dtype==Dtype::fp16)
            next.floating_f32_bytes=weight_bytes_add(next.floating_f32_bytes,execution_weight_bytes(tensor,WeightRepresentation::f32));
        else next.raw_nonfloating_bytes=weight_bytes_add(next.raw_nonfloating_bytes,raw);
        *this=next;
    }
    void add(const Shard& shard){
        shard.check_unchanged();auto next=*this;
        for(std::size_t i=0;i<shard.tensor_count();++i)next.add(shard.tensor_at(i));
        shard.check_unchanged();*this=next;
    }
};
}
