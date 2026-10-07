#pragma once
#include "kadan/layer_math.hpp"
#include <cstdint>
#include <cmath>
#include <stdexcept>
namespace kadan::math::detail {
inline void require(bool ok,const char* error) {if(!ok) throw std::invalid_argument(error);}
inline void matrix(std::size_t rows,std::size_t width,std::size_t count) {
    require(rows>0 && rows<=1024 && width>0 && width<=65536 && rows<=1048576/width,"math_shape");
    require(count==rows*width,"math_span");
}
inline void norm(Norm s,std::size_t input,std::size_t weight,std::size_t output) {
    matrix(s.rows,s.width,input);require(output==input && weight==s.width,"math_span");
    require(std::isfinite(s.epsilon) && s.epsilon>0,"math_epsilon");
    require(s.scale==NormScale::direct || s.scale==NormScale::one_plus,"math_scale");
}
inline void gate(Gate g,std::size_t input,std::size_t modulation,std::size_t output) {
    require(input>0 && input<=1048576 && modulation==input && output==input,"math_span");
    require(g==Gate::sigmoid || g==Gate::silu,"math_gate");
}
inline void rotary(Rotary s,std::size_t position,std::size_t freq,std::size_t input,std::size_t output) {
    matrix(s.heads,s.head_dim,input);require(output==input,"math_span");
    require(s.rotary_dim>0 && s.rotary_dim<=s.head_dim && s.rotary_dim%2==0 && freq==s.rotary_dim/2,"math_rotary_shape");
    require(s.context>0 && s.context<=262144 && position<s.context,"math_position");
}
inline bool overlap(const void* a,std::size_t ab,const void* b,std::size_t bb) {
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    require(a && b && ab<=UINTPTR_MAX-x && bb<=UINTPTR_MAX-y,"math_pointer");
    return x<y+bb && y<x+ab;
}
inline void output_alias(std::span<const float> in,std::span<float> out,bool exact) {
    const bool shared=overlap(in.data(),in.size_bytes(),out.data(),out.size_bytes());
    require(!shared || (exact && in.data()==out.data() && in.size()==out.size()),"math_overlap");
}
inline void finite(std::span<const float> values) {for(float x:values) require(std::isfinite(x),"math_nonfinite");}
inline float result(double x) {
    require(std::isfinite(x) && x<=3.40282346638528859812e38 && x>=-3.40282346638528859812e38,"math_result_range");return static_cast<float>(x);
}
} // namespace kadan::math::detail
