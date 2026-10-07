#include "kadan/layer_math.hpp"
#include "layer_math_validation.hpp"
#include <cmath>
namespace kadan::math {
namespace {
double sigmoid(double x) {const auto e=std::exp(-std::abs(x));return x>=0?1/(1+e):e/(1+e);}
}
void rms_norm(Norm s,std::span<const float> in,std::span<const float> weight,std::span<float> out) {
    detail::norm(s,in.size(),weight.size(),out.size());detail::output_alias(in,out,true);detail::output_alias(weight,out,false);
    detail::finite(in);detail::finite(weight);
    for(std::size_t row=0;row<s.rows;++row) {
        double squared=0;for(std::size_t c=0;c<s.width;++c) {double x=in[row*s.width+c];squared+=x*x;}
        const double denominator=std::sqrt(squared/s.width+s.epsilon);
        for(std::size_t c=0;c<s.width;++c) {
            const double scale=double(weight[c])+(s.scale==NormScale::one_plus?1:0);
            out[row*s.width+c]=detail::result((double(in[row*s.width+c])/denominator)*scale);
        }
    }
}
void gate(Gate g,std::span<const float> in,std::span<const float> modulation,std::span<float> out) {
    detail::gate(g,in.size(),modulation.size(),out.size());detail::output_alias(in,out,true);detail::output_alias(modulation,out,true);
    detail::finite(in);detail::finite(modulation);
    for(std::size_t i=0;i<in.size();++i) {
        const double z=modulation[i];double factor=sigmoid(z);if(g==Gate::silu) factor*=z;
        out[i]=detail::result(double(in[i])*factor);
    }
}
void inverse_frequencies(double theta,std::span<float> out) {
    detail::require(std::isfinite(theta) && theta>1 && !out.empty() && out.size()<=32768 && out.data(),"math_frequency_shape");
    for(std::size_t i=0;i<out.size();++i) {
        const float f=static_cast<float>(std::pow(theta,-double(i)/out.size()));
        detail::require(std::isfinite(f) && f>0,"math_frequency_range");out[i]=f;
    }
}
void rope(Rotary s,std::size_t position,std::span<const float> freq,std::span<const float> in,std::span<float> out) {
    detail::rotary(s,position,freq.size(),in.size(),out.size());detail::output_alias(in,out,true);detail::output_alias(freq,out,false);
    detail::finite(in);for(float f:freq) detail::require(std::isfinite(f) && f>0 && f<=1,"math_frequency_range");
    const auto half=s.rotary_dim/2;
    for(std::size_t head=0;head<s.heads;++head) {
        const auto base=head*s.head_dim;
        for(std::size_t i=0;i<half;++i) {
            const float angle=static_cast<float>(position)*freq[i];
            const double c=std::cos(angle),t=std::sin(angle),a=in[base+i],b=in[base+half+i];
            out[base+i]=detail::result(a*c-b*t);out[base+half+i]=detail::result(a*t+b*c);
        }
        for(std::size_t i=s.rotary_dim;i<s.head_dim;++i) out[base+i]=in[base+i];
    }
}
} // namespace kadan::math
