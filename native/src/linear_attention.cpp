#include "kadan/linear_attention.hpp"
#include "kadan/cuda_plan.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace kadan::linear {
namespace {
void require(bool ok,const char* why){if(!ok)throw std::invalid_argument(why);}
std::size_t add(std::size_t a,std::size_t b){require(b<=SIZE_MAX-a,"linear_overflow");return a+b;}
std::size_t mul(std::size_t a,std::size_t b){require(!b||a<=SIZE_MAX/b,"linear_overflow");return a*b;}
float finite(float x){require(std::isfinite(x),"linear_nonfinite");return x;}
float sig(float x){const float e=std::exp(-std::abs(x));return x>=0?1/(1+e):e/(1+e);}
float expand(std::uint16_t x){return std::bit_cast<float>(std::uint32_t(x)<<16);}
std::uint16_t store(float x){return std::uint16_t(std::bit_cast<std::uint32_t>(bf16_round(x))>>16);}
void norm(std::span<const float> x,std::span<const float> weights,std::span<float> out,float epsilon,bool offset){
    float sum=0;for(float value:x)sum=finite(sum+value*value);
    const float inverse=1/std::sqrt(sum/x.size()+epsilon);
    for(std::size_t j=0;j<x.size();++j) {
        const float n=finite(x[j]*inverse);
        // Offset norm rounds only after scaling; direct gated norm rounds both
        // the normalized activation and the subsequent BF16 weight product.
        out[j]=bf16_round(offset?n*(1+weights[j]):bf16_round(n)*weights[j]);
    }
}
void project(const quantization::Matrix& w,std::span<const float> x,std::span<float> out,bool bf16_weights){
    for(std::size_t row=0;row<w.rows;++row){double sum=0;for(std::size_t j=0;j<w.columns;++j){
        const float weight=quantization::e4m3fn(w.weights[row*w.columns+j])*w.multipliers[0];sum+=double(bf16_weights?bf16_round(weight):weight)*x[j];}
        require(std::isfinite(sum)&&std::abs(sum)<=std::numeric_limits<float>::max(),"linear_nonfinite");out[row]=bf16_round(float(sum));}
}
void dense(std::span<const float> w,std::span<const float> x,std::span<float> out){
    for(std::size_t row=0;row<out.size();++row){float sum=0;for(std::size_t j=0;j<x.size();++j)sum=finite(sum+w[row*x.size()+j]*x[j]);out[row]=bf16_round(sum);}
}
}
float bf16_round(float value){
    finite(value);auto bits=std::bit_cast<std::uint32_t>(value);bits+=0x7fffu+((bits>>16)&1u);bits&=0xffff0000u;return finite(std::bit_cast<float>(bits));
}
Plan plan(Config c){
    require(c.hidden>0&&c.hidden<=4096&&c.key_heads>0&&c.key_heads<=64&&c.value_heads>0&&c.value_heads<=64&&c.value_heads%c.key_heads==0,"linear_shape");
    require(c.key_dim>0&&c.key_dim<=256&&c.value_dim>0&&c.value_dim<=256&&c.conv_kernel>0&&c.conv_kernel<=16&&c.capacity>0&&c.capacity<=262144,"linear_shape");
    require(std::isfinite(c.epsilon)&&c.epsilon>0,"linear_epsilon");Plan p{};
    p.keys=mul(c.key_heads,c.key_dim);p.values=mul(c.value_heads,c.value_dim);p.channels=add(mul(2,p.keys),p.values);
    p.conv_elements=mul(p.channels,c.conv_kernel);p.recurrent_elements=mul(mul(c.value_heads,c.key_dim),c.value_dim);
    auto region=[&](std::size_t n){auto at=p.scratch_floats;p.scratch_floats=add(at,n);return at;};
    p.norm_offset=region(c.hidden);p.qkv_offset=region(p.channels);p.z_offset=region(p.values);p.a_offset=region(c.value_heads);p.b_offset=region(c.value_heads);
    p.core_offset=region(p.values);p.gate_offset=region(p.values);p.out_offset=region(c.hidden);
    p.aux_elements=add(add(add(c.hidden,p.conv_elements),mul(2,mul(c.value_heads,c.hidden))),add(mul(2,c.value_heads),c.value_dim));
    p.host_state_workspace_bytes=add(mul(p.conv_elements,2),mul(add(p.recurrent_elements,p.scratch_floats),4));return p;
}
void validate_weights(Config c,const Weights& w){
    const auto p=plan(c);
    auto projection=[](const quantization::Matrix& m,std::size_t rows,std::size_t cols){
        require(m.encoding==quantization::Encoding::modelopt_fp8&&m.rows==rows&&m.columns==cols&&m.multipliers.size()==1,"linear_projection_shape");(void)cuda::plan_fp8(m,SIZE_MAX);
    };
    projection(w.qkv,p.channels,c.hidden);projection(w.z,p.values,c.hidden);projection(w.out,c.hidden,p.values);
    const auto aux=[](std::span<const float> x,std::size_t n){require(x.size()==n,"linear_weight_shape");for(float v:x)require(bf16_round(v)==v,"linear_weight_bf16");};
    aux(w.input_norm,c.hidden);aux(w.conv,p.conv_elements);aux(w.a,c.value_heads*c.hidden);aux(w.b,c.value_heads*c.hidden);
    aux(w.a_log,c.value_heads);aux(w.dt_bias,c.value_heads);aux(w.output_norm,c.value_dim);
}
struct Reference::Impl {
    Config c;Weights w;Plan p;StateCursor cursor;std::vector<std::uint16_t> conv;std::vector<float> state,scratch;
    Impl(Config config,Weights weights,std::size_t budget):c(config),w(weights),p(plan(c)),cursor(1,c.capacity){
        validate_weights(c,w);require(p.host_state_workspace_bytes<=budget,"linear_host_budget");conv.resize(p.conv_elements);state.resize(p.recurrent_elements);scratch.resize(p.scratch_floats);
    }
    std::span<float> span(std::size_t offset,std::size_t n){return std::span(scratch).subspan(offset,n);}
    void step(std::span<const float> input,std::span<float> output){
        auto token=cursor.begin();
        try {
            require(input.size()==c.hidden&&output.size()==c.hidden,"linear_input_shape");for(float x:input)require(bf16_round(x)==x,"linear_input_bf16");
            auto normalized=span(p.norm_offset,c.hidden),qkv=span(p.qkv_offset,p.channels),z=span(p.z_offset,p.values);
            auto a=span(p.a_offset,c.value_heads),b=span(p.b_offset,c.value_heads),core=span(p.core_offset,p.values),gated=span(p.gate_offset,p.values),projected=span(p.out_offset,c.hidden);
            norm(input,w.input_norm,normalized,c.epsilon,true);project(w.qkv,normalized,qkv,c.bf16_weights);project(w.z,normalized,z,c.bf16_weights);dense(w.a,normalized,a);dense(w.b,normalized,b);
            for(std::size_t channel=0;channel<p.channels;++channel){
                auto start=channel*c.conv_kernel;for(std::size_t j=1;j<c.conv_kernel;++j)conv[start+j-1]=conv[start+j];conv[start+c.conv_kernel-1]=store(qkv[channel]);
                float sum=0;for(std::size_t j=0;j<c.conv_kernel;++j)sum=finite(sum+expand(conv[start+j])*w.conv[start+j]);
                const float rounded=bf16_round(sum);qkv[channel]=bf16_round(rounded*sig(rounded));
            }
            // Normalize Q/K once per original key head; each pair of value heads
            // consumes the same normalized vectors but owns a distinct state.
            for(std::size_t head=0;head<c.key_heads;++head){
                float qs=0,ks=0;for(std::size_t j=0;j<c.key_dim;++j){float q=qkv[head*c.key_dim+j],k=qkv[p.keys+head*c.key_dim+j];qs=finite(qs+q*q);ks=finite(ks+k*k);}
                const float qi=1/std::sqrt(qs+1e-6f),ki=1/std::sqrt(ks+1e-6f);
                for(std::size_t j=0;j<c.key_dim;++j){qkv[head*c.key_dim+j]=(qkv[head*c.key_dim+j]*qi)/std::sqrt(float(c.key_dim));qkv[p.keys+head*c.key_dim+j]*=ki;}
            }
            for(std::size_t head=0;head<c.value_heads;++head){
                const float t=finite(a[head]+w.dt_bias[head]);const float soft=std::max(t,0.0f)+std::log1p(std::exp(-std::abs(t)));
                const float g=finite(-std::exp(w.a_log[head])*soft),decay=std::exp(g),beta=bf16_round(sig(b[head]));
                const auto kh=head/(c.value_heads/c.key_heads);const auto base=head*c.key_dim*c.value_dim;
                for(std::size_t v=0;v<c.value_dim;++v){
                    float prediction=0;for(std::size_t k=0;k<c.key_dim;++k){auto& s=state[base+k*c.value_dim+v];s=finite(s*decay);prediction=finite(prediction+s*qkv[p.keys+kh*c.key_dim+k]);}
                    const float delta=finite((qkv[2*p.keys+head*c.value_dim+v]-prediction)*beta);float result=0;
                    for(std::size_t k=0;k<c.key_dim;++k){auto& s=state[base+k*c.value_dim+v];s=finite(s+qkv[p.keys+kh*c.key_dim+k]*delta);result=finite(result+s*qkv[kh*c.key_dim+k]);}
                    core[head*c.value_dim+v]=bf16_round(result);
                }
                norm(core.subspan(head*c.value_dim,c.value_dim),w.output_norm,gated.subspan(head*c.value_dim,c.value_dim),c.epsilon,false);
            }
            for(std::size_t j=0;j<p.values;++j)gated[j]=bf16_round(gated[j]*(z[j]*sig(z[j])));
            project(w.out,gated,projected,c.bf16_weights);
            for(std::size_t j=0;j<c.hidden;++j)projected[j]=bf16_round(input[j]+projected[j]);
            cursor.written(token,0);cursor.commit(token);std::copy(projected.begin(),projected.end(),output.begin());
        }catch(...){cursor.invalidate();throw;}
    }
};
Reference::Reference(Config c,Weights w,std::size_t budget):impl_(std::make_unique<Impl>(c,w,budget)){}
Reference::~Reference()=default;
void Reference::step(std::span<const float> x,std::span<float> y){impl_->step(x,y);}
void Reference::reset(){std::fill(impl_->conv.begin(),impl_->conv.end(),0);std::fill(impl_->state.begin(),impl_->state.end(),0);impl_->cursor.reset();}
bool Reference::valid()const{return impl_->cursor.valid();}
std::size_t Reference::tokens()const{return impl_->cursor.committed_tokens();}
std::span<const std::uint16_t> Reference::convolution()const{return impl_->conv;}
std::span<const float> Reference::recurrent()const{return impl_->state;}
std::span<const float> Reference::core()const{return impl_->span(impl_->p.core_offset,impl_->p.values);}
std::span<const float> Reference::gated()const{return impl_->span(impl_->p.gate_offset,impl_->p.values);}
} // namespace kadan::linear
