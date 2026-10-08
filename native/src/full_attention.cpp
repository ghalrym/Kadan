#include "kadan/full_attention.hpp"
#include "kadan/linear_attention.hpp"
#include "kadan/cuda_plan.hpp"
#include "layer_math_validation.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace kadan::full {
namespace {
void require(bool x,const char* error){if(!x)throw std::invalid_argument(error);}
float finite(float x){require(std::isfinite(x),"full_nonfinite");return x;}
float bf(float x){return linear::bf16_round(x);}
float expand(std::uint16_t x){return std::bit_cast<float>(std::uint32_t(x)<<16);}
std::uint16_t bits(float x){return std::uint16_t(std::bit_cast<std::uint32_t>(bf(x))>>16);}
void norm(std::span<const float> x,std::span<const float> weight,std::span<float> y,float epsilon){
    float sum=0;for(float v:x)sum=finite(sum+v*v);
    const float inverse=1/std::sqrt(sum/float(x.size())+epsilon);
    for(std::size_t j=0;j<x.size();++j)y[j]=bf((x[j]*inverse)*(1+weight[j]));
}
void project(const quantization::Matrix& w,std::span<const float> x,std::span<float> y){
    for(std::size_t r=0;r<w.rows;++r){double sum=0;for(std::size_t j=0;j<w.columns;++j)sum+=double(quantization::e4m3fn(w.weights[r*w.columns+j])*w.multipliers[0])*x[j];
        require(std::isfinite(sum)&&std::abs(sum)<=std::numeric_limits<float>::max(),"full_nonfinite");y[r]=bf(float(sum));}
}
void rotate(std::span<float> x,std::span<const float> freq,std::size_t position){
    // Eager BF16 tensors: cosine/sine, each product, then sum all round.
    for(std::size_t j=0;j<freq.size();++j){
        const float angle=float(position)*freq[j],c=bf(std::cos(angle)),s=bf(std::sin(angle)),a=x[j],b=x[j+freq.size()];
        x[j]=bf(bf(a*c)-bf(b*s));x[j+freq.size()]=bf(bf(a*s)+bf(b*c));
    }
}
float sigmoid(float x){const float e=std::exp(-std::abs(x));return x>=0?1/(1+e):e/(1+e);}
}
Plan plan(Config c){
    require(c.hidden>0&&c.hidden<=4096&&c.heads>0&&c.heads<=64&&c.kv_heads>0&&c.kv_heads<=c.heads&&c.heads%c.kv_heads==0,"full_shape");
    require(c.head_dim>0&&c.head_dim<=256&&c.rotary_dim>0&&c.rotary_dim<=c.head_dim&&c.rotary_dim%2==0&&c.capacity>0&&c.capacity<=262144,"full_shape");
    require(std::isfinite(c.epsilon)&&c.epsilon>0,"full_epsilon");
    // Bounds above make all products below fit size_t on supported 64-bit hosts.
    static_assert(sizeof(std::size_t)>=8);Plan p{};p.queries=c.heads*c.head_dim;p.kv=c.kv_heads*c.head_dim;p.history_elements=c.capacity*p.kv;p.norm_weights=c.hidden+2*c.head_dim;
    auto region=[&](std::size_t n){auto offset=p.scratch_floats;p.scratch_floats+=n;return offset;};
    p.norm_offset=region(c.hidden);p.qg_offset=region(2*p.queries);p.k_offset=region(p.kv);p.v_offset=region(p.kv);
    p.q_offset=region(p.queries);p.gate_offset=region(p.queries);p.prob_offset=region(c.heads*c.capacity);
    p.core_offset=region(p.queries);p.gated_offset=region(p.queries);p.out_offset=region(c.hidden);
    p.host_state_workspace_bytes=4*p.history_elements+4*p.scratch_floats;return p;
}
void validate_weights(Config c,const Weights& w){
    auto p=plan(c);auto projection=[](const quantization::Matrix& m,std::size_t rows,std::size_t cols){
        require(m.encoding==quantization::Encoding::modelopt_fp8&&m.rows==rows&&m.columns==cols&&m.multipliers.size()==1,"full_projection_shape");(void)cuda::plan_fp8(m,SIZE_MAX);
    };
    projection(w.q_gate,2*p.queries,c.hidden);projection(w.key,p.kv,c.hidden);projection(w.value,p.kv,c.hidden);projection(w.out,c.hidden,p.queries);
    auto weights=[](std::span<const float> x,std::size_t n){require(x.size()==n,"full_weight_shape");for(float v:x)require(bf(v)==v,"full_weight_bf16");};
    weights(w.input_norm,c.hidden);weights(w.query_norm,c.head_dim);weights(w.key_norm,c.head_dim);
    require(w.frequencies.size()==c.rotary_dim/2,"full_frequency_shape");
    for(float v:w.frequencies)require(std::isfinite(v)&&v>0&&v<=1,"full_frequency_range");
}
struct Reference::Impl {
    Config c;Weights w;Plan p;StateCursor cursor;std::vector<std::uint16_t> keys,values;std::vector<float> scratch;
    Impl(Config shape,Weights weights,std::size_t budget):c(shape),w(weights),p(plan(c)),cursor(1,c.capacity){
        validate_weights(c,w);require(p.host_state_workspace_bytes<=budget,"full_host_budget");keys.resize(p.history_elements);values.resize(p.history_elements);scratch.resize(p.scratch_floats);
    }
    std::span<float> span(std::size_t offset,std::size_t n){return std::span(scratch).subspan(offset,n);}
    std::span<const float> diagnostic(std::size_t offset,std::size_t n){require(cursor.valid()&&cursor.committed_tokens()>0,"full_diagnostic_unavailable");return span(offset,n);}
    void step(std::span<const float> input,std::span<float> output){
        auto token=cursor.begin();const auto position=cursor.committed_tokens();
        try{
            require(input.size()==c.hidden&&output.size()==c.hidden,"full_input_shape");math::detail::output_alias(input,output,true);
            for(float x:input)require(bf(x)==x,"full_input_bf16");
            auto normalized=span(p.norm_offset,c.hidden),qg=span(p.qg_offset,2*p.queries),k=span(p.k_offset,p.kv),v=span(p.v_offset,p.kv);
            auto q=span(p.q_offset,p.queries),gate=span(p.gate_offset,p.queries),prob=span(p.prob_offset,c.heads*c.capacity);
            auto core=span(p.core_offset,p.queries),gated=span(p.gated_offset,p.queries),out=span(p.out_offset,c.hidden);
            norm(input,w.input_norm,normalized,c.epsilon);project(w.q_gate,normalized,qg);project(w.key,normalized,k);project(w.value,normalized,v);
            for(std::size_t h=0;h<c.heads;++h){
                norm(qg.subspan(h*2*c.head_dim,c.head_dim),w.query_norm,q.subspan(h*c.head_dim,c.head_dim),c.epsilon);
                std::copy_n(qg.begin()+(h*2+1)*c.head_dim,c.head_dim,gate.begin()+h*c.head_dim);
                rotate(q.subspan(h*c.head_dim,c.head_dim),w.frequencies,position);
            }
            for(std::size_t h=0;h<c.kv_heads;++h){norm(k.subspan(h*c.head_dim,c.head_dim),w.key_norm,k.subspan(h*c.head_dim,c.head_dim),c.epsilon);rotate(k.subspan(h*c.head_dim,c.head_dim),w.frequencies,position);}
            for(std::size_t j=0;j<p.kv;++j){keys[position*p.kv+j]=bits(k[j]);values[position*p.kv+j]=bits(v[j]);}
            const float scale=1/std::sqrt(float(c.head_dim));
            for(std::size_t h=0;h<c.heads;++h){
                auto row=prob.subspan(h*c.capacity,c.capacity);std::fill(row.begin(),row.end(),0);float maximum=-std::numeric_limits<float>::infinity();
                const auto kh=h/(c.heads/c.kv_heads);
                for(std::size_t t=0;t<=position;++t){float dot=0;for(std::size_t j=0;j<c.head_dim;++j)dot=finite(dot+q[h*c.head_dim+j]*expand(keys[t*p.kv+kh*c.head_dim+j]));
                    row[t]=bf(bf(dot)*scale);maximum=std::max(maximum,row[t]);}
                float denominator=0;for(std::size_t t=0;t<=position;++t){row[t]=std::exp(row[t]-maximum);denominator=finite(denominator+row[t]);}
                for(std::size_t t=0;t<=position;++t)row[t]=bf(row[t]/denominator);
                for(std::size_t j=0;j<c.head_dim;++j){float sum=0;for(std::size_t t=0;t<=position;++t)sum=finite(sum+row[t]*expand(values[t*p.kv+kh*c.head_dim+j]));
                    const auto at=h*c.head_dim+j;core[at]=bf(sum);gated[at]=bf(core[at]*bf(sigmoid(gate[at])));}
            }
            project(w.out,gated,out);for(std::size_t j=0;j<c.hidden;++j)out[j]=bf(input[j]+out[j]);
            cursor.written(token,0);cursor.commit(token);std::copy(out.begin(),out.end(),output.begin());
        }catch(...){cursor.invalidate();throw;}
    }
};
Reference::Reference(Config c,Weights w,std::size_t budget):impl_(std::make_unique<Impl>(c,w,budget)){}
Reference::~Reference()=default;
void Reference::step(std::span<const float> x,std::span<float> y){impl_->step(x,y);}
void Reference::reset(){std::fill(impl_->keys.begin(),impl_->keys.end(),0);std::fill(impl_->values.begin(),impl_->values.end(),0);std::fill(impl_->scratch.begin(),impl_->scratch.end(),0);impl_->cursor.reset();}
bool Reference::valid()const{return impl_->cursor.valid();}
std::size_t Reference::tokens()const{return impl_->cursor.committed_tokens();}
std::span<const std::uint16_t> Reference::keys()const{return impl_->keys;}
std::span<const std::uint16_t> Reference::values()const{return impl_->values;}
std::span<const float> Reference::probabilities()const{return impl_->diagnostic(impl_->p.prob_offset,impl_->c.heads*impl_->c.capacity);}
std::span<const float> Reference::core()const{return impl_->diagnostic(impl_->p.core_offset,impl_->p.queries);}
std::span<const float> Reference::gated()const{return impl_->diagnostic(impl_->p.gated_offset,impl_->p.queries);}
} // namespace kadan::full
