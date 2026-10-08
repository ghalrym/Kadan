#include "kadan/moe.hpp"
#include "kadan/linear_attention.hpp"
#include "kadan/cuda_plan.hpp"
#include "layer_math_validation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace kadan::moe {
namespace {
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
float finite(float x){require(std::isfinite(x),"moe_nonfinite");return x;}
float bf(float x){return linear::bf16_round(x);}
std::size_t align(std::size_t x){return (x+255)&~std::size_t{255};}
float sigmoid(float x){const float e=std::exp(-std::abs(x));return x>=0?1/(1+e):e/(1+e);}
float dense(std::span<const float> w,std::span<const float> x){float total=0;for(std::size_t j=0;j<x.size();++j)total=finite(total+w[j]*x[j]);return bf(total);}
void project(const quantization::Matrix& w,std::span<const float> x,std::span<float> y){
    for(std::size_t r=0;r<w.rows;++r){double sum=0;for(std::size_t j=0;j<w.columns;++j){
        auto packed=w.weights[r*(w.columns/2)+j/2];auto code=(packed>>(4*(j%2)))&15;
        const double decoded=double(quantization::e2m1(code))*quantization::e4m3fn(w.block_scales[r*(w.columns/16)+j/16])*w.multipliers[0];
        require(std::isfinite(decoded)&&std::abs(decoded)<=std::numeric_limits<float>::max(),"moe_nonfinite_weight");sum+=double(float(decoded))*x[j];}
        require(std::isfinite(sum)&&std::abs(sum)<=std::numeric_limits<float>::max(),"moe_nonfinite");y[r]=bf(float(sum));}
}
}
Plan plan(Config c){
    require(c.hidden>0&&c.hidden<=4096&&c.hidden%16==0&&c.experts>0&&c.experts<=256&&c.top_k>0&&c.top_k<=c.experts&&c.top_k<=8,"moe_shape");
    require(c.intermediate>0&&c.intermediate<=4096&&c.intermediate%16==0&&c.shared_intermediate>0&&c.shared_intermediate<=4096&&c.shared_intermediate%16==0,"moe_shape");
    static_assert(sizeof(std::size_t)>=8);Plan p{};const auto width=std::max(c.intermediate,c.shared_intermediate);
    auto scratch=[&](std::size_t n){auto at=p.scratch_floats;p.scratch_floats+=n;return at;};
    p.logits=scratch(c.experts);p.probabilities=scratch(c.experts);p.top_weights=scratch(c.top_k);
    p.gate=scratch(width);p.up=scratch(width);p.activation=scratch(width);p.down=scratch(c.hidden);
    p.accumulator=scratch(c.hidden);p.shared=scratch(c.hidden);p.result=scratch(c.hidden);p.shared_factor=scratch(1);
    p.host_workspace_bytes=p.scratch_floats*4+c.top_k*sizeof(unsigned);
    auto layout=[&](std::size_t middle){ExpertLayout e{};std::size_t end=0;auto region=[&](std::size_t n){auto at=end;end=align(end+n);return at;};
        e.gate_weights=region(middle*c.hidden/2);e.gate_scales=region(middle*c.hidden/16);e.up_weights=region(middle*c.hidden/2);e.up_scales=region(middle*c.hidden/16);
        e.down_weights=region(c.hidden*middle/2);e.down_scales=region(c.hidden*middle/16);e.bytes=end;return e;};
    p.routed_layout=layout(c.intermediate);p.shared_layout=layout(c.shared_intermediate);
    std::size_t end=0;auto region=[&](std::size_t n){auto at=end;end=align(end+n);return at;};
    p.router_offset=region(c.experts*c.hidden*2);p.shared_gate_offset=region(c.hidden*2);p.experts_offset=region(c.experts*p.routed_layout.bytes);p.shared_offset=region(p.shared_layout.bytes);
    p.scratch_offset=region(p.scratch_floats*4);p.indices_offset=region(c.top_k*sizeof(unsigned));p.status_offset=region(sizeof(unsigned));p.device_bytes=end;return p;
}
void validate_weights(Config c,const Weights& w){
    (void)plan(c);require(w.router.size()==c.experts*c.hidden&&w.shared_gate.size()==c.hidden&&w.experts.size()==c.experts,"moe_weight_shape");
    for(auto values:{w.router,w.shared_gate})for(float v:values)require(bf(v)==v,"moe_weight_bf16");
    auto projection=[](const quantization::Matrix& m,std::size_t rows,std::size_t columns){require(m.encoding==quantization::Encoding::modelopt_nvfp4&&m.rows==rows&&m.columns==columns,"moe_projection_shape");(void)cuda::plan_nvfp4(m,SIZE_MAX);
        // Reject unrepresentable decoded weights at admission without dense allocation.
        for(std::size_t r=0;r<rows;++r)for(std::size_t j=0;j<columns;++j){auto packed=m.weights[r*(columns/2)+j/2];auto code=(packed>>(4*(j%2)))&15;
            double value=double(quantization::e2m1(code))*quantization::e4m3fn(m.block_scales[r*(columns/16)+j/16])*m.multipliers[0];require(std::abs(value)<=std::numeric_limits<float>::max(),"moe_nonfinite_weight");}
    };
    auto expert=[&](const Expert& e,std::size_t middle){projection(e.gate,middle,c.hidden);projection(e.up,middle,c.hidden);projection(e.down,c.hidden,middle);};
    for(const auto& e:w.experts)expert(e,c.intermediate);
    expert(w.shared,c.shared_intermediate);
}
struct Reference::Impl {
    Config c;Weights w;Plan p;std::vector<float> scratch;std::vector<unsigned> selected;bool invalid=false,ready=false;
    Impl(Config shape,Weights weights,std::size_t budget):c(shape),w(weights),p(plan(c)){validate_weights(c,w);require(p.host_workspace_bytes<=budget,"moe_host_budget");scratch.resize(p.scratch_floats);selected.resize(c.top_k);}
    std::span<float> span(std::size_t offset,std::size_t size){return std::span(scratch).subspan(offset,size);}
    std::span<const float> diagnostic(std::size_t offset,std::size_t n){require(!invalid&&ready,"moe_diagnostic_unavailable");return span(offset,n);}
    void expert(const Expert& e,std::size_t middle,std::span<const float> x){
        auto gate=span(p.gate,middle),up=span(p.up,middle),activation=span(p.activation,middle),down=span(p.down,c.hidden);
        project(e.gate,x,gate);project(e.up,x,up);
        for(std::size_t j=0;j<middle;++j)activation[j]=bf(bf(gate[j]*sigmoid(gate[j]))*up[j]);
        project(e.down,activation,down);
    }
    void forward(std::span<const float> x,std::span<float> y){
        require(!invalid,"moe_reset_required");ready=false;
        try{
            require(x.size()==c.hidden&&y.size()==c.hidden,"moe_input_shape");math::detail::output_alias(x,y,true);for(float v:x)require(bf(v)==v,"moe_input_bf16");
            auto logits=span(p.logits,c.experts),probs=span(p.probabilities,c.experts),top=span(p.top_weights,c.top_k),accum=span(p.accumulator,c.hidden);
            float maximum=-std::numeric_limits<float>::infinity();for(std::size_t e=0;e<c.experts;++e){logits[e]=dense(w.router.subspan(e*c.hidden,c.hidden),x);maximum=std::max(maximum,logits[e]);}
            float total=0;for(std::size_t e=0;e<c.experts;++e){probs[e]=std::exp(logits[e]-maximum);total=finite(total+probs[e]);}for(auto& v:probs)v/=total;
            float picked=0;for(std::size_t rank=0;rank<c.top_k;++rank){unsigned best=unsigned(c.experts);for(unsigned e=0;e<c.experts;++e){bool used=false;for(std::size_t j=0;j<rank;++j)used|=selected[j]==e;
                if(!used&&(best==c.experts||probs[e]>probs[best]))best=e;}
                selected[rank]=best;picked+=probs[best];}
            for(std::size_t rank=0;rank<c.top_k;++rank)top[rank]=bf(probs[selected[rank]]/picked);
            std::fill(accum.begin(),accum.end(),0);
            // Score order is retained for diagnostics; execution/accumulation is ascending expert id.
            for(unsigned e=0;e<c.experts;++e)for(std::size_t rank=0;rank<c.top_k;++rank)if(selected[rank]==e){
                expert(w.experts[e],c.intermediate,x);auto down=span(p.down,c.hidden);for(std::size_t j=0;j<c.hidden;++j)accum[j]=bf(accum[j]+bf(down[j]*top[rank]));}
            const float factor=bf(sigmoid(dense(w.shared_gate,x)));scratch[p.shared_factor]=factor;
            expert(w.shared,c.shared_intermediate,x);auto down=span(p.down,c.hidden),shared=span(p.shared,c.hidden),result=span(p.result,c.hidden);
            for(std::size_t j=0;j<c.hidden;++j){shared[j]=bf(down[j]*factor);result[j]=bf(accum[j]+shared[j]);}
            std::copy(result.begin(),result.end(),y.begin());ready=true;
        }catch(...){invalid=true;throw;}
    }
};
Reference::Reference(Config c,Weights w,std::size_t budget):impl_(std::make_unique<Impl>(c,w,budget)){}
Reference::~Reference()=default;
void Reference::forward(std::span<const float> x,std::span<float> y){impl_->forward(x,y);}
void Reference::reset(){std::fill(impl_->scratch.begin(),impl_->scratch.end(),0);std::fill(impl_->selected.begin(),impl_->selected.end(),0);impl_->invalid=impl_->ready=false;}
bool Reference::valid()const{return !impl_->invalid;}
std::span<const unsigned> Reference::selected()const{impl_->diagnostic(0,0);return impl_->selected;}
std::span<const float> Reference::logits()const{return impl_->diagnostic(impl_->p.logits,impl_->c.experts);}
std::span<const float> Reference::probabilities()const{return impl_->diagnostic(impl_->p.probabilities,impl_->c.experts);}
std::span<const float> Reference::top_weights()const{return impl_->diagnostic(impl_->p.top_weights,impl_->c.top_k);}
std::span<const float> Reference::routed()const{return impl_->diagnostic(impl_->p.accumulator,impl_->c.hidden);}
std::span<const float> Reference::shared()const{return impl_->diagnostic(impl_->p.shared,impl_->c.hidden);}
std::span<const float> Reference::result()const{return impl_->diagnostic(impl_->p.result,impl_->c.hidden);}
} // namespace kadan::moe
