#include "kadan/decoder.hpp"
#include "layer_math_validation.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace kadan::decoder {
namespace {
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
std::size_t add(std::size_t a,std::size_t b){if(b>SIZE_MAX-a)throw std::overflow_error("decoder_layout");return a+b;}
void cancel(const std::atomic_bool* flag){if(flag&&flag->load(std::memory_order_relaxed))throw std::invalid_argument("decoder_cancelled");}
}
Plan plan(Config c){
    Plan p{};require(c.attention==Attention::linear||c.attention==Attention::full,"decoder_attention");
    const bool linear=c.attention==Attention::linear;
    if(linear){p.linear=linear::plan(c.linear);p.hidden=c.linear.hidden;p.capacity=c.linear.capacity;p.epsilon=c.linear.epsilon;}
    else{p.full=full::plan(c.full);p.hidden=c.full.hidden;p.capacity=c.full.capacity;p.epsilon=c.full.epsilon;}
    p.moe=moe::plan(c.moe);require(c.moe.hidden==p.hidden,"decoder_hidden");
    auto region=[&](std::size_t bytes){auto at=p.device_bytes;p.device_bytes=add(p.device_bytes,add(bytes,255)&~std::size_t{255});return at;};
    auto projection=[&](std::size_t rows,std::size_t columns){auto& v=p.projections[p.projection_count++];v={region(rows*columns),region(4),rows,columns};};
    if(linear){projection(p.linear.channels,p.hidden);projection(p.linear.values,p.hidden);projection(p.hidden,p.linear.values);}
    else{projection(2*p.full.queries,p.hidden);projection(p.full.kv,p.hidden);projection(p.full.kv,p.hidden);projection(p.hidden,p.full.queries);}
    p.auxiliary=region((linear?p.linear.aux_elements:p.full.norm_weights)*2);
    p.frequencies=region(linear?0:c.full.rotary_dim/2*4);p.post_norm=region(p.hidden*2);
    p.moe_offset=region(p.moe.device_bytes);
    p.state_first_bytes=linear?p.linear.conv_elements*2:p.full.history_elements*2;
    p.state_second_bytes=linear?p.linear.recurrent_elements*4:p.full.history_elements*2;
    p.state_first=region(p.state_first_bytes);p.state_second=region(p.state_second_bytes);
    p.attention_scratch=region((linear?p.linear.scratch_floats:p.full.scratch_floats)*4);
    p.workspace=region(4*p.hidden*4);p.status=region(4);
    p.host_numeric_bytes=add(add(linear?p.linear.host_state_workspace_bytes:p.full.host_state_workspace_bytes,p.moe.host_workspace_bytes),4*p.hidden*4);
    return p;
}
void validate_weights(Config c,const Weights& w){
    const auto p=plan(c);if(c.attention==Attention::linear)linear::validate_weights(c.linear,w.linear);else full::validate_weights(c.full,w.full);
    moe::validate_weights(c.moe,w.moe);require(w.post_norm.size()==p.hidden,"decoder_post_norm_shape");
    for(float x:w.post_norm)require(linear::bf16_round(x)==x,"decoder_post_norm_bf16");
}
struct Reference::Impl {
    Config c;Weights w;Plan p;StateCursor cursor;bool ready=false;
    std::unique_ptr<linear::Reference> linear;std::unique_ptr<full::Reference> full;std::unique_ptr<moe::Reference> moe;
    std::vector<float> workspace;
    Impl(Config config,Weights weights,std::size_t budget):c(config),w(weights),p(plan(c)),cursor(1,p.capacity){
        validate_weights(c,w);require(budget>=p.host_numeric_bytes,"decoder_host_budget");workspace.resize(4*p.hidden);
        if(c.attention==Attention::linear)linear=std::make_unique<linear::Reference>(c.linear,w.linear,p.linear.host_state_workspace_bytes);
        else full=std::make_unique<full::Reference>(c.full,w.full,p.full.host_state_workspace_bytes);
        moe=std::make_unique<moe::Reference>(c.moe,w.moe,p.moe.host_workspace_bytes);
    }
    std::span<float> span(std::size_t i){return std::span(workspace).subspan(i*p.hidden,p.hidden);}
    void step(std::span<const float> x,std::span<float> y,const std::atomic_bool* cancelled){
        const auto step=cursor.begin();ready=false;
        try{
            require(x.size()==p.hidden&&y.size()==p.hidden,"decoder_input_shape");math::detail::output_alias(x,y,true);cancel(cancelled);
            auto a=span(0),u=span(1),m=span(2),result=span(3);
            if(linear)linear->step(x,a);else full->step(x,a);
            cancel(cancelled);float sum=0;for(float v:a)sum+=v*v;
            require(std::isfinite(sum),"decoder_norm_nonfinite");const float inv=1/std::sqrt(sum/float(p.hidden)+p.epsilon);
            for(std::size_t j=0;j<p.hidden;++j)u[j]=linear::bf16_round((a[j]*inv)*(1+w.post_norm[j]));
            moe->forward(u,m);for(std::size_t j=0;j<p.hidden;++j)result[j]=linear::bf16_round(a[j]+m[j]);cancel(cancelled);
            cursor.written(step,0);cursor.ready_to_commit(step);
            std::copy(result.begin(),result.end(),y.begin());cursor.commit(step);ready=true;
        }catch(...){cursor.invalidate();throw;}
    }
};
Reference::Reference(Config c,Weights w,std::size_t b):impl_(std::make_unique<Impl>(c,w,b)){}
Reference::~Reference()=default;
void Reference::step(std::span<const float> x,std::span<float> y,const std::atomic_bool* c){impl_->step(x,y,c);}
void Reference::reset(){auto& i=*impl_;if(i.linear)i.linear->reset();else i.full->reset();i.moe->reset();std::fill(i.workspace.begin(),i.workspace.end(),0);i.cursor.reset();i.ready=false;}
bool Reference::valid()const{return impl_->cursor.valid();}
std::size_t Reference::tokens()const{return impl_->cursor.committed_tokens();}
void Reference::read_state(std::span<std::uint8_t>a,std::span<std::uint8_t>b)const{
    auto& i=*impl_;require(valid()&&a.size()==i.p.state_first_bytes&&b.size()==i.p.state_second_bytes,"decoder_state_read");
    if(i.linear){std::memcpy(a.data(),i.linear->convolution().data(),a.size());std::memcpy(b.data(),i.linear->recurrent().data(),b.size());}
    else{std::memcpy(a.data(),i.full->keys().data(),a.size());std::memcpy(b.data(),i.full->values().data(),b.size());}
}
void Reference::read_intermediates(std::span<float>a,std::span<float>u,std::span<float>m)const{auto& i=*impl_;require(valid()&&i.ready&&a.size()==i.p.hidden&&u.size()==i.p.hidden&&m.size()==i.p.hidden,"decoder_diagnostic");for(auto pair:{std::pair{a,0},std::pair{u,1},std::pair{m,2}})std::copy(i.span(pair.second).begin(),i.span(pair.second).end(),pair.first.begin());}
} // namespace kadan::decoder
