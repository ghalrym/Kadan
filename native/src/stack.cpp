#include "kadan/stack.hpp"
#include "kadan/cuda_plan.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace kadan::stack {
namespace {
void require(bool x,const char* e){if(!x)throw std::invalid_argument(e);}
std::size_t add(std::size_t a,std::size_t b){if(b>SIZE_MAX-a)throw std::overflow_error("stack_layout");return a+b;}
void cancel(const std::atomic_bool* f){if(f&&f->load(std::memory_order_relaxed))throw std::invalid_argument("stack_cancelled");}
float weight(const quantization::Matrix& w,std::size_t r,std::size_t j){
    auto code=(w.weights[r*w.columns/2+j/2]>>(4*(j%2)))&15;
    const double v=double(quantization::e2m1(code))*quantization::e4m3fn(w.block_scales[r*w.columns/16+j/16])*w.multipliers[0];
    require(std::isfinite(v)&&std::abs(v)<=std::numeric_limits<float>::max(),"stack_head_weight");return float(v);
}
}
Plan plan(Config c){
    require(c.vocabulary>0&&c.vocabulary<=262144&&c.eos<c.vocabulary,"stack_vocabulary");require(std::isfinite(c.epsilon)&&c.epsilon>0,"stack_epsilon");Plan p{};
    auto region=[&](std::size_t n){auto at=p.device_bytes;p.device_bytes=add(p.device_bytes,add(n,255)&~std::size_t{255});return at;};
    for(std::size_t i=0;i<layers;++i){require(c.layer[i].attention==(i==3?decoder::Attention::full:decoder::Attention::linear),"stack_layer_order");p.layer[i]=decoder::plan(c.layer[i]);if(i==0){p.hidden=p.layer[i].hidden;p.capacity=p.layer[i].capacity;}
        require(p.layer[i].hidden==p.hidden&&p.layer[i].capacity==p.capacity,"stack_layer_shape");p.offset[i]=region(p.layer[i].device_bytes);p.host_numeric_bytes=add(p.host_numeric_bytes,p.layer[i].host_numeric_bytes);}
    p.embedding=region(c.vocabulary*p.hidden*2);p.final_norm=region(p.hidden*2);p.head_weights=region(c.vocabulary*p.hidden/2);p.head_scales=region(c.vocabulary*p.hidden/16);
    p.embedded=region(p.hidden*4);p.normalized=region(p.hidden*4);p.logits=region(c.vocabulary*4);p.selected=region(4);p.status=region(4);
    p.host_numeric_bytes=add(p.host_numeric_bytes,(6*p.hidden+c.vocabulary)*4);return p;
}
void validate_weights(Config c,const Weights&w){
    auto p=plan(c);for(std::size_t i=0;i<layers;++i)decoder::validate_weights(c.layer[i],w.layer[i]);
    require(w.embedding.size()==c.vocabulary*p.hidden&&w.final_norm.size()==p.hidden,"stack_weight_shape");for(auto span:{w.embedding,w.final_norm})for(float v:span)require(linear::bf16_round(v)==v,"stack_weight_bf16");
    require(w.head.encoding==quantization::Encoding::modelopt_nvfp4&&w.head.rows==c.vocabulary&&w.head.columns==p.hidden,"stack_head_shape");(void)cuda::plan_nvfp4(w.head,SIZE_MAX);
    for(std::size_t r=0;r<c.vocabulary;++r)for(std::size_t j=0;j<p.hidden;++j)(void)weight(w.head,r,j);
}
struct Reference::Impl {
    Config c;Weights w;Plan p;StateCursor cursor;bool ended=false,ready=false;std::array<std::unique_ptr<decoder::Reference>,layers> layer;
    std::vector<float> scratch;
    Impl(Config config,Weights weights,std::size_t budget):c(config),w(weights),p(plan(c)),cursor(layers,p.capacity){
        validate_weights(c,w);require(budget>=p.host_numeric_bytes,"stack_host_budget");scratch.resize(6*p.hidden+c.vocabulary);
        for(std::size_t i=0;i<layers;++i)layer[i]=std::make_unique<decoder::Reference>(c.layer[i],w.layer[i],p.layer[i].host_numeric_bytes);
    }
    std::span<float> hidden(std::size_t i){return std::span(scratch).subspan(i*p.hidden,p.hidden);}
    std::span<float> logits(){return std::span(scratch).subspan(6*p.hidden,c.vocabulary);}
    Selection step(unsigned input,bool stop,const std::atomic_bool* cancelled){
        require(!ended,"stack_eos");auto token=cursor.begin();ready=false;
        try{
            require(input<c.vocabulary,"stack_input_id");cancel(cancelled);auto x=hidden(0);std::copy_n(w.embedding.begin()+input*p.hidden,p.hidden,x.begin());
            for(std::size_t i=0;i<layers;++i){layer[i]->step(x,hidden(i+1),cancelled);x=hidden(i+1);cursor.written(token,i);cancel(cancelled);}
            float sum=0;for(float v:x)sum+=v*v;require(std::isfinite(sum),"stack_norm_nonfinite");float inv=1/std::sqrt(sum/float(p.hidden)+c.epsilon);auto n=hidden(5);
            for(std::size_t j=0;j<p.hidden;++j)n[j]=linear::bf16_round((x[j]*inv)*(1+w.final_norm[j]));
            auto scores=logits();for(std::size_t r=0;r<c.vocabulary;++r){double total=0;for(std::size_t j=0;j<p.hidden;++j)total+=double(weight(w.head,r,j))*n[j];require(std::isfinite(total)&&std::abs(total)<=std::numeric_limits<float>::max(),"stack_head_nonfinite");scores[r]=linear::bf16_round(float(total));}
            unsigned selected=0;for(unsigned j=1;j<c.vocabulary;++j)if(scores[j]>scores[selected])selected=j;
            Selection result{selected,selected==c.eos};cancel(cancelled);cursor.ready_to_commit(token);cursor.commit(token);ready=true;ended=stop&&result.eos;return result;
        }catch(...){cursor.invalidate();throw;}
    }
};
Reference::Reference(Config c,Weights w,std::size_t b):impl_(std::make_unique<Impl>(c,w,b)){}
Reference::~Reference()=default;
Selection Reference::step(unsigned x,bool stop,const std::atomic_bool*c){return impl_->step(x,stop,c);}
void Reference::reset(){auto&i=*impl_;for(auto&l:i.layer)l->reset();std::fill(i.scratch.begin(),i.scratch.end(),0);i.cursor.reset();i.ended=i.ready=false;}
bool Reference::valid()const{return impl_->cursor.valid();}bool Reference::finished()const{return impl_->ended;}std::size_t Reference::tokens()const{return impl_->cursor.committed_tokens();}
void Reference::read_layer(std::size_t index,std::span<float> out)const{auto&i=*impl_;require(valid()&&i.ready&&index<layers&&out.size()==i.p.hidden,"stack_diagnostic");std::copy(i.hidden(index+1).begin(),i.hidden(index+1).end(),out.begin());}
void Reference::read_output(std::span<float>n,std::span<float>logits)const{auto&i=*impl_;require(valid()&&i.ready&&n.size()==i.p.hidden&&logits.size()==i.c.vocabulary,"stack_diagnostic");std::copy(i.hidden(5).begin(),i.hidden(5).end(),n.begin());std::copy(i.logits().begin(),i.logits().end(),logits.begin());}
void Reference::read_state(std::size_t index,std::span<std::uint8_t>a,std::span<std::uint8_t>b)const{auto&i=*impl_;require(valid()&&index<layers,"stack_state");i.layer[index]->read_state(a,b);}
}
