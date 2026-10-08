#pragma once
// Private trusted producer. Its arena and cursor are borrowed from ONE owner.
// No admission, allocation, lifetime management or logical publication here.
#include "kadan/decoder.hpp"
#include "linear_kernel.cuh"
#include "full_kernel.cuh"
#include "moe_kernel.cuh"
#include "decoder_kernel.cuh"
#include "fp8_kernel.cuh"
#include "nvfp4_kernel.cuh"
#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
namespace kadan::cuda::detail {
struct DecoderProducer {
    struct Projection {const std::uint8_t *weights=nullptr,*scales=nullptr;float global=0;std::size_t rows=0,columns=0;};
    decoder::Config c;decoder::Plan p;StateCursor& cursor;void* storage=nullptr;
    LinearBuffers linear{};FullBuffers full{};MoeBuffers moe{};
    std::array<std::array<Projection,3>,257> experts{};
    DecoderProducer(decoder::Config config,decoder::Plan layout,StateCursor& authority):c(config),p(layout),cursor(authority){}
    DecoderProducer(const DecoderProducer&)=delete;DecoderProducer& operator=(const DecoderProducer&)=delete;
    static void require(bool ok,const char* error){if(!ok)throw std::invalid_argument(error);}
    static void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
    std::uint8_t* bytes(std::size_t offset){return static_cast<std::uint8_t*>(storage)+offset;}
    template<class T>T* pointer(std::size_t offset){return reinterpret_cast<T*>(bytes(offset));}
    float* work(std::size_t index){return pointer<float>(p.workspace)+index*p.hidden;}
    unsigned* flags(){return pointer<unsigned>(p.status);}
    const std::uint16_t* upload(std::span<const float> values,std::size_t offset){
        auto* dst=pointer<std::uint16_t>(offset);std::array<std::uint16_t,1024> staging{};
        for(std::size_t at=0;at<values.size();at+=staging.size()){auto n=std::min(staging.size(),values.size()-at);for(std::size_t j=0;j<n;++j)staging[j]=std::uint16_t(std::bit_cast<std::uint32_t>(values[at+j])>>16);check(cudaMemcpy(dst+at,staging.data(),n*2,cudaMemcpyHostToDevice));}return dst;
    }
    void initialize(const decoder::Weights& w){
        const bool is_linear=c.attention==decoder::Attention::linear;
        const std::array<quantization::Matrix,4> projections=is_linear?std::array<quantization::Matrix,4>{w.linear.qkv,w.linear.z,w.linear.out,{}}:std::array<quantization::Matrix,4>{w.full.q_gate,w.full.key,w.full.value,w.full.out};
        for(std::size_t i=0;i<p.projection_count;++i){check(cudaMemcpy(bytes(p.projections[i].weights),projections[i].weights.data(),projections[i].weights.size_bytes(),cudaMemcpyHostToDevice));check(cudaMemcpy(bytes(p.projections[i].scale),projections[i].multipliers.data(),4,cudaMemcpyHostToDevice));}
        auto offset=p.auxiliary;auto aux=[&](std::span<const float> v){auto* result=upload(v,offset);offset+=v.size()*2;return result;};
        auto* scratch=pointer<float>(p.attention_scratch);
        if(is_linear){
            linear.input_norm=aux(w.linear.input_norm);linear.conv_weight=aux(w.linear.conv);linear.a_weight=aux(w.linear.a);linear.b_weight=aux(w.linear.b);linear.a_log=aux(w.linear.a_log);linear.dt_bias=aux(w.linear.dt_bias);linear.output_norm=aux(w.linear.output_norm);
            const auto& l=p.linear;linear.normalized=scratch+l.norm_offset;linear.qkv=scratch+l.qkv_offset;linear.z=scratch+l.z_offset;linear.a=scratch+l.a_offset;linear.b=scratch+l.b_offset;linear.core=scratch+l.core_offset;linear.gated=scratch+l.gate_offset;linear.projected=scratch+l.out_offset;
            linear.convolution=pointer<std::uint16_t>(p.state_first);linear.recurrent=pointer<float>(p.state_second);linear.status=flags();
        }else{
            full.input_norm=aux(w.full.input_norm);full.query_norm=aux(w.full.query_norm);full.key_norm=aux(w.full.key_norm);
            full.frequencies=pointer<float>(p.frequencies);check(cudaMemcpy(bytes(p.frequencies),w.full.frequencies.data(),w.full.frequencies.size_bytes(),cudaMemcpyHostToDevice));
            const auto& f=p.full;full.normalized=scratch+f.norm_offset;full.qg=scratch+f.qg_offset;full.key=scratch+f.k_offset;full.value=scratch+f.v_offset;full.query=scratch+f.q_offset;full.gate=scratch+f.gate_offset;full.probabilities=scratch+f.prob_offset;full.core=scratch+f.core_offset;full.gated=scratch+f.gated_offset;full.projected=scratch+f.out_offset;
            full.keys=pointer<std::uint16_t>(p.state_first);full.values=pointer<std::uint16_t>(p.state_second);full.status=flags();
        }
        upload(w.post_norm,p.post_norm);const auto& m=p.moe;auto base=p.moe_offset;
        moe.router=upload(w.moe.router,base+m.router_offset);moe.shared_gate=upload(w.moe.shared_gate,base+m.shared_gate_offset);
        auto matrix=[&](const quantization::Matrix& source,std::size_t at,std::size_t weights,std::size_t scales){
            check(cudaMemcpy(bytes(at+weights),source.weights.data(),source.weights.size_bytes(),cudaMemcpyHostToDevice));check(cudaMemcpy(bytes(at+scales),source.block_scales.data(),source.block_scales.size_bytes(),cudaMemcpyHostToDevice));
            return Projection{bytes(at+weights),bytes(at+scales),source.multipliers[0],source.rows,source.columns};};
        for(std::size_t e=0;e<=c.moe.experts;++e){const bool shared=e==c.moe.experts;const auto& source=shared?w.moe.shared:w.moe.experts[e];const auto& l=shared?m.shared_layout:m.routed_layout;const auto at=base+(shared?m.shared_offset:m.experts_offset+e*m.routed_layout.bytes);
            experts[e]={matrix(source.gate,at,l.gate_weights,l.gate_scales),matrix(source.up,at,l.up_weights,l.up_scales),matrix(source.down,at,l.down_weights,l.down_scales)};}
        auto* s=pointer<float>(base+m.scratch_offset);moe.logits=s+m.logits;moe.probabilities=s+m.probabilities;moe.top_weights=s+m.top_weights;moe.gate=s+m.gate;moe.up=s+m.up;moe.activation=s+m.activation;moe.down=s+m.down;moe.accumulator=s+m.accumulator;moe.shared=s+m.shared;moe.result=s+m.result;moe.shared_factor=s+m.shared_factor;moe.selected=pointer<unsigned>(base+m.indices_offset);moe.status=flags();
        zero();
    }
    void bind() {
        const bool is_linear=c.attention==decoder::Attention::linear;
        auto at=p.auxiliary;auto aux=[&](std::size_t n){auto* v=pointer<std::uint16_t>(at);at+=2*n;return v;};
        auto* scratch=pointer<float>(p.attention_scratch);
        if(is_linear){const auto& l=p.linear;
            linear.input_norm=aux(p.hidden);linear.conv_weight=aux(l.conv_elements);linear.a_weight=aux(c.linear.value_heads*p.hidden);linear.b_weight=aux(c.linear.value_heads*p.hidden);linear.a_log=aux(c.linear.value_heads);linear.dt_bias=aux(c.linear.value_heads);linear.output_norm=aux(c.linear.value_dim);
            linear.normalized=scratch+l.norm_offset;linear.qkv=scratch+l.qkv_offset;linear.z=scratch+l.z_offset;linear.a=scratch+l.a_offset;linear.b=scratch+l.b_offset;linear.core=scratch+l.core_offset;linear.gated=scratch+l.gate_offset;linear.projected=scratch+l.out_offset;
            linear.convolution=pointer<std::uint16_t>(p.state_first);linear.recurrent=pointer<float>(p.state_second);linear.status=flags();
        }else{const auto& f=p.full;
            full.input_norm=aux(p.hidden);full.query_norm=aux(c.full.head_dim);full.key_norm=aux(c.full.head_dim);full.frequencies=pointer<float>(p.frequencies);
            full.normalized=scratch+f.norm_offset;full.qg=scratch+f.qg_offset;full.key=scratch+f.k_offset;full.value=scratch+f.v_offset;full.query=scratch+f.q_offset;full.gate=scratch+f.gate_offset;full.probabilities=scratch+f.prob_offset;full.core=scratch+f.core_offset;full.gated=scratch+f.gated_offset;full.projected=scratch+f.out_offset;
            full.keys=pointer<std::uint16_t>(p.state_first);full.values=pointer<std::uint16_t>(p.state_second);full.status=flags();
        }
        const auto& m=p.moe;const auto base=p.moe_offset;moe.router=pointer<std::uint16_t>(base+m.router_offset);moe.shared_gate=pointer<std::uint16_t>(base+m.shared_gate_offset);
        for(std::size_t e=0;e<=c.moe.experts;++e){const bool shared=e==c.moe.experts;const auto& l=shared?m.shared_layout:m.routed_layout;const auto middle=shared?c.moe.shared_intermediate:c.moe.intermediate;const auto start=base+(shared?m.shared_offset:m.experts_offset+e*l.bytes);
            experts[e]={Projection{bytes(start+l.gate_weights),bytes(start+l.gate_scales),0,middle,p.hidden},Projection{bytes(start+l.up_weights),bytes(start+l.up_scales),0,middle,p.hidden},Projection{bytes(start+l.down_weights),bytes(start+l.down_scales),0,p.hidden,middle}};}
        auto* s=pointer<float>(base+m.scratch_offset);moe.logits=s+m.logits;moe.probabilities=s+m.probabilities;moe.top_weights=s+m.top_weights;moe.gate=s+m.gate;moe.up=s+m.up;moe.activation=s+m.activation;moe.down=s+m.down;moe.accumulator=s+m.accumulator;moe.shared=s+m.shared;moe.result=s+m.result;moe.shared_factor=s+m.shared_factor;moe.selected=pointer<unsigned>(base+m.indices_offset);moe.status=flags();
    }
    void zero(){check(cudaMemsetAsync(bytes(p.moe_offset+p.moe.scratch_offset),0,p.moe.device_bytes-p.moe.scratch_offset,cudaStreamLegacy));check(cudaMemsetAsync(bytes(p.state_first),0,p.device_bytes-p.state_first,cudaStreamLegacy));}
    void status(){check(cudaStreamSynchronize(cudaStreamLegacy));unsigned value=0;check(cudaMemcpy(&value,flags(),4,cudaMemcpyDeviceToHost));if(value)throw std::overflow_error("decoder_numeric_failure");}
    void fp8(std::size_t i,const float* x,float* y){const auto& v=p.projections[i];check((c.linear.bf16_weights||c.full.bf16_weights?kadan_launch_fp8_bf16:kadan_launch_fp8)(bytes(v.weights),pointer<float>(v.scale),false,x,y,flags(),v.rows,v.columns));status();}
    // Private borrowed producers: coordinator validates the owner-bound active
    // capability on entry. No child owns admission, state lifetime or progress.
    void attention(StateStep token,const float* input){
        cursor.check_step(token);
        if(c.attention==decoder::Attention::linear){
            check(detail::linear_normalize(c.linear,linear,input));status();fp8(0,linear.normalized,linear.qkv);fp8(1,linear.normalized,linear.z);
            check(detail::linear_core(c.linear,linear));status();fp8(2,linear.gated,linear.projected);check(detail::linear_residual(c.linear,linear,input,work(0)));status();
        }else{
            check(detail::full_normalize(c.full,full,input));status();fp8(0,full.normalized,full.qg);fp8(1,full.normalized,full.key);fp8(2,full.normalized,full.value);
            check(detail::full_core(c.full,full,cursor.committed_tokens()));status();fp8(3,full.gated,full.projected);check(detail::full_residual(c.full,full,input,work(0)));status();
        }
    }
    void project(const Projection& m,const float* x,float* y){check((c.moe.bf16_weights?kadan_launch_nvfp4_bf16:kadan_launch_nvfp4)(m.weights,m.scales,m.global,x,y,flags(),m.rows,m.columns));}
    void expert(std::size_t e,std::size_t middle){project(experts[e][0],work(1),moe.gate);project(experts[e][1],work(1),moe.up);check(detail::moe_activate(middle,moe));status();project(experts[e][2],moe.activation,moe.down);}
    void mixture(StateStep token){
        cursor.check_step(token);check(detail::moe_route(c.moe,moe,work(1)));status();std::array<unsigned,8> selected{};
        check(cudaMemcpy(selected.data(),moe.selected,c.moe.top_k*sizeof(unsigned),cudaMemcpyDeviceToHost));std::sort(selected.begin(),selected.begin()+c.moe.top_k);
        for(std::size_t i=0;i<c.moe.top_k;++i){require(selected[i]<c.moe.experts&&(i==0||selected[i]!=selected[i-1]),"decoder_device_route");expert(selected[i],c.moe.intermediate);check(detail::moe_accumulate(c.moe,moe,selected[i]));status();}
        expert(c.moe.experts,c.moe.shared_intermediate);check(detail::moe_finish(c.moe,moe,work(2)));status();
    }
    void cancel(const std::atomic_bool* flag){if(flag&&flag->load(std::memory_order_relaxed))throw std::invalid_argument("decoder_cancelled");}
    void run(StateStep step,const float* input,const std::atomic_bool* cancelled){
        cursor.check_step(step);check(cudaMemsetAsync(flags(),0,4,cudaStreamLegacy));
        attention(step,input);cancel(cancelled);
        check(detail::decoder_norm(p.hidden,p.epsilon,pointer<std::uint16_t>(p.post_norm),work(0),work(1),flags()));status();mixture(step);cancel(cancelled);
        check(detail::decoder_residual(p.hidden,work(0),work(2),work(3),flags()));status();cancel(cancelled);
    }
};
} // namespace kadan::cuda::detail
