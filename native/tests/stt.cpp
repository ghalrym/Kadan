#include "kadan/stt.hpp"
#include <cmath>
#include <iostream>
#include <limits>
using namespace kadan;
void check(bool ok){if(!ok)throw std::runtime_error("test_failed");}
template<class F>void rejects(F f,const char* error){try{f();}catch(const std::exception& e){check(std::string(e.what())==error);return;}throw std::runtime_error("expected_failure");}
int main(){
    using stt::LogMel;std::atomic_bool cancel{false};
    for(std::size_t bins:{80,128}){
        auto ledger=std::make_shared<Resources>(Footprint{2*1024*1024,0});
        LogMel frontend(ledger);std::vector<float> filters(bins*201,0),pcm(400,0),output(bins*2);
        for(std::size_t band=0;band<bins;++band)filters[band*201]=1;
        rejects([&]{frontend.execute(pcm,output,cancel);},"stt_not_loaded");
        cancel=true;rejects([&]{frontend.load(bins,filters,cancel);},"stt_cancelled");cancel=false;
        rejects([&]{frontend.load(81,filters,cancel);},"stt_mel_bins");
        rejects([&]{frontend.load(bins,{},cancel);},"stt_filter_shape");
        for(float invalid:{-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
            filters[0]=invalid;rejects([&]{frontend.load(bins,filters,cancel);},"stt_invalid_filter");check(ledger->snapshot().residents==0);
        }
        filters[0]=1;
        auto small=std::make_shared<Resources>(Footprint{LogMel::resident_bytes(bins)-1});LogMel denied(small);
        rejects([&]{denied.load(bins,filters,cancel);},"exhausted");check(small->snapshot().residents==0);
        frontend.load(bins,filters,cancel);const auto bytes=LogMel::resident_bytes(bins);
        check(ledger->snapshot().used==Footprint({bytes,0}));
        rejects([&]{frontend.load(bins,filters,cancel);},"stt_already_loaded");
        auto pressure=ledger->reserve(Workload::speech,{ledger->snapshot().capacity[0]-bytes,0});
        rejects([&]{frontend.execute(pcm,output,cancel);},"exhausted");ledger->released(pressure);
        frontend.execute(pcm,output,cancel,[&](std::size_t){
            check(ledger->snapshot().used==Footprint({bytes+LogMel::scratch_bytes,0}));
            rejects([&]{frontend.unload();},"busy");
        });
        for(float v:output)check(v==-1.5f);
        std::fill(pcm.begin(),pcm.end(),1);frontend.execute(pcm,output,cancel);
        const float constant=(std::log10(40000.f)+4)/4;for(float v:output)check(std::abs(v-constant)<1e-6f);
        const auto first=output;frontend.execute(pcm,output,cancel);check(output==first);
        rejects([&]{frontend.execute(pcm,output,cancel,[&](std::size_t frame){if(frame==1)cancel=true;});},"stt_cancelled");cancel=false;
        rejects([&]{frontend.execute(pcm,output,cancel,[](std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
        pcm[0]=std::numeric_limits<float>::quiet_NaN();rejects([&]{frontend.execute(pcm,output,cancel);},"stt_nonfinite_input");pcm[0]=0;
        rejects([&]{frontend.execute(pcm,{},cancel);},"stt_output_shape");
        rejects([&]{frontend.execute(pcm,{pcm.data(),bins*2},cancel);},"stt_buffer_overlap");
        check(ledger->snapshot().used==Footprint({bytes,0}));frontend.unload();frontend.unload();check(ledger->snapshot().residents==0);
        {LogMel scoped(ledger);scoped.load(bins,filters,cancel);}check(ledger->snapshot().residents==0);
    }
    for(std::size_t n:{0,200,480001})rejects([&]{LogMel::frames(n);},"stt_sample_count");
    check(LogMel::frames(201)==1 && LogMel::frames(319)==1 && LogMel::frames(320)==2 && LogMel::frames(16000)==100 && LogMel::frames(480000)==3000);
    {
        constexpr std::size_t bins=80;
        const auto resident=LogMel::resident_bytes(bins),window=LogMel::max_samples*sizeof(float);
        auto ledger=std::make_shared<Resources>(Footprint{resident+window+LogMel::scratch_bytes,0});
        LogMel frontend(ledger);std::vector<float> filters(bins*201,0),output(bins*3000,42),pcm(LogMel::max_samples,1);
        for(std::size_t b=0;b<bins;++b)filters[b*201]=1;
        frontend.load(bins,filters,cancel);
        auto pressure=ledger->reserve(Workload::speech,{1,0});
        rejects([&]{frontend.execute_window({},output,cancel);},"exhausted");
        check(output.front()==42 && output.back()==42);ledger->released(pressure);
        rejects([&]{frontend.execute_window({},output,cancel,[&](std::size_t frame){
            check(frame==0);check(ledger->snapshot().used==Footprint({resident+window+LogMel::scratch_bytes,0}));
            rejects([&]{frontend.unload();},"busy");cancel=true;
        });},"stt_cancelled");cancel=false;
        check(ledger->snapshot().used==Footprint({resident,0}));
        rejects([&]{frontend.execute_window({},output,cancel,[](std::size_t){throw std::runtime_error("hook_failure");});},"hook_failure");
        rejects([&]{frontend.execute_window({}, {},cancel);},"stt_output_shape");
        rejects([&]{frontend.execute_window(pcm,{pcm.data(),output.size()},cancel);},"stt_buffer_overlap");
        frontend.execute_window(pcm,output,cancel);
        const float expected=(std::log10(40000.f)+4)/4;
        for(float v:output)check(std::abs(v-expected)<1e-6f);
        check(ledger->snapshot().used==Footprint({resident,0}));
        frontend.unload();check(ledger->snapshot().used==Footprint({0,0}));
    }
    std::cout<<"stt numerics, cancellation, pins, budgets and cleanup passed\n";
}
