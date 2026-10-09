#include "kadan/stt.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace kadan::stt {
namespace {
void check(bool ok,const char* error){if(!ok)throw std::runtime_error(error);}
void cancelled(const std::atomic_bool& flag){check(!flag.load(),"stt_cancelled");}
struct Reservation {
    Resources& r; Handle h;
    Reservation(Resources& ledger,Footprint bytes):r(ledger),h(r.reserve(Workload::speech,std::move(bytes))){}
    ~Reservation(){if(h)r.released(h);}
};
struct Pin {
    Resources& r;Handle h;
    Pin(Resources& ledger,Handle handle):r(ledger),h(handle){r.pin(h);}
    ~Pin(){r.unpin(h);}
};
}
LogMel::LogMel(std::shared_ptr<Resources> resources):resources_(std::move(resources)){
    check(bool(resources_),"stt_resources_required");
}
LogMel::~LogMel(){unload();}
Footprint LogMel::host(Bytes bytes) const {
    auto result=resources_->snapshot().capacity;
    std::fill(result.begin(),result.end(),0);result[0]=bytes;return result;
}
std::size_t LogMel::frames(std::size_t samples){
    check(samples>=min_samples && samples<=max_samples,"stt_sample_count");return samples/hop;
}
Bytes LogMel::resident_bytes(std::size_t bins){
    check(bins==80 || bins==128,"stt_mel_bins");return table_bytes+bins*frequencies*sizeof(float);
}
void LogMel::load(std::size_t bins,std::span<const float> filters,const std::atomic_bool& cancel){
    cancelled(cancel);check(!loaded(),"stt_already_loaded");
    const auto bytes=resident_bytes(bins);
    check(filters.size()==bins*frequencies,"stt_filter_shape");
    Reservation admission(*resources_,host(bytes));
    auto tables=std::make_unique<double[]>(table_bytes/sizeof(double));
    auto bank=std::make_unique<float[]>(filters.size());
    for(std::size_t i=0;i<filters.size();++i){
        if(i%frequencies==0)cancelled(cancel);
        check(std::isfinite(filters[i]) && filters[i]>=0,"stt_invalid_filter");bank[i]=filters[i];
    }
    for(std::size_t k=0;k<frequencies;++k){
        cancelled(cancel);
        for(std::size_t n=0;n<fft;++n){
            const double phase=2*std::numbers::pi*double(k*n)/fft;
            tables[k*fft+n]=std::cos(phase);
            tables[frequencies*fft+k*fft+n]=-std::sin(phase);
        }
    }
    for(std::size_t n=0;n<fft;++n)
        tables[2*frequencies*fft+n]=.5-.5*std::cos(2*std::numbers::pi*double(n)/fft);
    cancelled(cancel);resources_->loaded(admission.h);
    tables_=std::move(tables);filters_=std::move(bank);bins_=bins;resident_=admission.h;admission.h=0;
}
void LogMel::unload(){
    if(!resident_)return;
    resources_->begin_eviction(resident_);
    filters_.reset();tables_.reset();bins_=0;
    resources_->released(resident_);resident_=0;
}
void LogMel::execute(std::span<const float> pcm,std::span<float> output,
        const std::atomic_bool& cancel,const std::function<void(std::size_t)>& observed){
    cancelled(cancel);check(loaded(),"stt_not_loaded");
    const auto count=frames(pcm.size());check(output.size()==bins_*count,"stt_output_shape");
    const auto in=reinterpret_cast<std::uintptr_t>(pcm.data()),out=reinterpret_cast<std::uintptr_t>(output.data());
    check(out>=in+pcm.size_bytes() || in>=out+output.size_bytes(),"stt_buffer_overlap");
    for(float v:pcm)check(std::isfinite(v),"stt_nonfinite_input");
    Pin pin(*resources_,resident_);Reservation scratch(*resources_,host(scratch_bytes));
    std::array<double,fft> windowed;
    std::array<double,frequencies> power;
    for(std::size_t frame=0;frame<count;++frame){
        if(observed)observed(frame);
        cancelled(cancel);
        for(std::size_t n=0;n<fft;++n){
            auto index=static_cast<std::ptrdiff_t>(frame*hop+n)-static_cast<std::ptrdiff_t>(fft/2);
            if(index<0)index=-index;
            if(index>=static_cast<std::ptrdiff_t>(pcm.size()))index=2*static_cast<std::ptrdiff_t>(pcm.size())-2-index;
            windowed[n]=pcm[static_cast<std::size_t>(index)]*tables_[2*frequencies*fft+n];
        }
        for(std::size_t k=0;k<frequencies;++k){
            cancelled(cancel);double real=0,imaginary=0;
            for(std::size_t n=0;n<fft;++n){
                real+=windowed[n]*tables_[k*fft+n];
                imaginary+=windowed[n]*tables_[frequencies*fft+k*fft+n];
            }
            power[k]=real*real+imaginary*imaginary;
        }
        for(std::size_t band=0;band<bins_;++band){
            cancelled(cancel);double energy=0;
            for(std::size_t k=0;k<frequencies;++k)energy+=filters_[band*frequencies+k]*power[k];
            const float value=static_cast<float>(std::log10(std::max(energy,1e-10)));
            check(std::isfinite(value),"stt_nonfinite_output");output[band*count+frame]=value;
        }
    }
    const float floor=*std::max_element(output.begin(),output.end())-8;
    for(auto& value:output){cancelled(cancel);value=(std::max(value,floor)+4)/4;}
    cancelled(cancel);
}
}
