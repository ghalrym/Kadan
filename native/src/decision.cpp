#include "kadan/decision.hpp"
#include "kadan/checkpoint.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

namespace kadan::decision {
namespace {
void check(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void cancelled(const std::atomic_bool& flag) { check(!flag.load(), "decision_cancelled"); }
float half(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const auto fraction = bits & 1023;
    check(exponent != 31, "decision_nonfinite_weight");
    float value = exponent ? std::ldexp(float(1024 + fraction), exponent - 25)
                           : std::ldexp(float(fraction), -24);
    return bits & 32768 ? -value : value;
}
struct Reservation {
    Resources& ledger;
    Handle handle;
    Reservation(Resources& r, Footprint bytes) : ledger(r), handle(r.reserve(Workload::decision, std::move(bytes))) {}
    ~Reservation() { if (handle) ledger.released(handle); }
};
struct Pin {
    Resources& ledger; Handle handle;
    Pin(Resources& r, Handle h) : ledger(r), handle(h) { ledger.pin(handle); }
    ~Pin() { ledger.unpin(handle); }
};
}
TerminalHeads::TerminalHeads(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "decision_resources_required");
}
TerminalHeads::~TerminalHeads() { unload(); }
Footprint TerminalHeads::host(Bytes bytes) const {
    auto footprint = resources_->snapshot().capacity;
    for (auto& value : footprint) value = 0;
    footprint[0] = bytes;
    return footprint;
}
void TerminalHeads::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "decision_already_loaded");
    Reservation metadata(*resources_, host(metadata_bytes));
    auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root, basename, budget, {1024*1024, 2048, 4096});
    struct Spec { const char* name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,10> specs{{
        {"scorer.0.weight", {1024}, 1}, {"scorer.0.bias", {1024}, 1},
        {"scorer.1.weight", {1024,1024}, 2}, {"scorer.1.bias", {1024}, 1},
        {"scorer.3.weight", {1,1024}, 2}, {"scorer.3.bias", {1}, 1},
        {"act_head.0.weight", {256,1028}, 2}, {"act_head.0.bias", {256}, 1},
        {"act_head.2.weight", {2,256}, 2}, {"act_head.2.bias", {2}, 1}
    }};
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        check(tensor.dtype == checkpoint::Dtype::fp16 && tensor.rank == spec.rank, "decision_tensor_layout");
        for (std::size_t i=0; i<spec.rank; ++i) check(tensor.shape[i] == spec.shape[i], "decision_tensor_layout");
    }
    Reservation allocation(*resources_, host(weight_bytes));
    auto weights = std::make_unique<float[]>(weight_bytes / sizeof(float));
    // Fixed 4-KiB stack transfer staging is explicitly charged.
    Reservation staging(*resources_, host(4096));
    std::array<std::uint8_t,4096> buffer;
    std::size_t destination = 0;
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        for (std::size_t offset=0; offset<tensor.bytes;) {
            cancelled(cancel);
            auto count = std::min<std::size_t>(buffer.size(), tensor.bytes-offset);
            shard.read_tensor(spec.name, offset, {buffer.data(), count});
            for (std::size_t i=0; i<count; i+=2)
                weights[destination++] = half(std::uint16_t(buffer[i]) | (std::uint16_t(buffer[i+1]) << 8));
            offset += count;
        }
    }
    shard.check_unchanged();
    cancelled(cancel);
    resources_->loaded(allocation.handle);
    weights_ = std::move(weights);
    resident_ = allocation.handle;
    allocation.handle = 0;
}
void TerminalHeads::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    weights_.reset();
    resources_->released(resident_);
    resident_ = 0;
}
std::array<float,3> TerminalHeads::execute(std::span<const float> marker, std::span<const float> action,
        const std::atomic_bool& cancel,const std::function<void()>& observed) {
    cancelled(cancel);
    check(loaded(),"decision_not_loaded");
    check(marker.size()==hidden && action.size()==action_input,"decision_input_shape");
    for(auto input:{marker,action}) for(float v:input) check(std::isfinite(v),"decision_nonfinite_input");
    Pin pin(*resources_,resident_);
    Reservation scratch(*resources_,host((2*hidden+256+3)*sizeof(float)));
    if(observed) observed();
    std::array<float,hidden> normalized, intermediate;
    std::array<float,256> action_hidden;
    std::array<float,3> result;
    const float* norm=weights_.get(), *norm_bias=norm+1024, *score_w=norm_bias+1024,
        *score_b=score_w+1024*1024, *score_out=score_b+1024, *score_bias=score_out+1024,
        *act_w=score_bias+1, *act_b=act_w+256*1028, *act_out=act_b+256, *act_bias=act_out+2*256;
    // LayerNorm epsilon and erf GELU match pinned laya.common.DecisionModel.
    double mean=0,variance=0;
    for(float v:marker) mean+=v;
    mean/=hidden;
    for(float v:marker) {double d=v-mean;variance+=d*d;}
    const double inverse=1/std::sqrt(variance/hidden+1e-5);
    for(std::size_t i=0;i<hidden;++i) normalized[i]=float((marker[i]-mean)*inverse)*norm[i]+norm_bias[i];
    auto linear=[&](std::span<const float> input, const float* weights,const float* bias,
                    std::span<float> output,bool activation) {
        for(std::size_t r=0;r<output.size();++r) {
            cancelled(cancel);
            float v=bias[r];
            for(std::size_t c=0;c<input.size();++c) v+=weights[r*input.size()+c]*input[c];
            if(activation) v=float(0.5*double(v)*(1+std::erf(double(v)/std::sqrt(2.0))));
            check(std::isfinite(v),"decision_nonfinite_output");output[r]=v;
        }
    };
    linear(normalized,score_w,score_b,intermediate,true);
    linear(intermediate,score_out,score_bias,{result.data(),1},false);
    linear(action,act_w,act_b,action_hidden,true);
    linear(action_hidden,act_out,act_bias,{result.data()+1,2},false);
    cancelled(cancel);
    return result;
}
} // namespace kadan::decision
