#include "kadan/tts.hpp"
#include "kadan/checkpoint.hpp"
#include <array>
#include <bit>
#include <cmath>

namespace kadan::tts {
namespace {
void check(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void cancelled(const std::atomic_bool& flag) { check(!flag.load(), "tts_cancelled"); }
float bfloat(std::uint16_t bits) {
    float value=std::bit_cast<float>(std::uint32_t(bits)<<16);
    check(std::isfinite(value),"tts_nonfinite_weight");
    return value;
}
struct Reservation {
    Resources& ledger;
    Handle handle;
    Reservation(Resources& r, Footprint bytes) : ledger(r), handle(r.reserve(Workload::tts, std::move(bytes))) {}
    ~Reservation() { if (handle) ledger.released(handle); }
};
struct Pin {
    Resources& ledger; Handle handle;
    Pin(Resources& r, Handle h) : ledger(r), handle(h) { ledger.pin(handle); }
    ~Pin() { ledger.unpin(handle); }
};
}
TextProjection::TextProjection(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "tts_resources_required");
}
TextProjection::~TextProjection() { unload(); }
Footprint TextProjection::host(Bytes bytes) const {
    auto footprint = resources_->snapshot().capacity;
    for (auto& value : footprint) value = 0;
    footprint[0] = bytes;
    return footprint;
}
void TextProjection::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "tts_already_loaded");
    Reservation metadata(*resources_, host(metadata_bytes));
    auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root, basename, budget, {1024*1024, 2048, 4096});
    struct Spec { const char* name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,4> specs{{
        {"talker.text_projection.linear_fc1.weight", {2048,2048}, 2},
        {"talker.text_projection.linear_fc1.bias", {2048}, 1},
        {"talker.text_projection.linear_fc2.weight", {2048,2048}, 2},
        {"talker.text_projection.linear_fc2.bias", {2048}, 1}
    }};
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        check(tensor.dtype == checkpoint::Dtype::bf16 && tensor.rank == spec.rank, "tts_tensor_layout");
        for (std::size_t i=0; i<spec.rank; ++i) check(tensor.shape[i] == spec.shape[i], "tts_tensor_layout");
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
                weights[destination++] = bfloat(std::uint16_t(buffer[i]) | (std::uint16_t(buffer[i+1]) << 8));
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
void TextProjection::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    weights_.reset();
    resources_->released(resident_);
    resident_ = 0;
}
std::array<float,TextProjection::hidden> TextProjection::execute(std::span<const float> input,
        const std::atomic_bool& cancel,const std::function<void()>& observed) {
    cancelled(cancel);
    check(loaded(),"tts_not_loaded");
    check(input.size()==hidden,"tts_input_shape");
    for(float v:input) check(std::isfinite(v),"tts_nonfinite_input");
    Pin pin(*resources_,resident_);
    Reservation scratch(*resources_,host(2*hidden*sizeof(float)));
    if(observed) observed();
    std::array<float,hidden> intermediate,result;
    const float* first=weights_.get(), *first_bias=first+hidden*hidden,
        *second=first_bias+hidden, *second_bias=second+hidden*hidden;
    auto linear=[&](std::span<const float> values,const float* weights,const float* bias,
                    std::span<float> output,bool activation) {
        for(std::size_t r=0;r<hidden;++r) {
            cancelled(cancel);
            float value=bias[r];
            for(std::size_t c=0;c<hidden;++c) value+=weights[r*hidden+c]*values[c];
            check(std::isfinite(value),"tts_nonfinite_output");
            // Stable SiLU; negative tail avoids overflow in exp(-value).
            if(activation) {
                double v=value;
                value=float(v>=0 ? v/(1+std::exp(-v)) : v*std::exp(v)/(1+std::exp(v)));
            }
            check(std::isfinite(value),"tts_nonfinite_output");output[r]=value;
        }
    };
    linear(input,first,first_bias,intermediate,true);
    linear(intermediate,second,second_bias,result,false);
    cancelled(cancel);
    return result;
}
} // namespace kadan::tts
