#include "kadan/video.hpp"
#include "kadan/checkpoint.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

namespace kadan::video {
namespace {
void check(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void cancelled(const std::atomic_bool& flag) { check(!flag.load(), "video_cancelled"); }
float half(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31;
    const auto fraction = bits & 1023;
    check(exponent != 31, "video_nonfinite_weight");
    float value = exponent ? std::ldexp(float(1024 + fraction), exponent - 25)
                           : std::ldexp(float(fraction), -24);
    return bits & 32768 ? -value : value;
}
struct Reservation {
    Resources& ledger;
    Handle handle;
    Reservation(Resources& r, Footprint bytes) : ledger(r), handle(r.reserve(Workload::video, std::move(bytes))) {}
    ~Reservation() { if (handle) ledger.released(handle); }
};
struct Pin {
    Resources& ledger; Handle handle;
    Pin(Resources& r, Handle h) : ledger(r), handle(h) { ledger.pin(handle); }
    ~Pin() { ledger.unpin(handle); }
};
// Trusted output directory; unique sibling temp, atomic no-overwrite publication.
struct Artifact {
    std::string temporary;
    int fd = -1;
    explicit Artifact(const std::string& path) : temporary(path + ".partial-XXXXXX") {
        fd = mkstemp(temporary.data());
        check(fd >= 0, "video_artifact_open");
    }
    ~Artifact() { if (fd >= 0) close(fd); if (!temporary.empty()) unlink(temporary.c_str()); }
    void write(const void* data, std::size_t bytes) {
        const auto* p = static_cast<const char*>(data);
        while (bytes) {
            auto n = ::write(fd, p, bytes);
            if (n < 0 && errno == EINTR) continue;
            check(n > 0, "video_artifact_write");
            p += n; bytes -= n;
        }
    }
    void publish(const std::string& path) {
        check(fsync(fd) == 0, "video_artifact_sync");
        check(link(temporary.c_str(), path.c_str()) == 0, "video_artifact_publish");
    }
};
}
H3DecoderInput::H3DecoderInput(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "video_resources_required");
}
H3DecoderInput::~H3DecoderInput() { unload(); }
Footprint H3DecoderInput::host(Bytes bytes) const {
    auto footprint = resources_->snapshot().capacity;
    for (auto& value : footprint) value = 0;
    footprint[0] = bytes;
    return footprint;
}
void H3DecoderInput::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    Reservation metadata(*resources_, host(metadata_bytes));
    auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root, basename, budget, {1024*1024, 2048, 4096});
    load_from(shard,cancel);
}
void H3DecoderInput::load_from(checkpoint::Shard& shard, const std::atomic_bool& cancel) {
    cancelled(cancel);check(!loaded(),"video_already_loaded");
    struct Spec { const char* name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,6> specs{{
        {"latents_mean", {24}, 1}, {"latents_std", {24}, 1},
        {"post_quant_conv.weight", {24,24,1,1,1}, 5}, {"post_quant_conv.bias", {24}, 1},
        {"decoder.x_embedder.weight", {2048,24}, 2}, {"decoder.x_embedder.bias", {2048}, 1}
    }};
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        check(tensor.dtype == checkpoint::Dtype::fp16 && tensor.rank == spec.rank, "video_tensor_layout");
        for (std::size_t i=0; i<spec.rank; ++i) check(tensor.shape[i] == spec.shape[i], "video_tensor_layout");
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
    for (std::size_t i=24; i<48; ++i) check(weights[i] > 0, "video_latent_std");
    shard.check_unchanged();
    cancelled(cancel);
    resources_->loaded(allocation.handle);
    weights_ = std::move(weights);
    resident_ = allocation.handle;
    allocation.handle = 0;
}
void H3DecoderInput::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    weights_.reset();
    resources_->released(resident_);
    resident_ = 0;
}
void H3DecoderInput::execute(std::span<const float> normalized, const std::string& output,
                           const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_token) {
    cancelled(cancel);check(loaded(),"video_not_loaded");
    Pin publication(*resources_,resident_);
    std::unique_ptr<Artifact> artifact;
    compute(normalized,cancel,on_token,[&](std::span<const float> row) {
        if(!artifact) {
            artifact=std::make_unique<Artifact>(output);
            const auto header="KADAN_H3_DECODER_INPUT_V1\n"+std::to_string(normalized.size()/channels)+" 2048\nF32LE\n";
            artifact->write(header.data(),header.size());
        }
        artifact->write(row.data(),row.size_bytes());
    });
    cancelled(cancel);artifact->publish(output);
}
void H3DecoderInput::compute(std::span<const float> normalized, const std::atomic_bool& cancel,
    const std::function<void(std::size_t)>& on_token, const std::function<void(std::span<const float>)>& sink) {
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!normalized.empty() && normalized.size()%channels == 0, "video_input_shape");
    check(normalized.size()/channels <= max_tokens, "video_input_limit");
    Pin pin(*resources_, resident_);
    Reservation scratch(*resources_, host((hidden+channels*2)*sizeof(float)));
    const auto tokens = normalized.size()/channels;
    std::array<float,channels> latent, projected;
    std::array<float,hidden> row;
    const float* mean=weights_.get(), *stddev=mean+24, *conv=stddev+24, *bias=conv+576,
               *embed=bias+24, *embed_bias=embed+49152;
    for (std::size_t token=0; token<tokens; ++token) {
        cancelled(cancel);
        for (std::size_t c=0; c<channels; ++c) {
            check(std::isfinite(normalized[token*channels+c]), "video_nonfinite_input");
            latent[c] = normalized[token*channels+c]*stddev[c]+mean[c];
            check(std::isfinite(latent[c]), "video_nonfinite_output");
        }
        for (std::size_t r=0; r<channels; ++r) {
            float value=bias[r];
            for (std::size_t c=0; c<channels; ++c) value += conv[r*channels+c]*latent[c];
            check(std::isfinite(value), "video_nonfinite_output");
            projected[r]=value;
        }
        for (std::size_t r=0; r<hidden; ++r) {
            float value=embed_bias[r];
            for (std::size_t c=0; c<channels; ++c) value += embed[r*channels+c]*projected[c];
            check(std::isfinite(value), "video_nonfinite_output");
            row[r]=value;
        }
        sink(row);
        if (on_token) on_token(token+1);
    }
    cancelled(cancel);
}
H3DecoderQkv::H3DecoderQkv(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "video_resources_required");
}
H3DecoderQkv::~H3DecoderQkv() { unload(); }
Footprint H3DecoderQkv::host(Bytes bytes) const {
    auto footprint = resources_->snapshot().capacity;
    for (auto& value : footprint) value = 0;
    footprint[0] = bytes;
    return footprint;
}
void H3DecoderQkv::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    Reservation metadata(*resources_, host(metadata_bytes));
    auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root, basename, budget, {1024*1024, 2048, 4096});
    load_from(shard,cancel);
}
void H3DecoderQkv::load_from(checkpoint::Shard& shard, const std::atomic_bool& cancel, std::size_t block) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    check(block<H3VideoDecoder::layers,"video_block_index");
    const auto prefix="decoder.transformer_blocks."+std::to_string(block)+".";
    struct Spec { std::string name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,3> specs{{
        {prefix+"norm1.weight", {2048}, 1},
        {prefix+"attn.to_qkv.weight", {6144,2048}, 2},
        {prefix+"attn.to_qkv.bias", {6144}, 1}
    }};
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        check(tensor.dtype == checkpoint::Dtype::fp16 && tensor.rank == spec.rank, "video_tensor_layout");
        for (std::size_t i=0; i<spec.rank; ++i) check(tensor.shape[i] == spec.shape[i], "video_tensor_layout");
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
void H3DecoderQkv::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    weights_.reset();
    resources_->released(resident_);
    resident_ = 0;
}
void H3DecoderQkv::execute(std::span<const float> input, const std::string& output,
                         const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_rows) {
    cancelled(cancel);check(loaded(),"video_not_loaded");
    Pin publication(*resources_,resident_);
    std::unique_ptr<Artifact> artifact;
    compute(input, cancel, on_rows, [&](std::span<const float> row) {
        if (!artifact) {
            artifact=std::make_unique<Artifact>(output);
            const auto header="KADAN_H3_DECODER_QKV_V1\n"+std::to_string(input.size()/hidden)+" 32 3 64\nF32LE\n";
            artifact->write(header.data(),header.size());
        }
        artifact->write(row.data(),row.size_bytes());
    });
    cancelled(cancel);artifact->publish(output);
}
void H3DecoderQkv::compute(std::span<const float> input, const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_rows,
    const std::function<void(std::span<const float>)>& sink) {
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!input.empty() && input.size()%hidden == 0, "video_input_shape");
    const auto tokens = input.size()/hidden;
    check(tokens <= max_tokens, "video_input_limit");
    Pin pin(*resources_, resident_);
    Reservation scratch(*resources_, host(scratch_bytes));
    std::array<float,hidden> normalized;
    std::array<float,width> row;
    const auto* norm=weights_.get();
    const auto* matrix=norm+hidden;
    const auto* bias=matrix+width*hidden;
    for (std::size_t token=0; token<tokens; ++token) {
        cancelled(cancel);
        float squares=0;
        for (std::size_t c=0; c<hidden; ++c) {
            const float value=input[token*hidden+c];
            check(std::isfinite(value), "video_nonfinite_input");
            squares += value*value;
        }
        check(std::isfinite(squares), "video_nonfinite_output");
        const float inverse=1.0f/std::sqrt(squares/float(hidden)+1e-5f);
        for (std::size_t c=0; c<hidden; ++c) {
            normalized[c]=(input[token*hidden+c]*inverse)*norm[c];
            check(std::isfinite(normalized[c]), "video_nonfinite_output");
        }
        for (std::size_t r=0; r<width; ++r) {
            if (r%64==0) cancelled(cancel);
            float value=bias[r];
            for (std::size_t c=0; c<hidden; ++c) value += matrix[r*hidden+c]*normalized[c];
            check(std::isfinite(value), "video_nonfinite_output");
            row[r]=value;
            if ((r+1)%64==0 && on_rows) on_rows(token*width+r+1);
        }
        cancelled(cancel);
        sink(row);
    }
    cancelled(cancel); // Caller publishes only after every row succeeds.
}
H3QkRope::H3QkRope(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "video_resources_required");
}
H3QkRope::~H3QkRope() { unload(); }
Footprint H3QkRope::host(Bytes bytes) const {
    auto footprint=resources_->snapshot().capacity;
    for (auto& value:footprint) value=0;
    footprint[0]=bytes;
    return footprint;
}
void H3QkRope::load(const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    Reservation allocation(*resources_,host(resident_bytes));
    auto frequencies=std::make_unique<float[]>(8);
    for (std::size_t i=0;i<8;++i) frequencies[i]=1.0f/std::pow(100.0f,float(i)/8.0f);
    cancelled(cancel);
    resources_->loaded(allocation.handle);
    frequencies_=std::move(frequencies);
    resident_=allocation.handle; allocation.handle=0;
}
void H3QkRope::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    frequencies_.reset();
    resources_->released(resident_); resident_=0;
}
void H3QkRope::execute(std::span<const float> qkv, std::span<const float> coordinates,
                     const std::string& output, const std::atomic_bool& cancel,
                     const std::function<void(std::size_t)>& on_heads) {
    cancelled(cancel);check(loaded(),"video_not_loaded");
    Pin publication(*resources_,resident_);
    std::unique_ptr<Artifact> artifact;
    compute(qkv, coordinates, cancel, on_heads, [&](std::span<const float> row) {
        if (!artifact) {
            artifact=std::make_unique<Artifact>(output);
            const auto header="KADAN_H3_QK_ROPE_V1\n"+std::to_string(qkv.size()/width)+" 32 3 64\nF32LE\n";
            artifact->write(header.data(),header.size());
        }
        artifact->write(row.data(),row.size_bytes());
    });
    cancelled(cancel);artifact->publish(output);
}
void H3QkRope::compute(std::span<const float> qkv, std::span<const float> coordinates, const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_heads,
    const std::function<void(std::span<const float>)>& sink, std::size_t token_limit) {
    static_assert(std::endian::native == std::endian::little && sizeof(float)==4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!qkv.empty() && qkv.size()%width==0, "video_input_shape");
    const auto tokens=qkv.size()/width;
    check(tokens<=token_limit, "video_input_limit");
    check(coordinates.size()==tokens*3, "video_coordinate_shape");
    Pin pin(*resources_,resident_);
    Reservation scratch(*resources_,host(scratch_bytes));
    std::array<float,24> cosine,sine;
    std::array<float,192> row;
    constexpr float two_pi=6.2831853071795864769f;
    for (std::size_t token=0;token<tokens;++token) {
        cancelled(cancel);
        for (std::size_t axis=0;axis<3;++axis) {
            const auto coordinate=coordinates[token*3+axis];
            check(std::isfinite(coordinate) && coordinate>=-1 && coordinate<=1, "video_coordinate_range");
            for (std::size_t i=0;i<8;++i) {
                const float angle=(two_pi*coordinate)*frequencies_[i];
                cosine[axis*8+i]=std::cos(angle); sine[axis*8+i]=std::sin(angle);
            }
        }
        for (std::size_t head=0;head<heads;++head) {
            cancelled(cancel);
            const auto begin=token*width+head*192;
            for (std::size_t i=0;i<192;++i) {
                const float value=qkv[begin+i];
                check(std::isfinite(value), "video_nonfinite_input");
                row[i]=value;
            }
            for (std::size_t kind=0;kind<2;++kind) {
                const auto offset=kind*head_dim;
                float squares=0;
                for (std::size_t i=0;i<head_dim;++i) squares+=row[offset+i]*row[offset+i];
                check(std::isfinite(squares), "video_nonfinite_output");
                const float inverse=1.0f/std::sqrt(squares/64.0f+1e-5f);
                for (std::size_t i=0;i<head_dim;++i) row[offset+i]*=inverse;
                // NeoX split-half rotation over 48 channels, not adjacent pairs.
                for (std::size_t i=0;i<24;++i) {
                    const float x=row[offset+i],y=row[offset+i+24];
                    row[offset+i]=x*cosine[i]-y*sine[i];
                    row[offset+i+24]=y*cosine[i]+x*sine[i];
                }
            }
            for (float value:row) check(std::isfinite(value), "video_nonfinite_output");
            if (on_heads) on_heads(token*heads+head+1);
            cancelled(cancel);
            sink(row);
        }
    }
    cancelled(cancel);
}
H3DecoderAttention::H3DecoderAttention(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "video_resources_required");
}
H3DecoderAttention::~H3DecoderAttention() { unload(); }
Footprint H3DecoderAttention::host(Bytes bytes) const {
    auto footprint = resources_->snapshot().capacity;
    for (auto& value : footprint) value = 0;
    footprint[0] = bytes;
    return footprint;
}
void H3DecoderAttention::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    Reservation metadata(*resources_, host(metadata_bytes));
    auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root, basename, budget, {1024*1024, 2048, 4096});
    load_from(shard,cancel);
}
void H3DecoderAttention::load_from(checkpoint::Shard& shard, const std::atomic_bool& cancel, std::size_t block) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    check(block<H3VideoDecoder::layers,"video_block_index");
    const auto prefix="decoder.transformer_blocks."+std::to_string(block)+".";
    struct Spec { std::string name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,2> specs{{
        {prefix+"attn.to_out.weight", {2048,2048}, 2},
        {prefix+"attn.to_out.bias", {2048}, 1}
    }};
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        check(tensor.dtype == checkpoint::Dtype::fp16 && tensor.rank == spec.rank, "video_tensor_layout");
        for (std::size_t i=0; i<spec.rank; ++i) check(tensor.shape[i] == spec.shape[i], "video_tensor_layout");
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
void H3DecoderAttention::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    weights_.reset();
    resources_->released(resident_);
    resident_ = 0;
}
void H3DecoderAttention::execute(std::span<const float> qkv, const std::string& output,
                                const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_rows) {
    cancelled(cancel);check(loaded(),"video_not_loaded");
    Pin publication(*resources_,resident_);
    std::unique_ptr<Artifact> artifact;
    compute(qkv, cancel, on_rows, [&](std::span<const float> row) {
        if (!artifact) {
            artifact=std::make_unique<Artifact>(output);
            const auto header="KADAN_H3_ATTENTION_V1\n"+std::to_string(qkv.size()/width)+" 2048\nF32LE\n";
            artifact->write(header.data(),header.size());
        }
        artifact->write(row.data(),row.size_bytes());
    });
    cancelled(cancel);artifact->publish(output);
}
void H3DecoderAttention::compute(std::span<const float> qkv, const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_rows,
    const std::function<void(std::span<const float>)>& sink) {
    static_assert(std::endian::native == std::endian::little && sizeof(float)==4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!qkv.empty() && qkv.size()%width==0, "video_input_shape");
    const auto tokens=qkv.size()/width;
    check(tokens<=max_tokens, "video_input_limit");
    Pin pin(*resources_,resident_);
    Reservation scratch(*resources_,host(scratch_bytes));
    // Validate every Q/K/V even if an underflowed softmax would hide it.
    for (std::size_t i=0;i<qkv.size();++i) {
        if (i%192==0) cancelled(cancel);
        check(std::isfinite(qkv[i]), "video_nonfinite_input");
    }
    std::array<float,max_tokens> probabilities;
    std::array<float,hidden> attended,projected;
    for (std::size_t token=0;token<tokens;++token) {
        for (std::size_t head=0;head<heads;++head) {
            cancelled(cancel);
            float maximum=-INFINITY;
            for (std::size_t key=0;key<tokens;++key) {
                float dot=0;
                for (std::size_t c=0;c<head_dim;++c)
                    dot+=qkv[token*width+head*192+c]*qkv[key*width+head*192+64+c];
                check(std::isfinite(dot), "video_nonfinite_output");
                probabilities[key]=dot*0.125f;
                maximum=std::max(maximum,probabilities[key]);
            }
            float denominator=0;
            for (std::size_t key=0;key<tokens;++key) {
                probabilities[key]=std::exp(probabilities[key]-maximum);
                denominator+=probabilities[key];
            }
            for (std::size_t key=0;key<tokens;++key) probabilities[key]/=denominator;
            for (std::size_t c=0;c<head_dim;++c) {
                float value=0;
                for (std::size_t key=0;key<tokens;++key)
                    value+=probabilities[key]*qkv[key*width+head*192+128+c];
                check(std::isfinite(value), "video_nonfinite_output");
                attended[head*head_dim+c]=value;
            }
        }
        const auto* bias=weights_.get()+hidden*hidden;
        for (std::size_t r=0;r<hidden;++r) {
            if (r%64==0) cancelled(cancel);
            float value=bias[r];
            for (std::size_t c=0;c<hidden;++c) value+=weights_[r*hidden+c]*attended[c];
            check(std::isfinite(value), "video_nonfinite_output");
            projected[r]=value;
            if ((r+1)%64==0 && on_rows) on_rows(token*hidden+r+1);
        }
        cancelled(cancel);
        sink(projected);
    }
    cancelled(cancel);
}
H3DecoderFeedForward::H3DecoderFeedForward(std::shared_ptr<Resources> resources) : resources_(std::move(resources)) {
    check(bool(resources_), "video_resources_required");
}
H3DecoderFeedForward::~H3DecoderFeedForward() { unload(); }
Footprint H3DecoderFeedForward::host(Bytes bytes) const {
    auto footprint = resources_->snapshot().capacity;
    for (auto& value : footprint) value = 0;
    footprint[0] = bytes;
    return footprint;
}
void H3DecoderFeedForward::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    Reservation metadata(*resources_, host(metadata_bytes));
    auto budget = std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root, basename, budget, {1024*1024, 2048, 4096});
    load_from(shard,cancel);
}
void H3DecoderFeedForward::load_from(checkpoint::Shard& shard, const std::atomic_bool& cancel, std::size_t block) {
    cancelled(cancel);
    check(!loaded(), "video_already_loaded");
    check(block<H3VideoDecoder::layers,"video_block_index");
    const auto prefix="decoder.transformer_blocks."+std::to_string(block)+".";
    struct Spec { std::string name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,7> specs{{
        {prefix+"scale1", {2048}, 1},
        {prefix+"norm2.weight", {2048}, 1},
        {prefix+"ff.w1.weight", {16384,2048}, 2},
        {prefix+"ff.w1.bias", {16384}, 1},
        {prefix+"ff.w2.weight", {2048,8192}, 2},
        {prefix+"ff.w2.bias", {2048}, 1},
        {prefix+"scale2", {2048}, 1}
    }};
    for (const auto& spec : specs) {
        auto tensor = shard.tensor(spec.name);
        check(tensor.dtype == checkpoint::Dtype::fp16 && tensor.rank == spec.rank, "video_tensor_layout");
        for (std::size_t i=0; i<spec.rank; ++i) check(tensor.shape[i] == spec.shape[i], "video_tensor_layout");
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
void H3DecoderFeedForward::unload() {
    if (!resident_) return;
    resources_->begin_eviction(resident_);
    weights_.reset();
    resources_->released(resident_);
    resident_ = 0;
}
void H3DecoderFeedForward::execute(std::span<const float> residual, std::span<const float> attention,
                                  const std::string& output, const std::atomic_bool& cancel,
                                  const std::function<void(std::size_t)>& on_rows) {
    cancelled(cancel);check(loaded(),"video_not_loaded");
    Pin publication(*resources_,resident_);
    std::unique_ptr<Artifact> artifact;
    compute(residual, attention, cancel, on_rows, [&](std::span<const float> row) {
        if (!artifact) {
            artifact=std::make_unique<Artifact>(output);
            const auto header="KADAN_H3_FEED_FORWARD_V1\n"+std::to_string(residual.size()/hidden)+" 2048\nF32LE\n";
            artifact->write(header.data(),header.size());
        }
        artifact->write(row.data(),row.size_bytes());
    });
    cancelled(cancel);artifact->publish(output);
}
void H3DecoderFeedForward::compute(std::span<const float> residual, std::span<const float> attention, const std::atomic_bool& cancel, const std::function<void(std::size_t)>& on_rows,
    const std::function<void(std::span<const float>)>& sink) {
    static_assert(std::endian::native == std::endian::little && sizeof(float)==4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!residual.empty() && residual.size()%hidden==0, "video_input_shape");
    const auto tokens=residual.size()/hidden;
    check(tokens<=max_tokens, "video_input_limit");
    check(attention.size()==residual.size(), "video_attention_shape");
    Pin pin(*resources_,resident_);
    Reservation scratch(*resources_,host(scratch_bytes));
    std::array<float,hidden> sum,normalized,result;
    std::array<float,inner> gate,values;
    const auto* scale1=weights_.get(); const auto* norm=scale1+hidden;
    const auto* w1=norm+hidden; const auto* b1=w1+expanded*hidden;
    const auto* w2=b1+expanded; const auto* b2=w2+hidden*inner;
    const auto* scale2=b2+hidden;
    for (std::size_t token=0;token<tokens;++token) {
        cancelled(cancel);
        float squares=0;
        for (std::size_t c=0;c<hidden;++c) {
            const auto x=residual[token*hidden+c],a=attention[token*hidden+c];
            check(std::isfinite(x) && std::isfinite(a), "video_nonfinite_input");
            sum[c]=x+a*scale1[c];
            squares+=sum[c]*sum[c];
        }
        check(std::isfinite(squares), "video_nonfinite_output");
        const float inverse=1.0f/std::sqrt(squares/float(hidden)+1e-5f);
        for (std::size_t c=0;c<hidden;++c) {
            normalized[c]=(sum[c]*inverse)*norm[c];
            check(std::isfinite(normalized[c]), "video_nonfinite_output");
        }
        for (std::size_t row=0;row<expanded;++row) {
            if(row%64==0) cancelled(cancel);
            float value=b1[row];
            for(std::size_t c=0;c<hidden;++c) value+=w1[row*hidden+c]*normalized[c];
            check(std::isfinite(value), "video_nonfinite_output");
            (row<inner ? gate[row] : values[row-inner])=value;
            if((row+1)%64==0 && on_rows) on_rows(token*(expanded+hidden)+row+1);
        }
        cancelled(cancel);
        for(std::size_t c=0;c<inner;++c) {
            // Stable SiLU, first half is the gate, second half the value.
            const float x=gate[c];
            const float activation=x>=0 ? x/(1.0f+std::exp(-x)) : x*std::exp(x)/(1.0f+std::exp(x));
            values[c]=activation*values[c];
            check(std::isfinite(values[c]), "video_nonfinite_output");
        }
        for(std::size_t row=0;row<hidden;++row) {
            if(row%64==0) cancelled(cancel);
            float value=b2[row];
            for(std::size_t c=0;c<inner;++c) value+=w2[row*inner+c]*values[c];
            check(std::isfinite(value), "video_nonfinite_output");
            result[row]=sum[row]+value*scale2[row];
            check(std::isfinite(result[row]), "video_nonfinite_output");
            if((row+1)%64==0 && on_rows) on_rows(token*(expanded+hidden)+expanded+row+1);
        }
        cancelled(cancel);
        sink(result);
    }
    cancelled(cancel);
}
H3DecoderBlock::H3DecoderBlock(std::shared_ptr<Resources> resources)
    : resources_(std::move(resources)),qkv_(resources_),rope_(resources_),attention_(resources_),ff_(resources_) {}
Footprint H3DecoderBlock::host(Bytes bytes) const {
    auto result=resources_->snapshot().capacity;
    for(auto& value:result)value=0;
    result[0]=bytes;return result;
}
void H3DecoderBlock::load(const char* root, const std::string& basename, const std::atomic_bool& cancel) {
    cancelled(cancel);check(!executing_,"busy");check(!loaded(),"video_already_loaded");
    // All weight reads share one O_NOFOLLOW descriptor, including its mutation
    // checks. Replacing the pathname between stages cannot mix checkpoint files.
    Reservation metadata(*resources_,host(metadata_bytes));
    auto budget=std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    checkpoint::Shard shard(root,basename,budget,{1024*1024,2048,4096});
    load_from(shard,cancel,0);
}
void H3DecoderBlock::load_from(checkpoint::Shard& shard, const std::atomic_bool& cancel, std::size_t block) {
    cancelled(cancel);check(!executing_,"busy");check(!loaded(),"video_already_loaded");
    const auto prefix="decoder.transformer_blocks."+std::to_string(block)+".";
    const std::array<const char*,4> names{"attn.to_qkv.weight","attn.to_out.weight","ff.w1.weight","ff.w2.weight"};
    std::array<WeightIdentity,4> identities{};
    for(std::size_t i=0;i<names.size();++i)identities[i]=shard.tensor_identity(prefix+names[i]);
    try {
        qkv_.load_from(shard,cancel,block);rope_.load(cancel);
        attention_.load_from(shard,cancel,block);ff_.load_from(shard,cancel,block);
        shard.check_unchanged();cancelled(cancel);
        weight_identities_=identities;
    } catch(...) {unload();throw;}
}
void H3DecoderBlock::unload() {
    check(!executing_,"busy");
    ff_.unload();attention_.unload();rope_.unload();qkv_.unload();
}
void H3DecoderBlock::execute(std::span<const float> input, std::span<const float> coordinates,
    const std::string& output, const std::atomic_bool& cancel, const Hook& hook) {
    std::unique_ptr<Artifact> artifact;
    compute(input,coordinates,cancel,hook,[&](std::span<const float> row) {
        if(!artifact) {
            artifact=std::make_unique<Artifact>(output);
            const auto header="KADAN_H3_BLOCK_V1\n"+std::to_string(input.size()/hidden)+" 2048\nF32LE\n";
            artifact->write(header.data(),header.size());
        }
        artifact->write(row.data(),row.size_bytes());
    });
    cancelled(cancel);artifact->publish(output);
}
void H3DecoderBlock::compute(std::span<const float> input, std::span<const float> coordinates,
    const std::atomic_bool& cancel, const Hook& hook, const std::function<void(std::span<const float>)>& output, H3Compute* accelerator) {
    cancelled(cancel);check(!executing_,"busy");check(loaded(),"video_not_loaded");
    check(!input.empty() && input.size()%hidden==0,"video_input_shape");
    const auto tokens=input.size()/hidden;
    check(tokens<=(accelerator?28224+5:max_tokens),"video_input_limit");
    check(coordinates.size()==tokens*3,"video_coordinate_shape");
    for(float value:coordinates)check(std::isfinite(value)&&value>=-1&&value<=1,"video_coordinate_range");
    // Keep every resident pinned through the final publication, including hooks.
    Pin qpin(*resources_,qkv_.resident_),rpin(*resources_,rope_.resident_);
    Pin apin(*resources_,attention_.resident_),fpin(*resources_,ff_.resident_);
    struct Active {bool& flag;explicit Active(bool& f):flag(f){flag=true;}~Active(){flag=false;}} active(executing_);
    Reservation intermediate(*resources_,host(intermediate_bytes(tokens)));
    auto buffers=std::make_unique<float[]>(intermediate_bytes(tokens)/sizeof(float));
    auto q=std::span<float>(buffers.get(),tokens*6144);
    auto r=std::span<float>(buffers.get()+tokens*6144,tokens*6144);
    auto a=std::span<float>(buffers.get()+tokens*6144*2,tokens*hidden);
    auto observe=[&](const char* stage){return [&,stage](std::size_t count){if(hook)hook(stage,count);};};
    auto sink=[](std::span<float> destination){return [destination,offset=std::size_t{0}](std::span<const float> row) mutable {
        check(offset<=destination.size() && row.size()<=destination.size()-offset,"video_output_shape");
        std::copy(row.begin(),row.end(),destination.begin()+offset);offset+=row.size();
    };};
    if(accelerator) {
        auto normalize=[&](std::span<const float> x,const float* weight,std::span<float> y){
            for(std::size_t t=0;t<tokens;++t){cancelled(cancel);float squares=0;for(std::size_t c=0;c<hidden;++c)squares+=x[t*hidden+c]*x[t*hidden+c];const float inverse=1/std::sqrt(squares/float(hidden)+1e-5f);for(std::size_t c=0;c<hidden;++c)y[t*hidden+c]=(x[t*hidden+c]*inverse)*weight[c];}
        };
        const auto* qnorm=qkv_.weights_.get();const auto* qw=qnorm+hidden;const auto* qb=qw+6144*hidden;
        normalize(input,qnorm,a);accelerator->dense_weight(weight_identities_[0],{qw,6144*hidden},{qb,6144},a,hidden,6144,q,cancel);
        rope_.compute(q,coordinates,cancel,observe("rope"),sink(r),28224+5);
        auto query=q.first(tokens*hidden),key=q.subspan(tokens*hidden,tokens*hidden),value=q.subspan(2*tokens*hidden,tokens*hidden);
        for(std::size_t t=0;t<tokens;++t)for(std::size_t h=0;h<32;++h)for(std::size_t c=0;c<64;++c){const auto at=(t*32+h)*64+c,source=(t*32+h)*192+c;query[at]=r[source];key[at]=r[source+64];value[at]=r[source+128];}
        accelerator->attention(query,key,value,tokens,32,32,64,false,a,cancel);
        const auto* aw=attention_.weights_.get();auto attended=r.first(tokens*hidden);
        accelerator->dense_weight(weight_identities_[1],{aw,hidden*hidden},{aw+hidden*hidden,hidden},a,hidden,hidden,attended,cancel);
        const auto* scale1=ff_.weights_.get();const auto* norm=scale1+hidden;const auto* w1=norm+hidden;const auto* b1=w1+16384*hidden;const auto* w2=b1+16384;const auto* b2=w2+hidden*8192;const auto* scale2=b2+hidden;
        auto sum=q.first(tokens*hidden);for(std::size_t i=0;i<sum.size();++i)sum[i]=input[i]+attended[i]*scale1[i%hidden];normalize(sum,norm,a);
        Reservation mlp_admission(*resources_,host(tokens*16384*sizeof(float)));auto expanded=std::make_unique<float[]>(tokens*16384);std::span<float> mlp(expanded.get(),tokens*16384);
        accelerator->dense_weight(weight_identities_[2],{w1,16384*hidden},{b1,16384},a,hidden,16384,mlp,cancel);
        for(std::size_t t=0;t<tokens;++t){cancelled(cancel);for(std::size_t c=0;c<8192;++c){const float x=mlp[t*16384+c];const float activation=x>=0?x/(1+std::exp(-x)):x*std::exp(x)/(1+std::exp(x));mlp[t*8192+c]=activation*mlp[t*16384+8192+c];}}
        accelerator->dense_weight(weight_identities_[3],{w2,hidden*8192},{b2,hidden},mlp.first(tokens*8192),8192,hidden,a,cancel);
        for(std::size_t i=0;i<a.size();++i){a[i]=sum[i]+a[i]*scale2[i%hidden];check(std::isfinite(a[i]),"video_nonfinite_output");}
        cancelled(cancel);output(a);return;
    }
    qkv_.compute(input,cancel,observe("qkv"),sink(q));
    rope_.compute(q,coordinates,cancel,observe("rope"),sink(r));
    attention_.compute(r,cancel,observe("attention"),sink(a));
    ff_.compute(input,a,cancel,observe("feed_forward"),output);
    cancelled(cancel);
}
H3VideoDecoder::H3VideoDecoder(std::shared_ptr<Resources> resources,std::shared_ptr<H3Compute> compute)
    : resources_(std::move(resources)),compute_(std::move(compute)),input_(resources_) {}
H3VideoDecoder::~H3VideoDecoder() { unload(); }
Footprint H3VideoDecoder::host(Bytes bytes) const {
    auto result=resources_->snapshot().capacity;
    for(auto& value:result)value=0;
    result[0]=bytes;return result;
}
void H3VideoDecoder::load(const char* root, const std::string& basename, const std::atomic_bool& cancel,std::shared_ptr<checkpoint::ReadCache> cache) {
    cancelled(cancel);check(!executing_,"busy");check(!loaded(),"video_already_loaded");
    Reservation metadata(*resources_,host(H3DecoderInput::metadata_bytes));
    auto budget=std::make_shared<checkpoint::MemoryBudget>(H3DecoderInput::metadata_bytes);
    auto shard=std::make_unique<checkpoint::Shard>(root,basename,budget,checkpoint::Limits{1024*1024,2048,4096});shard->cache_reads(std::move(cache),&cancel);
    struct Spec {const char* name;std::array<std::uint64_t,3> shape;std::size_t rank;};
    const std::array<Spec,5> specs{{
        {"decoder.register_tokens",{1,4,hidden},3},
        {"decoder.norm_out.weight",{hidden},1},{"decoder.norm_out.bias",{hidden},1},
        {"decoder.proj_out.weight",{patch_values,hidden},2},{"decoder.proj_out.bias",{patch_values},1}
    }};
    for(const auto& spec:specs) {
        const auto t=shard->tensor(spec.name);
        check(t.dtype==checkpoint::Dtype::fp16 && t.rank==spec.rank,"video_tensor_layout");
        for(std::size_t i=0;i<spec.rank;++i)check(t.shape[i]==spec.shape[i],"video_tensor_layout");
    }
    Reservation allocation(*resources_,host(weight_bytes));
    auto weights=std::make_unique<float[]>(weight_bytes/sizeof(float));
    Reservation staging(*resources_,host(4096));
    std::array<std::uint8_t,4096> buffer;
    std::size_t destination=0;
    for(const auto& spec:specs) {
        const auto t=shard->tensor(spec.name);
        for(std::size_t offset=0;offset<t.bytes;) {
            cancelled(cancel);
            const auto count=std::min<std::size_t>(buffer.size(),t.bytes-offset);
            shard->read_tensor(spec.name,offset,{buffer.data(),count});
            for(std::size_t i=0;i<count;i+=2)
                weights[destination++]=half(std::uint16_t(buffer[i])|(std::uint16_t(buffer[i+1])<<8));
            offset+=count;
        }
    }
    try {
        input_.load_from(*shard,cancel);
        shard->check_unchanged();cancelled(cancel);
        resources_->loaded(allocation.handle);
        resident_=allocation.handle;allocation.handle=0;
        metadata_=metadata.handle;metadata.handle=0;
        weights_=std::move(weights);shard_=std::move(shard);
    } catch(...) {input_.unload();throw;}
}
void H3VideoDecoder::unload() {
    check(!executing_,"busy");
    input_.unload();
    if(resident_) {
        resources_->begin_eviction(resident_);weights_.reset();
        resources_->released(resident_);resident_=0;
    }
    shard_.reset();
    if(metadata_) {resources_->released(metadata_);metadata_=0;}
}
void H3VideoDecoder::execute(std::span<const float> normalized, std::size_t time, std::size_t height,
    std::size_t width, const std::string& output, const std::atomic_bool& cancel, const Hook& hook) {
    cancelled(cancel);check(!executing_,"busy");check(loaded(),"video_not_loaded");
    // Check before multiplication, including hostile dimensions and empty axes.
    const auto token_limit=compute_?28224:max_tokens;
    check(time && height && width && time<=token_limit && height<=token_limit/time &&
          width<=token_limit/(time*height),"video_latent_shape_or_limit");
    const auto patches=time*height*width,tokens=patches+5;
    check(normalized.size()==patches*24,"video_input_shape");
    Pin resident(*resources_,resident_),input_pin(*resources_,input_.resident_);
    struct Active {bool& flag;explicit Active(bool& f):flag(f){flag=true;}~Active(){flag=false;}} active(executing_);
    const Bytes values=tokens*(2*hidden+3)+patches*patch_values;
    Reservation admission(*resources_,host(values*sizeof(float)));
    auto memory=std::make_unique<float[]>(values);
    auto current=std::span<float>(memory.get(),tokens*hidden);
    auto next=std::span<float>(memory.get()+tokens*hidden,tokens*hidden);
    auto coordinates=std::span<float>(memory.get()+tokens*hidden*2,tokens*3);
    auto frames=std::span<float>(memory.get()+tokens*(hidden*2+3),patches*patch_values);
    auto sink=[](std::span<float> destination){return [destination,offset=std::size_t{0}](std::span<const float> row) mutable {
        check(offset<=destination.size() && row.size()<=destination.size()-offset,"video_output_shape");
        std::copy(row.begin(),row.end(),destination.begin()+offset);offset+=row.size();
    };};
    shard_->check_unchanged();
    if(compute_){
        Reservation input_admission(*resources_,host(patches*48*sizeof(float)));auto scratch=std::make_unique<float[]>(patches*48);std::span<float> latent(scratch.get(),patches*24),projected(scratch.get()+patches*24,patches*24);
        const auto* mean=input_.weights_.get();const auto* deviation=mean+24;const auto* conv=deviation+24;const auto* bias=conv+576;const auto* embed=bias+24;const auto* embed_bias=embed+49152;
        for(std::size_t i=0;i<latent.size();++i)latent[i]=normalized[i]*deviation[i%24]+mean[i%24];
        compute_->dense_weight(shard_->tensor_identity("post_quant_conv.weight"),{conv,576},{bias,24},latent,24,24,projected,cancel);compute_->dense_weight(shard_->tensor_identity("decoder.x_embedder.weight"),{embed,49152},{embed_bias,2048},projected,24,2048,current.first(patches*hidden),cancel);
    }else input_.compute(normalized,cancel,[&](std::size_t count){if(hook)hook("input",count);},sink(current.first(patches*hidden)));
    std::copy_n(weights_.get(),4*hidden,current.data()+patches*hidden);
    // Fifth suffix is the inference-only zero class token, not mask_token.
    std::fill(current.begin()+(patches+4)*hidden,current.end(),0.0f);
    for(std::size_t t=0;t<time;++t)for(std::size_t h=0;h<height;++h)for(std::size_t w=0;w<width;++w) {
        const auto i=((t*height+h)*width+w)*3;
        coordinates[i]=2.0f*((float(t)+0.5f)/float(time))-1.0f;
        coordinates[i+1]=2.0f*((float(h)+0.5f)/float(height))-1.0f;
        coordinates[i+2]=2.0f*((float(w)+0.5f)/float(width))-1.0f;
    }
    std::fill(coordinates.begin()+patches*3,coordinates.end(),0.0f);
    for(std::size_t layer=0;layer<layers;++layer) {
        cancelled(cancel);
        // This is real checkpoint I/O on each block, not disk advertised as RAM.
        H3DecoderBlock block(resources_);
        block.load_from(*shard_,cancel,layer);
        if(hook)hook("block_loaded",layer);
        block.compute(current,coordinates,cancel,[&](const char* stage,std::size_t count){if(hook)hook(stage,count);},sink(next),compute_.get());
        block.unload();std::swap(current,next);
        if(hook)hook("block_completed",layer+1);
    }
    const auto* norm=weights_.get()+4*hidden;
    const auto* bias=norm+hidden;
    const auto* projection=bias+hidden;
    const auto* projection_bias=projection+patch_values*hidden;
    Reservation scratch(*resources_,host(hidden*sizeof(float)));
    std::array<float,hidden> row;
    const auto video_t=time*4,video_h=height*16,video_w=width*16;
    Reservation output_admission(*resources_,host(compute_?patches*patch_values*sizeof(float):0));
    std::unique_ptr<float[]> gpu_patches;if(compute_)gpu_patches=std::make_unique<float[]>(patches*patch_values);
    for(std::size_t patch=0;patch<patches;++patch) {
        cancelled(cancel);
        // LayerNorm, not the RMSNorm used inside the transformer blocks.
        float mean=0;
        for(std::size_t c=0;c<hidden;++c)mean+=current[patch*hidden+c];
        mean/=float(hidden);
        float variance=0;
        for(std::size_t c=0;c<hidden;++c){const float d=current[patch*hidden+c]-mean;variance+=d*d;}
        const float inverse=1.0f/std::sqrt(variance/float(hidden)+1e-5f);
        for(std::size_t c=0;c<hidden;++c) {
            row[c]=((current[patch*hidden+c]-mean)*inverse)*norm[c]+bias[c];
            check(std::isfinite(row[c]),"video_nonfinite_output");
        }
        if(compute_){std::copy(row.begin(),row.end(),next.begin()+patch*hidden);continue;}
        const auto lt=patch/(height*width),lh=(patch/width)%height,lw=patch%width;
        for(std::size_t r=0;r<patch_values;++r) {
            if(r%64==0)cancelled(cancel);
            float value=projection_bias[r];
            for(std::size_t c=0;c<hidden;++c)value+=projection[r*hidden+c]*row[c];
            check(std::isfinite(value),"video_nonfinite_output");
            const auto channel=r/1024,pt=(r/256)%4,ph=(r/16)%16,pw=r%16;
            frames[((channel*video_t+lt*4+pt)*video_h+lh*16+ph)*video_w+lw*16+pw]=value;
            if((r+1)%64==0 && hook)hook("output",patch*patch_values+r+1);
        }
    }
    if(compute_){
        std::span<float> projected(gpu_patches.get(),patches*patch_values);compute_->dense_weight(shard_->tensor_identity("decoder.proj_out.weight"),{projection,patch_values*hidden},{projection_bias,patch_values},next.first(patches*hidden),hidden,patch_values,projected,cancel);
        for(std::size_t patch=0;patch<patches;++patch){cancelled(cancel);const auto lt=patch/(height*width),lh=(patch/width)%height,lw=patch%width;for(std::size_t r=0;r<patch_values;++r){const auto channel=r/1024,pt=(r/256)%4,ph=(r/16)%16,pw=r%16;frames[((channel*video_t+lt*4+pt)*video_h+lh*16+ph)*video_w+lw*16+pw]=projected[patch*patch_values+r];}}
    }
    cancelled(cancel);shard_->check_unchanged();
    Artifact artifact(output);
    const auto header="KADAN_H3_FRAMES_V1\n3 "+std::to_string(video_t)+" "+std::to_string(video_h)+" "+std::to_string(video_w)+"\nF32LE\n";
    artifact.write(header.data(),header.size());artifact.write(frames.data(),frames.size_bytes());
    if(hook)hook("publish",patches);
    cancelled(cancel);shard_->check_unchanged();artifact.publish(output);
}

} // namespace kadan::video
