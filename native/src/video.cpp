#include "kadan/video.hpp"
#include "kadan/checkpoint.hpp"
#include <array>
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
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!normalized.empty() && normalized.size()%channels == 0, "video_input_shape");
    check(normalized.size()/channels <= max_tokens, "video_input_limit");
    Pin pin(*resources_, resident_);
    Reservation scratch(*resources_, host((hidden+channels*2)*sizeof(float)));
    Artifact artifact(output);
    const auto tokens = normalized.size()/channels;
    const std::string header = "KADAN_H3_DECODER_INPUT_V1\n" + std::to_string(tokens) + " 2048\nF32LE\n";
    artifact.write(header.data(), header.size());
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
        artifact.write(row.data(), sizeof(row));
        if (on_token) on_token(token+1);
    }
    cancelled(cancel); // Link below is the completion/cancellation linearization point.
    artifact.publish(output);
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
    struct Spec { const char* name; std::array<std::uint64_t,5> shape; std::size_t rank; };
    const std::array<Spec,3> specs{{
        {"decoder.transformer_blocks.0.norm1.weight", {2048}, 1},
        {"decoder.transformer_blocks.0.attn.to_qkv.weight", {6144,2048}, 2},
        {"decoder.transformer_blocks.0.attn.to_qkv.bias", {6144}, 1}
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
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    cancelled(cancel);
    check(loaded(), "video_not_loaded");
    check(!input.empty() && input.size()%hidden == 0, "video_input_shape");
    const auto tokens = input.size()/hidden;
    check(tokens <= max_tokens, "video_input_limit");
    Pin pin(*resources_, resident_);
    Reservation scratch(*resources_, host(scratch_bytes));
    Artifact artifact(output);
    const auto header = "KADAN_H3_DECODER_QKV_V1\n" + std::to_string(tokens) + " 32 3 64\nF32LE\n";
    artifact.write(header.data(), header.size());
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
        artifact.write(row.data(), sizeof(row));
    }
    cancelled(cancel); // Atomic no-overwrite publication is the completion boundary.
    artifact.publish(output);
}
} // namespace kadan::video
