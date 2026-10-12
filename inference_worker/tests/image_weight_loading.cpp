// Checkpoint ingress/ownership with a fake device; no CUDA or projection execution.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unistd.h>

#include "kadan/image.hpp"

void check(bool b) {
    if (!b)
        throw std::runtime_error("test_failed");
}

// Checkpoint ingress and allocation ownership only; no projection callbacks.
struct Device final : kadan::MediaDeviceOperations {
    kadan::Bytes available = 32 * 1024 * 1024, uploaded = 0;
    std::map<void*, kadan::Bytes> allocations;
    std::size_t probes = 0;

    std::size_t allocation_alignment() const override {
        return 256;
    }

    void* stream(int) override {
        return nullptr;
    }

    kadan::Bytes free_memory(int) override {
        ++probes;
        return available;
    }

    bool direct_peer(int, int) override {
        return false;
    }

    void* allocate(int, kadan::Bytes bytes, std::size_t alignment) override {
        check(probes >= 2);
        void* p = ::operator new(bytes, std::align_val_t(alignment));
        allocations.emplace(p, bytes);
        available -= bytes;
        return p;
    }

    void free(int, void* p) override {
        available += allocations.at(p);
        allocations.erase(p);
        ::operator delete(p, std::align_val_t(256));
    }

    void wait(int, kadan::MediaEvent) override {
    }

    kadan::MediaEvent record(int d) override {
        return {d, 1};
    }

    void synchronize(int) override {
    }

    void release_events(int) override {
    }

    void upload(int, void*, std::span<const std::byte> raw) override {
        uploaded += raw.size();
        for (auto byte : raw)
            check(byte == std::byte{0});
    }

    void download(int, std::span<std::byte>, const void*) override {
        throw std::runtime_error("unexpected_download");
    }

    void peer(int, const void*, int, void*, kadan::Bytes) override {
        throw std::runtime_error("unexpected_peer_copy");
    }

    std::uint32_t read_status(int, const void*) override {
        throw std::runtime_error("unexpected_execution");
    }
};

void write_weights(const char* directory, bool f32) {
    std::ostringstream header;
    header << '{';
    std::size_t bytes = 0, count = 0;
    auto add = [&](std::string name, bool vector) {
        if (count++)
            header << ',';
        const auto size = (vector ? 2 : 4) * (f32 ? 4 : 2);
        header << "\"transformer_blocks.7." << name << ".weight\":{\"dtype\":\""
               << (f32 ? "F32" : "BF16")
               << "\",\"shape\":" << (vector ? "[2]" : "[2,2]") << ",\"data_offsets\":["
               << bytes << ',' << bytes + size << "]}";
        bytes += size;
    };
    for (auto name :
         {"attn.to_q",
          "attn.to_k",
          "attn.to_v",
          "attn.to_out.0",
          "img_mlp.proj",
          "img_mlp.gate_layer",
          "img_mlp.out"})
        add(name, false);
    for (auto name : {"attn.norm_q", "attn.norm_k"})
        add(name, true);
    header << '}';
    const auto text = header.str();
    {
        std::ofstream file(
            std::string(directory) + "/weights.safetensors", std::ios::binary
        );
        for (unsigned i = 0; i < 8; ++i)
            file.put(char(std::uint64_t(text.size()) >> (8 * i)));
        file << text;
        for (std::size_t i = 0; i < bytes; ++i)
            file.put(0);
    }
}

int main() {
    char directory[] = "/tmp/kadan-weight-load-XXXXXX";
    check(mkdtemp(directory));

    struct Cleanup {
        const char* path;

        ~Cleanup() {
            std::filesystem::remove_all(path);
        }
    } cleanup{directory};

    write_weights(directory, false);
    auto resources = std::make_shared<kadan::Resources>(
        kadan::Footprint{64 * 1024 * 1024, 32 * 1024 * 1024}
    );
    kadan::image::TransformerBlock block(resources);
    std::atomic_bool cancel = false;
    block.load(directory, "weights.safetensors", 7, {2, 1, 2, 2, {0, 0, 2}}, cancel);
    check(resources->snapshot().used[0] > 0);
    auto backend = std::make_unique<Device>();
    auto& device = *backend;
    auto activity = std::make_shared<kadan::MediaSessionActivity>();
    kadan::MediaSession session(
        resources, activity, std::move(backend), kadan::Workload::image, {{0}}, cancel
    );
    check(!block.prepare_feed_forward(session, 5, cancel));
    const auto& plan = session.plan();
    check(plan.weights.size() == 3 && device.uploaded == 24);
    for (const auto& slice : plan.weights) {
        check(slice.name.starts_with("transformer_blocks.7.img_mlp."));
        check(slice.stored_bytes == 8 && slice.identity.inode != 0);
        check(slice.first_row == 0 && slice.rows == 2);
    }
    check(session.evidence().budgets[0].physical_free == 32 * 1024 * 1024);
    auto normalized = session.tensor(
        0, "resident/normalized", kadan::DeviceDType::f32, {5, 2}, {2, 1}
    );
    check(normalized.layout().bytes == 40);
    normalized = {};
    session.advance(kadan::MediaPhase::ff);
    auto aggregate =
        session.tensor(0, "resident/status", kadan::DeviceDType::u32, {1}, {1});
    auto gate = session.tensor(0, "ff/gate-status", kadan::DeviceDType::u32, {1}, {1});
    auto up = session.tensor(0, "ff/up-status", kadan::DeviceDType::u32, {1}, {1});
    check(aggregate.layout().data != gate.layout().data);
    check(up.layout().data != gate.layout().data);
    check(up.layout().data != aggregate.layout().data);
    aggregate = {};
    gate = {};
    up = {};
    session.finish();
    check(resources->snapshot().used[1] == 0 && device.allocations.empty());
    block.unload();
    check(resources->snapshot().used[0] == 0);
    write_weights(directory, true);
    block.load(directory, "weights.safetensors", 7, {2, 1, 2, 2, {0, 0, 2}}, cancel);
    auto unsupported_backend = std::make_unique<Device>();
    auto& unsupported_device = *unsupported_backend;
    kadan::MediaSession unsupported(
        resources,
        activity,
        std::move(unsupported_backend),
        kadan::Workload::image,
        {{0}},
        cancel
    );
    const auto error = block.prepare_feed_forward(unsupported, 5, cancel);
    check(error && error->reason == kadan::PlanFailure::unsupported_layout);
    check(!unsupported_device.probes && !unsupported_device.uploaded);
    check(unsupported_device.allocations.empty());
    unsupported.finish();
    block.unload();
    check(resources->snapshot().used == kadan::Footprint({0, 0}));
    std::cout << "qualified block checkpoint names and release passed (no execution)\n";
}
