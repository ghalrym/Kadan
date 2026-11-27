#include "kadan/device_weights.hpp"
#include "kadan/weight_slab_pool.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
using namespace kadan;

void check(bool b) {
    if (!b)
        throw std::runtime_error("weight_test_failed");
}

template <class F> void fails(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("expected_failure");
}

struct Fake final : WeightDeviceOperations {
    Resources& resources;
    std::map<void*, Bytes> pointers;
    unsigned copies = 0, frees = 0, device_copies = 0;
    std::vector<std::string> events;
    std::vector<std::size_t> device_offsets, device_sizes;
    std::vector<const void*> device_sources;
    std::function<void()> on_device_copy, on_sync;
    std::string fail;
    bool fail_free = false;
    std::atomic_bool* cancellation = nullptr;
    Bytes free_bytes = 1024;
    bool oom = false, synchronized = false;
    std::uintptr_t next_pointer = 0;

    Bytes available_weight_bytes() override {
        return free_bytes;
    }

    explicit Fake(Resources& r) : resources(r) {
    }

    void* allocate_weight(Bytes bytes) override {
        events.emplace_back("allocate");
        check(
            resources.snapshot().used[1] >= bytes
            && resources.snapshot().used[0] >= DeviceWeights::metadata_bytes
        );
        if (oom)
            return nullptr;
        if (fail == "allocate")
            throw std::runtime_error("allocate");
        // Distinct fake handles; these are never dereferenced or sent to a GPU.
        auto pointer = reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
            ++next_pointer
        );
        pointers.emplace(pointer, bytes);
        return pointer;
    }

    void copy_weight(
        void* p, std::size_t offset, std::span<const std::byte> source
    ) override {
        check(
            pointers.contains(p) && source.size() <= 1024 * 1024
            && offset + source.size() <= pointers.at(p)
        );
        ++copies;
        events.emplace_back("copy_host");
        if (cancellation)
            cancellation->store(true);
        if (fail == "copy")
            throw std::runtime_error("copy");
    }

    void copy_weight_device(
        void* destination, std::size_t offset, const void* source, std::size_t bytes
    ) override {
        check(
            pointers.contains(destination) && bytes <= 1024 * 1024
            && offset + bytes <= pointers.at(destination)
        );
        ++device_copies;
        events.emplace_back("copy_device");
        device_offsets.push_back(offset);
        device_sizes.push_back(bytes);
        device_sources.push_back(source);
        if (on_device_copy)
            on_device_copy();
        if (cancellation)
            cancellation->store(true);
        if (fail == "copy_device")
            throw std::runtime_error("copy_device");
    }

    void free_weight(void* p) override {
        check(synchronized);
        synchronized = false;
        events.emplace_back("free");
        if (fail == "free" || fail_free)
            throw std::runtime_error("free");
        check(pointers.erase(p) == 1);
        ++frees;
    }

    void synchronize_weights() override {
        events.emplace_back("sync");
        if (on_sync)
            on_sync();
        synchronized = true;
        if (fail == "sync")
            throw std::runtime_error("sync");
    }
};

void device_copy_admission_and_publication() {
    constexpr Bytes host = DeviceWeights::metadata_bytes;
    constexpr Bytes device_capacity = 8 * 1024 * 1024;
    Resources resources({host, device_capacity});
    Fake gpu(resources);
    gpu.free_bytes = device_capacity;
    DeviceWeights bank(resources, 0, Workload::llm, device_capacity, gpu);
    const DeviceWeights::Key key{{.device = 1, .inode = 7}, 10, 70001, 34};
    std::vector<std::byte> scratch(key.count * key.width, std::byte{0x5a});
    std::atomic_bool cancel = false;
    gpu.on_device_copy = [&] {
        check(!bank.find(key));
    };
    gpu.on_sync = [&] {
        check(!bank.find(key));
    };
    auto* retained = bank.retain_device(key, scratch.data(), scratch.size(), cancel);
    check(retained && bank.find(key) == retained && gpu.copies == 0);
    check(
        gpu.events
        == std::vector<std::string>(
            {"allocate", "copy_device", "copy_device", "copy_device", "sync"}
        )
    );
    check(gpu.device_copies == 3 && bank.allocated() == scratch.size());
    constexpr std::size_t chunk = 1024 * 1024;
    check(gpu.device_offsets == std::vector<std::size_t>({0, chunk, 2 * chunk}));
    check(
        gpu.device_sizes
        == std::vector<std::size_t>({chunk, chunk, scratch.size() - 2 * chunk})
    );
    for (const auto* source : gpu.device_sources)
        check(source == scratch.data());
    const auto events_before_hit = gpu.events;
    gpu.free_bytes = 0;
    check(bank.retain_device(key, scratch.data(), scratch.size(), cancel) == retained);
    check(gpu.events == events_before_hit && gpu.device_copies == 3);
    gpu.on_device_copy = {};
    gpu.on_sync = {};
    bank.release();
    check(gpu.pointers.empty() && resources.snapshot().used == Footprint({0, 0}));
}

void device_copy_refusal_preserves_scratch() {
    constexpr Bytes host = DeviceWeights::metadata_bytes;
    Resources resources({host, 1024});
    Fake gpu(resources);
    DeviceWeights bank(resources, 0, Workload::llm, 64, gpu);
    std::array<std::byte, 64> scratch;
    scratch.fill(std::byte{0x6b});
    const auto original_scratch = scratch;
    std::atomic_bool cancel = false;
    DeviceWeights::Key key{{.device = 1}, 0, 16, 4};
    auto* retained = bank.retain_device(key, scratch.data(), scratch.size(), cancel);
    check(retained && gpu.device_copies == 1 && gpu.copies == 0);
    auto missing = key;
    missing.first = 16;
    const auto events_before_refusal = gpu.events;
    check(!bank.retain_device(missing, scratch.data(), scratch.size(), cancel));
    check(!bank.find(missing) && bank.find(key) == retained);
    check(gpu.events == events_before_refusal && scratch == original_scratch);
    check(bank.allocated() == 64 && gpu.pointers.size() == 1);
    bank.release();
    check(resources.snapshot().used == Footprint({0, 0}));

    for (const bool oom : {false, true}) {
        Resources next_resources({host, 1024});
        Fake next_gpu(next_resources);
        DeviceWeights next_bank(next_resources, 0, Workload::llm, 64, next_gpu);
        next_gpu.free_bytes = oom ? 1024 : 0;
        next_gpu.oom = oom;
        check(!next_bank.retain_device(key, scratch.data(), scratch.size(), cancel));
        check(
            next_gpu.device_copies == 0 && next_gpu.pointers.empty()
            && next_bank.allocated() == 0 && !next_bank.quarantined()
        );
        next_bank.release();
        check(next_resources.snapshot().used == Footprint({0, 0}));
    }
}

void device_copy_failures_preserve_ownership() {
    constexpr Bytes host = DeviceWeights::metadata_bytes;
    const DeviceWeights::Key key{{.device = 1}, 0, 8, 4};
    std::array<std::byte, 32> scratch{};
    std::atomic_bool cancel = false;
    for (const auto* failure : {"allocate", "copy_device"}) {
        Resources resources({host, 1024});
        Fake gpu(resources);
        DeviceWeights bank(resources, 0, Workload::llm, 64, gpu);
        gpu.fail = failure;
        fails([&] { bank.retain_device(key, scratch.data(), scratch.size(), cancel); });
        check(!bank.quarantined() && bank.allocated() == 0 && gpu.pointers.empty());
        check(!bank.find(key));
        if (gpu.fail == "copy_device")
            check(
                gpu.events
                == std::vector<std::string>({"allocate", "copy_device", "sync", "free"})
            );
        gpu.fail.clear();
        bank.release();
        check(resources.snapshot().used == Footprint({0, 0}));
    }
    for (const auto* failure : {"sync", "free"}) {
        Resources resources({host, 1024});
        Fake gpu(resources);
        DeviceWeights bank(resources, 0, Workload::llm, 64, gpu);
        gpu.fail = std::string(failure) == "sync" ? "sync" : "copy_device";
        gpu.fail_free = std::string(failure) == "free";
        bool combined = false;
        try {
            bank.retain_device(key, scratch.data(), scratch.size(), cancel);
        } catch (const OperationCleanupError& error) {
            combined = true;
            check(exception_message(error.operation) == gpu.fail);
            check(exception_message(error.cleanup) == failure);
        }
        check(combined && bank.quarantined() && bank.allocated() == scratch.size());
        check(
            gpu.pointers.size() == 1
            && resources.snapshot().used == Footprint({host, 64})
        );
        fails([&] { bank.find(key); });
        fails([&] { bank.release(); });
    }
    {
        constexpr Bytes capacity = 4 * 1024 * 1024;
        Resources resources({host, capacity});
        Fake gpu(resources);
        gpu.free_bytes = capacity;
        DeviceWeights bank(resources, 0, Workload::llm, capacity, gpu);
        auto large = key;
        large.count = (2 * 1024 * 1024 + 32) / large.width;
        std::vector<std::byte> large_scratch(large.count * large.width);
        gpu.cancellation = &cancel;
        fails([&] {
            bank.retain_device(
                large, large_scratch.data(), large_scratch.size(), cancel
            );
        });
        check(
            cancel.load() && gpu.device_copies == 1
            && gpu.device_sizes[0] == 1024 * 1024 && gpu.pointers.empty()
            && !bank.quarantined() && bank.allocated() == 0
        );
        cancel = false;
        bank.release();
        check(resources.snapshot().used == Footprint({0, 0}));
    }
    {
        Resources resources({host, 1024});
        Fake gpu(resources);
        DeviceWeights bank(resources, 0, Workload::llm, 64, gpu);
        cancel = true;
        fails([&] { bank.retain_device(key, scratch.data(), scratch.size(), cancel); });
        cancel = false;
        fails([&] { bank.retain_device(key, nullptr, scratch.size(), cancel); });
        fails([&] {
            bank.retain_device(key, scratch.data(), scratch.size() - 1, cancel);
        });
        check(gpu.events.empty() && resources.snapshot().used == Footprint({0, 0}));
    }
}

void weight_slab_pool_tests() {
    constexpr Bytes mib = 1024ULL * 1024;
    struct FakeRaw {
        std::map<void*, Bytes> live;
        std::uintptr_t cursor = 0x10000000ULL;
        Bytes cap = ~Bytes(0);
        Bytes used = 0;
        unsigned alloc_calls = 0, free_calls = 0;
        bool fail_free = false;

        void* alloc(Bytes bytes) {
            ++alloc_calls;
            if (bytes > cap - used)
                return nullptr;
            // Keep fake slab addresses 256-byte aligned and strictly disjoint.
            const Bytes step = ((bytes + 4095) / 4096) * 4096;
            auto* ptr = reinterpret_cast<void*>(cursor);  // NOLINT(performance-no-int-to-ptr)
            cursor += step;
            live.emplace(ptr, bytes);
            used += bytes;
            return ptr;
        }

        void free(void* ptr) {
            ++free_calls;
            if (fail_free)
                throw std::runtime_error("slab_raw_free");
            const auto it = live.find(ptr);
            check(it != live.end());
            used -= it->second;
            live.erase(it);
        }
    };

    // Invalid configuration rejected.
    {
        FakeRaw raw;
        auto a = [&](Bytes b) { return raw.alloc(b); };
        auto f = [&](void* p) { raw.free(p); };
        fails([&] { WeightSlabPool({}, f); });
        fails([&] { WeightSlabPool(a, {}); });
        fails([&] { WeightSlabPool(a, f, 0); });
        fails([&] { WeightSlabPool(a, f, 100); });
    }

    // Packing 200 sub-MiB routed expert slices into minimal 32 MiB slabs with
    // 256-byte alignment, disjoint intervals, and immediate slab reclamation.
    {
        FakeRaw raw;
        WeightSlabPool pool(
            [&](Bytes b) { return raw.alloc(b); },
            [&](void* p) { raw.free(p); }
        );
        check(pool.allocate(0) == nullptr);
        pool.free(nullptr);
        fails([&] { pool.free(reinterpret_cast<void*>(0x1234)); });

        constexpr Bytes slice_bytes = 884736;  // 864 KiB IQ4_NL routed expert slice
        std::vector<void*> ptrs;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> intervals;
        for (int i = 0; i < 200; ++i) {
            void* p = pool.allocate(slice_bytes);
            check(p != nullptr);
            const auto addr = reinterpret_cast<std::uintptr_t>(p);
            check(addr % WeightSlabPool::alignment == 0);
            ptrs.push_back(p);
            intervals.emplace_back(addr, addr + slice_bytes);
        }
        // 32 MiB / 864 KiB = 37 slices per slab -> 200 slices fit in 6 slabs!
        check(raw.alloc_calls == 6 && pool.slab_count() == 6 && pool.direct_count() == 0);
        std::sort(intervals.begin(), intervals.end());
        for (std::size_t i = 1; i < intervals.size(); ++i)
            check(intervals[i - 1].second <= intervals[i].first);

        for (void* p : ptrs)
            pool.free(p);
        check(
            pool.slab_count() == 0 && pool.direct_count() == 0
            && pool.largest_free_block() == 0 && raw.live.empty()
            && raw.free_calls == 6
        );
    }

    // Coalescing (left, right, both) and best-fit reuse inside a single slab.
    {
        FakeRaw raw;
        WeightSlabPool pool(
            [&](Bytes b) { return raw.alloc(b); },
            [&](void* p) { raw.free(p); },
            4096,
            2048
        );
        void* a = pool.allocate(512);
        void* b = pool.allocate(512);
        void* c = pool.allocate(512);
        void* d = pool.allocate(512);
        check(pool.slab_count() == 1 && pool.largest_free_block() == 2048);

        // Free a and c (two separate 512-byte holes), then free b to coalesce all three into 1536 bytes.
        pool.free(a);
        pool.free(c);
        check(pool.largest_free_block() == 2048);
        pool.free(b);
        // Best-fit for 1536 must reuse the coalesced [0, 1536) block rather than splitting the 2048 tail!
        void* merged = pool.allocate(1536);
        check(merged == a && pool.slab_count() == 1 && pool.largest_free_block() == 2048);

        pool.free(merged);
        pool.free(d);
        check(pool.slab_count() == 0 && raw.live.empty());
    }

    // Direct allocation for > max_pooled_bytes and OOM fallback when a full slab
    // cannot be allocated but a smaller direct allocation still fits.
    {
        FakeRaw raw;
        WeightSlabPool pool(
            [&](Bytes b) { return raw.alloc(b); },
            [&](void* p) { raw.free(p); },
            4 * mib,
            2 * mib
        );
        void* big = pool.allocate(3 * mib);
        check(big != nullptr && pool.direct_count() == 1 && pool.slab_count() == 0);
        pool.free(big);
        check(pool.direct_count() == 0 && raw.live.empty());

        // Restrict remaining raw capacity to 1 MiB (< 4 MiB slab): small alloc falls back to direct.
        raw.cap = 1 * mib;
        void* fallback = pool.allocate(512 * 1024);
        check(
            fallback != nullptr && pool.slab_count() == 0 && pool.direct_count() == 1
        );
        // Exceeding remaining cap returns nullptr cleanly.
        check(pool.allocate(1 * mib) == nullptr);
        pool.free(fallback);
        check(pool.direct_count() == 0 && raw.live.empty());
    }

    // If raw_free throws on last slice, slab remains tracked rather than leaked silently.
    {
        FakeRaw raw;
        WeightSlabPool pool(
            [&](Bytes b) { return raw.alloc(b); },
            [&](void* p) { raw.free(p); },
            4096,
            2048
        );
        void* p = pool.allocate(512);
        raw.fail_free = true;
        fails([&] { pool.free(p); });
        check(pool.slab_count() == 1 && raw.live.size() == 1);
        raw.fail_free = false;
        pool.free(p);
        check(pool.slab_count() == 0 && raw.live.empty());
    }
}

int main() {
    device_copy_admission_and_publication();
    device_copy_refusal_preserves_scratch();
    device_copy_failures_preserve_ownership();
    weight_slab_pool_tests();
    const Bytes host = DeviceWeights::metadata_bytes;
    std::atomic_bool cancel = false;
    std::array<float, 8> values{};
    auto bytes = std::as_bytes(std::span(values));
    DeviceWeights::Key key{{.device = 1}, 0, 8, 4};
    {
        Resources r({host, 2048});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::llm, 1024, gpu);
        std::array<std::byte, 288> packed{};
        // Identical block ranges distinguish all packed encodings from F32.
        // The callback boundary is a fake: a bank hit must skip source reads.
        unsigned reads = 0;
        auto fetch = [&](const DeviceWeights::Key& tile) {
            if (auto retained = bank.find(tile))
                return retained;
            ++reads;
            return bank.retain(
                tile, std::span(packed).first(tile.count * tile.width), cancel
            );
        };
        std::array<void*, 4> retained{};
        std::size_t i = 0;
        auto tile = key;
        tile.count = 2;
        for (std::size_t width : {110U, 144U, 34U, 4U}) {
            tile.width = width;
            check(!bank.find(tile));
            retained[i] = fetch(tile);
            check(retained[i]);
            for (std::size_t previous = 0; previous < i; ++previous)
                check(retained[previous] != retained[i]);
            check(fetch(tile) == retained[i]);
            ++i;
        }
        check(reads == 4 && gpu.copies == 4 && bank.allocated() == 584);
        auto expert = tile;
        expert.width = 110;
        expert.source.payload_begin = 220;
        expert.source.payload_end = 440;
        check(!bank.find(expert) && fetch(expert));
        check(reads == 5 && gpu.copies == 5 && bank.allocated() == 804);
        auto next_rows = expert;
        next_rows.first = 2;
        check(!bank.find(next_rows) && fetch(next_rows));
        check(bank.allocated() == 1024);
        next_rows.first = 4;
        check(!fetch(next_rows) && gpu.copies == 6 && bank.allocated() == 1024);
        // Cleanup remains available after cancellation and retires exact experts.
        cancel = true;
        check(bank.release_sources(std::span(&expert.source, 1)) == 440);
        fails([&] { bank.retain(tile, std::span(packed).first(8), cancel); });
        cancel = false;
        check(bank.find(tile) == retained.back());
        auto invalid = tile;
        invalid.count = std::numeric_limits<std::size_t>::max();
        invalid.width = 144;
        fails([&] { bank.retain(invalid, packed, cancel); });
        invalid = tile;
        invalid.width = 110;
        fails([&] { bank.retain(invalid, std::span(packed).first(219), cancel); });
        check(bank.allocated() == 584 && gpu.copies == 6);
        bank.release();
        check(gpu.frees == 6 && gpu.pointers.empty() && r.snapshot().residents == 0);
    }
    {
        // Prefill may fill a fixed bank with early-layer experts that decode
        // never selects. Retiring that stage lets later-layer selections warm.
        constexpr Bytes capacity = 2 * (110 + 144 + 34);
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::llm, capacity, gpu);
        std::array<std::byte, 288> packed{};
        std::array<DeviceWeights::Key, 3> prefill{key, key, key};
        constexpr std::array<std::size_t, 3> widths{110, 144, 34};
        for (std::size_t i = 0; i < prefill.size(); ++i) {
            auto& tile = prefill[i];
            tile.source.inode = 77;
            // Distinct expert intervals within one shard, including the same
            // block range: stage retirement matches the complete identity.
            tile.source.payload_begin = i * 4096;
            tile.source.payload_end = (i + 1) * 4096;
            tile.first = 2;
            tile.count = 2;
            tile.width = widths[i];
        }
        unsigned reads = 0;
        auto fetch = [&](const DeviceWeights::Key& tile) {
            if (auto retained = bank.find(tile))
                return retained;
            ++reads;
            return bank.retain(
                tile, std::span(packed).first(tile.count * tile.width), cancel
            );
        };
        for (const auto& tile : prefill)
            check(fetch(tile));
        check(bank.allocated() == capacity && gpu.copies == 3);
        auto later_layer = prefill.front();
        later_layer.source.payload_begin = 12288;
        later_layer.source.payload_end = 16384;
        check(!fetch(later_layer) && !bank.find(later_layer));
        check(bank.allocated() == capacity && gpu.copies == 3);

        const auto admitted = r.snapshot().used;
        const std::array first_experts{prefill[0].source, prefill[1].source};
        const auto unrelated = bank.find(prefill[2]);
        check(bank.release_sources(first_experts) == 2 * (110 + 144));
        check(!bank.find(prefill[0]) && !bank.find(prefill[1]));
        check(bank.find(prefill[2]) == unrelated && bank.allocated() == 68);
        check(bank.release_sources(std::span(&prefill[2].source, 1)) == 68);
        check(bank.allocated() == 0 && gpu.pointers.empty());
        check(
            bank.reserved() == capacity && bank.capacity() == capacity
            && bank.maximum_capacity() == capacity && r.snapshot().used == admitted
            && r.snapshot().residents == 1
        );

        std::array<void*, 3> decoded{};
        auto decode = prefill;
        for (std::size_t i = 0; i < decode.size(); ++i) {
            decode[i].source.payload_begin += 12288;
            decode[i].source.payload_end += 12288;
            decode[i].first = 4 + i * 2;
            decoded[i] = fetch(decode[i]);
            check(decoded[i]);
        }
        check(bank.allocated() == capacity && reads == 7 && gpu.copies == 6);
        // A second decode visit uses the exact retained row/block keys, without
        // another source read or fake device upload.
        for (std::size_t i = 0; i < decode.size(); ++i)
            check(fetch(decode[i]) == decoded[i]);
        check(reads == 7 && gpu.copies == 6 && gpu.frees == 3);
        bank.release();
        check(gpu.frees == 6 && gpu.pointers.empty() && r.snapshot().residents == 0);
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        auto bf16_key = key;
        bf16_key.width = 2;
        std::array<std::uint16_t, 8> raw{};
        auto retained = bank.retain(bf16_key, std::as_bytes(std::span(raw)), cancel);
        check(retained && bank.allocated() == 16 && !bank.find(key));
        check(bank.retain(key, bytes, cancel) != retained);
        check(bank.allocated() == 48 && gpu.copies == 2);
        check(bank.retain(bf16_key, std::as_bytes(std::span(raw)), cancel) == retained);
        check(gpu.copies == 2);
        bank.release();
        check(gpu.frees == 2 && r.snapshot().residents == 0);
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        auto first = bank.retain(key, bytes, cancel);
        check(
            first && gpu.copies == 1 && bank.allocated() == 32 && bank.reserved() == 64
        );
        check(bank.retain(key, bytes, cancel) == first && gpu.copies == 1);
        auto next = key;
        next.first = 8;
        check(bank.retain(next, bytes, cancel));
        next.first = 16;
        check(bank.retain(next, bytes, cancel) == nullptr && bank.allocated() == 64);
        check(r.snapshot().residents == 1);
        bank.release();
        check(gpu.frees == 2 && r.snapshot().residents == 0);
        bank.bound(16);
        check(!bank.retain(key, bytes, cancel) && r.snapshot().residents == 0);
        bank.bound(64);
        check(bank.retain(key, bytes, cancel) && bank.reserved() == 64);
        bank.release();
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        gpu.free_bytes = 16;
        check(
            !bank.retain(key, bytes, cancel) && gpu.pointers.empty()
            && r.snapshot().residents == 0
        );
        gpu.free_bytes = 1024;
        gpu.oom = true;
        check(
            !bank.retain(key, bytes, cancel) && bank.allocated() == 0
            && !bank.quarantined()
        );
        gpu.oom = false;
        check(bank.retain(key, bytes, cancel));
        gpu.free_bytes = 0;
        check(bank.retain(key, bytes, cancel) && gpu.copies == 1);
        bank.release();
        check(r.snapshot().residents == 0);
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 128, gpu);
        auto text = key;
        text.source.inode = 7;
        auto text_tail = text;
        text_tail.first = 8;
        auto denoiser = text;
        // Same shard, different tensor: identity includes the payload interval.
        denoiser.source.payload_begin = 64;
        denoiser.source.payload_end = 96;
        const std::array sources{text.source, text.source};
        auto warm = bank.retain(denoiser, bytes, cancel);
        for (int request = 0; request < 2; ++request) {
            for (int image = 0; image < 4; ++image) {
                check(bank.retain(text, bytes, cancel));
                check(bank.retain(text_tail, bytes, cancel));
                auto missing = text;
                missing.source.inode = 99;
                check(bank.release_sources(std::span(&missing.source, 1)) == 0);
                // Cancellation must not prevent stage cleanup after work is joined.
                cancel = true;
                check(bank.release_sources(sources) == 64);
                cancel = false;
                check(!bank.find(text) && !bank.find(text_tail));
                check(bank.find(denoiser) == warm && bank.allocated() == 32);
                check(bank.reserved() == 128 && bank.capacity() == 128);
                check(bank.release_sources(sources) == 0);
            }
        }
        // A full warmed bank may stream text; retirement still preserves it.
        auto other = denoiser;
        for (std::size_t first : {8, 16, 24}) {
            other.first = first;
            check(bank.retain(other, bytes, cancel));
        }
        check(!bank.retain(text, bytes, cancel));
        check(bank.release_sources(sources) == 0 && bank.allocated() == 128);
        check(bank.find(denoiser) == warm);
        bank.release();
        check(gpu.pointers.empty() && r.snapshot().residents == 0);
    }
    for (const auto* failure : {"sync", "free"}) {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        check(bank.retain(key, bytes, cancel));
        gpu.fail = failure;
        fails([&] { bank.release_sources(std::span(&key.source, 1)); });
        check(bank.quarantined() && bank.allocated() == 32 && bank.reserved() == 64);
        fails([&] { bank.find(key); });
        check(!gpu.pointers.empty() && r.snapshot().residents == 1);
    }
    for (const auto* fail : {"allocate", "copy"}) {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        gpu.fail = fail;
        fails([&] { bank.retain(key, bytes, cancel); });
        check(!bank.quarantined() && bank.allocated() == 0 && gpu.pointers.empty());
        gpu.fail.clear();
        bank.release();
        check(r.snapshot().residents == 0);
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        gpu.cancellation = &cancel;
        fails([&] { bank.retain(key, bytes, cancel); });
        check(bank.allocated() == 0 && gpu.pointers.empty());
        cancel = false;
        bank.release();
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        gpu.fail = "copy";
        gpu.fail_free = true;
        bool combined = false;
        try {
            bank.retain(key, bytes, cancel);
        } catch (const OperationCleanupError& e) {
            combined = true;
            check(
                exception_message(e.operation) == "copy"
                && exception_message(e.cleanup) == "free"
            );
        }
        check(
            combined && bank.quarantined() && bank.allocated() == 32
            && r.snapshot().used == Footprint({host, 64})
        );
    }
    for (const auto* failure : {"free", "sync"}) {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        bank.retain(key, bytes, cancel);
        gpu.fail = failure;
        fails([&] { bank.release(); });
        check(
            bank.quarantined() && r.snapshot().used == Footprint({host, 64})
            && bank.allocated() == 32
        );
        gpu.fail.clear();
        fails([&] { bank.retain(key, bytes, cancel); });
        fails([&] { bank.release(); });
    }
    {
        Resources r({host, 1024});
        Fake gpu(r);
        DeviceWeights bank(r, 0, Workload::image, 64, gpu);
        auto invalid = key;
        invalid.count = 1;
        fails([&] { bank.retain(invalid, bytes, cancel); });
        check(r.snapshot().residents == 0);
    }
    std::cout
        << "Mock immutable device-weight ownership passed (no CUDA or inference)\n";
}
