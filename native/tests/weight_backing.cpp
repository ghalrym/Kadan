#include "kadan/weight_backing.hpp"
#include <array>
#include <cstdlib>
#include <new>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

// Target only payload/staging array allocation, after ledger admission.
static bool fail_array_allocation = false;
void* operator new[](std::size_t size) {
    if (fail_array_allocation) { fail_array_allocation = false; throw std::bad_alloc(); }
    if (auto p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using kadan::serving::WeightBacking;
using kadan::Resources;
using kadan::Footprint;
void check(bool value) { if (!value) throw std::runtime_error("check_failed"); }
template<class F> void fails(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    check(rejected);
}
struct Fixture {
    std::filesystem::path root;
    std::shared_ptr<kadan::checkpoint::MemoryBudget> metadata = std::make_shared<kadan::checkpoint::MemoryBudget>(65536);
    Fixture() {
        char pattern[] = "/tmp/kadan-weight-XXXXXX";
        auto path = ::mkdtemp(pattern); if (!path) throw std::runtime_error("tempdir"); root = path;
        std::ofstream out(root / "model.safetensors", std::ios::binary);
        std::string header = R"({"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[8,16]},"c":{"dtype":"U8","shape":[8],"data_offsets":[16,24]}})";
        for (unsigned i = 0; i < 8; ++i) out.put(char(std::uint64_t(header.size()) >> (8 * i)));
        out << header;
        for (unsigned i = 0; i < 24; ++i) out.put(char(i));
    }
    ~Fixture() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    auto shard() { return std::make_shared<kadan::checkpoint::Shard>(root.c_str(), "model.safetensors", metadata); }
    void add(WeightBacking& store) {
        auto source = shard();
        store.add("qwen/rev/a", kadan::Workload::llm, source, "a");
        store.add("qwen/rev/b", kadan::Workload::llm, source, "b");
        store.add("image/rev/c", kadan::Workload::image, source, "c");
    }
};
void verify_copy(WeightBacking& store, std::shared_ptr<Resources> ledger,
                 const std::string& key, unsigned first, bool retained) {
    auto before = ledger->snapshot().used;
    auto ticket = store.begin_transfer(key, {0, 8});
    check(ledger->snapshot().used == Footprint({before[0] + (retained ? 0 : 3), 8}));
    std::array<std::uint8_t, 8> destination{};
    std::size_t calls = 0;
    store.copy(ticket, [&](std::size_t offset, auto data) {
        ++calls; check(data.size() <= 3);
        std::copy(data.begin(), data.end(), destination.begin() + offset);
        fails([&] { store.evict(key); });
        fails([&] { store.cleaned(ticket, true); });
    });
    check(calls == 3);
    for (unsigned i = 0; i < 8; ++i) check(destination[i] == first + i);
    fails([&] { store.copy(ticket, [](auto, auto) {}); });
    store.cleaned(ticket, false); check(ledger->snapshot().used[1] == 8);
    store.cleaned(ticket, true); check(ledger->snapshot().used == before);
    fails([&] { store.cleaned(ticket, true); });
}
void retention_and_cold() {
    Fixture fixture; auto ledger = std::make_shared<Resources>(Footprint{24, 8});
    {
        WeightBacking store(ledger, 16, 24, 3); fixture.add(store);
        check(store.stats().cold == 24 && store.stats().ram == 0);
        check(store.retain("qwen/rev/a") && store.retain("qwen/rev/b"));
        verify_copy(store, ledger, "qwen/rev/a", 0, true); // Touch a, so b is oldest.
        check(store.retain("image/rev/c"));
        check(store.stats().ram == 16);
        verify_copy(store, ledger, "qwen/rev/b", 8, false);
        verify_copy(store, ledger, "image/rev/c", 16, true);
        store.evict("qwen/rev/a"); check(store.stats().ram == 8);
        store.forget("image/rev/c"); check(store.stats().cold == 16 && store.stats().ram == 0);
    }
    check(ledger->snapshot().residents == 0);
    // Reconstruct from existing checkpoint; no cache files or writes to recover.
    WeightBacking recovered(ledger, 0, 24, 3); fixture.add(recovered);
    check(!recovered.retain("qwen/rev/a")); verify_copy(recovered, ledger, "qwen/rev/a", 0, false);
    check(std::distance(std::filesystem::directory_iterator(fixture.root), std::filesystem::directory_iterator{}) == 1);
}
void failures_and_cleanup() {
    Fixture fixture; auto ledger = std::make_shared<Resources>(Footprint{24, 8});
    WeightBacking store(ledger, 16, 24, 3); fixture.add(store);
    std::atomic_bool cancelled{true};
    fails([&] { store.retain("qwen/rev/a", &cancelled); }); check(ledger->snapshot().residents == 0);
    for (bool retained : {false, true}) {
        if (retained) check(store.retain("qwen/rev/a"));
        auto before = ledger->snapshot().used;
        auto ticket = store.begin_transfer("qwen/rev/a", {0, 8});
        fails([&] { store.copy(ticket, [](auto, auto) {}, &cancelled); });
        check(store.stats().transfer == ticket);
        store.cleaned(ticket, false); check(ledger->snapshot().used[1] == 8);
        fails([&] { store.begin_transfer("qwen/rev/b", {0, 8}); });
        store.cleaned(ticket, true); check(ledger->snapshot().used == before);
        ticket = store.begin_transfer("qwen/rev/a", {0, 8}); cancelled = false;
        fails([&] { store.copy(ticket, [&](auto, auto) { cancelled = true; }, &cancelled); });
        check(store.stats().transfer == ticket); store.cleaned(ticket, true);
        ticket = store.begin_transfer("qwen/rev/a", {0, 8});
        fails([&] { store.copy(ticket, [](auto, auto) { throw std::runtime_error("sink_failed"); }); });
        store.cleaned(ticket, true); verify_copy(store, ledger, "qwen/rev/a", 0, retained);
        cancelled = true;
    }
    // Mutated/truncated backing fails closed even if there is a RAM copy.
    std::filesystem::resize_file(fixture.root / "model.safetensors", 9);
    fails([&] { store.begin_transfer("qwen/rev/a", {0, 8}); });
    fails([&] { store.retain("qwen/rev/b"); });
    store.evict("qwen/rev/a"); check(ledger->snapshot().residents == 0);
}
void bounds_and_pressure() {
    Fixture fixture; auto ledger = std::make_shared<Resources>(Footprint{16, 8});
    WeightBacking store(ledger, 16, 24, 3); fixture.add(store);
    fails([&] { store.add("other", kadan::Workload::llm, fixture.shard(), "a"); });
    check(store.stats().cold == 24);
    fails([&] { store.begin_transfer("qwen/rev/a", {0, 7}); });
    fails([&] { store.begin_transfer("qwen/rev/a", {8, 8}); });
    check(store.retain("qwen/rev/a") && store.retain("qwen/rev/b"));
    // Cold source needs staging while both retained sources still occupy RAM.
    fails([&] { store.begin_transfer("image/rev/c", {0, 8}); });
    check(store.stats().transfer == 0 && ledger->snapshot().used == Footprint({16, 0}));
    store.evict("qwen/rev/b");
    auto state = ledger->reserve(kadan::Workload::llm, {8, 0});
    check(store.retain("image/rev/c")); // a can be evicted; external state cannot.
    check(ledger->snapshot().used == Footprint({16, 0}));
    store.evict("image/rev/c"); ledger->resize_loading(state, {16, 0});
    check(!store.retain("qwen/rev/a")); check(ledger->snapshot().used == Footprint({16, 0}));
    ledger->released(state);
    verify_copy(store, ledger, "qwen/rev/a", 0, false);
    WeightBacking small(ledger, 0, 8, 3, 1);
    small.add("a", kadan::Workload::llm, fixture.shard(), "a");
    fails([&] { small.add("b", kadan::Workload::llm, fixture.shard(), "b"); });
    small.forget("a"); small.add("b", kadan::Workload::llm, fixture.shard(), "b");
}
void abandonment_and_read_failure() {
    Fixture fixture; auto ledger = std::make_shared<Resources>(Footprint{16, 8});
    kadan::Handle ticket;
    {
        WeightBacking store(ledger, 8, 24, 3); fixture.add(store);
        check(store.retain("qwen/rev/a")); ticket = store.begin_transfer("qwen/rev/a", {0, 8});
    }
    check(ledger->snapshot().used == Footprint({0, 8})); // Never free unknown external destination.
    ledger->released(ticket); // Test-owned fake destination is physically gone.
    WeightBacking store(ledger, 0, 24, 3); fixture.add(store);
    ticket = store.begin_transfer("qwen/rev/b", {0, 8});
    fails([&] { store.copy(ticket, [&](auto, auto) { std::filesystem::resize_file(fixture.root / "model.safetensors", 9); }); });
    check(ledger->snapshot().used == Footprint({3, 8})); store.cleaned(ticket, true);
    check(ledger->snapshot().residents == 0);
}
void cleanup_is_irreversible() {
    Fixture fixture; auto ledger = std::make_shared<Resources>(Footprint{16, 8});
    WeightBacking store(ledger, 0, 24, 3); fixture.add(store);
    for (bool copied : {false, true}) {
        auto ticket = store.begin_transfer("qwen/rev/a", {0, 8});
        if (copied) store.copy(ticket, [](auto, auto) {});
        store.cleaned(ticket, false);
        check(ledger->snapshot().used == Footprint({3, 8}));
        fails([&] { store.copy(ticket, [](auto, auto) {}); });
        fails([&] { store.commit(ticket); });
        store.cleaned(ticket, false);
        check(ledger->snapshot().used == Footprint({3, 8}));
        store.cleaned(ticket, true);
        check(ledger->snapshot().residents == 0);
    }
}
void admission_allocation_and_commit() {
    Fixture fixture; auto ledger = std::make_shared<Resources>(Footprint{16, 8});
    WeightBacking store(ledger, 8, 24, 3); fixture.add(store);
    fail_array_allocation = true;
    fails([&] { store.retain("qwen/rev/a"); });
    check(!fail_array_allocation && ledger->snapshot().residents == 0 && store.stats().ram == 0);
    fail_array_allocation = true;
    fails([&] { store.begin_transfer("qwen/rev/a", {0, 8}); });
    check(!fail_array_allocation && ledger->snapshot().residents == 0 && store.stats().transfer == 0);
    auto ticket = store.begin_transfer("qwen/rev/a", {0, 8});
    fails([&] { store.commit(ticket); });
    store.copy(ticket, [](auto, auto) {});
    check(store.commit(ticket) == ticket);
    check(store.stats().staging == 0 && store.stats().transfer == 0);
    check(ledger->snapshot().used == Footprint({0, 8}));
    ledger->pin(ticket); ledger->unpin(ticket);
    check(store.retain("qwen/rev/b")); // Independent host backing can survive GPU cleanup.
    ledger->begin_eviction(ticket); ledger->released(ticket);
    check(ledger->snapshot().used == Footprint({8, 0}));
    fails([&] { store.commit(ticket); });
    auto later = store.begin_transfer("qwen/rev/b", {0, 8});
    fails([&] { store.copy(ticket, [](auto, auto) {}); });
    fails([&] { store.cleaned(ticket, true); });
    store.cleaned(later, true);
    // Host destination peak includes retained source bytes, too.
    later = store.begin_transfer("qwen/rev/b", {8, 0});
    check(ledger->snapshot().used == Footprint({16, 0}));
    store.cleaned(later, true);
}
void read_through_and_slot_pressure() {
    Fixture fixture; auto source=fixture.shard();auto ledger=std::make_shared<Resources>(Footprint{32,8});
    WeightBacking store(ledger,32,24,3);fixture.add(store);check(store.retain("qwen/rev/a"));
    std::vector<kadan::Handle> external;
    for(std::size_t i=1;i<Resources::max_residents;++i)external.push_back(ledger->reserve(kadan::Workload::video,{0,0}));
    check(store.retain("qwen/rev/b"));check(ledger->snapshot().residents==Resources::max_residents&&store.stats().ram==8);
    check(store.room_for_reservations(1)&&store.stats().ram==0);
    check(!store.room_for_reservations(2));
    for(auto id:external)ledger->released(id);
    std::array<std::uint8_t,4> out{};
    WeightBacking cold(ledger,0,0,3);cold.read_through("tensor",kadan::Workload::llm,source,"b",2,out);
    check(out==std::array<std::uint8_t,4>{10,11,12,13}&&cold.stats().entries==0);
    WeightBacking cache(ledger,8,8,3);cache.read_through("tensor",kadan::Workload::llm,source,"a",2,out);
    check(out==std::array<std::uint8_t,4>{2,3,4,5}&&cache.stats().ram==8);
    cache.read_through("tensor",kadan::Workload::llm,source,"a",3,out);
    check(out==std::array<std::uint8_t,4>{3,4,5,6});
    fails([&]{cache.read_through("tensor",kadan::Workload::llm,fixture.shard(),"a",0,out);});
}
void aggregate_full_cache_and_streaming() {
    Fixture fixture; auto source=fixture.shard();
    auto ledger=std::make_shared<Resources>(Footprint{12000,8});
    {
        WeightBacking store(ledger,12000,12000,3,1500,true);
        std::array<std::uint8_t,1> out{};
        for(unsigned i=0;i<1500;++i) store.read_through(std::to_string(i),kadan::Workload::llm,source,"a",0,out);
        auto filled=store.stats();
        check(filled.entries==1500 && filled.ram==12000 && filled.source_bytes==12000 && filled.misses==1500);
        check(ledger->snapshot().residents==1 && ledger->snapshot().used[0]==12000);
        check(store.room_for_reservations(3));
        // Repeated GPU reload reads only application buffers, independent of OS cache.
        for(unsigned i=0;i<1500;++i) store.read_through(std::to_string(i),kadan::Workload::llm,source,"a",3,out);
        auto warm=store.stats();check(warm.source_bytes==filled.source_bytes && warm.hits==1500 && warm.hit_bytes==1500 && warm.evictions==0);
        check(out[0]==3);
    }
    check(ledger->snapshot().residents==0);
    {
        WeightBacking partial(ledger,8,24,3,3,true);
        std::array<std::uint8_t,1> out{};
        for(unsigned i=0;i<20;++i) {
            partial.read_through("a",kadan::Workload::llm,source,"a",i%8,out);
            partial.read_through("b",kadan::Workload::llm,source,"b",i%8,out);
        }
        auto stats=partial.stats();
        check(stats.ram==8 && stats.source_bytes==28 && stats.evictions==0 && stats.hits==19 && stats.misses==21);
    }
    check(ledger->snapshot().residents==0);
    {
        WeightBacking failed(ledger,8,24,3,3,true);
        failed.add("a",kadan::Workload::llm,source,"a");
        fail_array_allocation=true;fails([&]{failed.retain("a");});
        check(failed.stats().ram==0 && ledger->snapshot().used[0]==8);
        std::atomic_bool cancel{true};fails([&]{failed.retain("a",&cancel);});
        check(failed.stats().ram==0 && failed.stats().source_bytes==0);
    }
    check(ledger->snapshot().residents==0);
}
int main() {
    aggregate_full_cache_and_streaming(); retention_and_cold(); failures_and_cleanup(); bounds_and_pressure(); abandonment_and_read_failure(); admission_allocation_and_commit(); cleanup_is_irreversible(); read_through_and_slot_pressure();
    std::cout << "weight backing tests passed\n";
}
