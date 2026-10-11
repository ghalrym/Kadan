#include "kadan/checkpoint.hpp"
#include "kadan/weight_inventory.hpp"
#include "kadan/checkpoint_cache.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace kadan::checkpoint;
namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("check_failed"); }
template<class F> void fails(F fn, std::string_view expected) {
    bool failed = false;
    try { fn(); } catch (const std::exception& error) {
        if (error.what() != expected) throw std::runtime_error(std::string("expected ") + std::string(expected) + ", got " + error.what());
        failed = true;
    }
    check(failed);
}
class Fixture {
public:
    Fixture() {
        std::array<char, 40> pattern{};
        std::string name = "/tmp/kadan-checkpoint-XXXXXX";
        std::copy(name.begin(), name.end(), pattern.begin());
        const auto path = ::mkdtemp(pattern.data());
        if (!path) throw std::runtime_error("tempdir_failed"); root = path;
    }
    ~Fixture() { std::error_code error; std::filesystem::remove_all(root, error); }
    void write(std::string header, std::vector<std::uint8_t> payload = {}, std::string_view name = "model.safetensors") {
        std::ofstream out(root / name, std::ios::binary | std::ios::trunc);
        const auto n = static_cast<std::uint64_t>(header.size());
        for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>(n >> (8 * i)));
        out << header;
        out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        if (!out) throw std::runtime_error("fixture_write_failed");
    }
    std::filesystem::path root;
};
std::string tensor(std::string_view name, std::string_view dtype, std::string_view shape,
                   unsigned begin, unsigned end) {
    return "\"" + std::string(name) + "\":{\"dtype\":\"" + std::string(dtype) +
        "\",\"shape\":" + std::string(shape) + ",\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
}
std::shared_ptr<MemoryBudget> budget(std::size_t bytes = 1024 * 1024) { return std::make_shared<MemoryBudget>(bytes); }
void nvfp4_fixture(Fixture& f) {
    // Same dtype/shape convention as the inspected Qwen expert/lm_head tensors.
    const auto header = "{" + tensor("p.weight_scale_2", "F32", "[]", 0, 4) + "," +
        tensor("p.weight_scale", "F8_E4M3", "[2,1]", 4, 6) + "," +
        tensor("p.weight", "U8", "[2,8]", 6, 22) + "," +
        tensor("unused", "BF16", "[1]", 22, 24) + ",\"__metadata__\":{\"format\":\"pt\"}}   ";
    std::vector<std::uint8_t> payload{0, 0, 0, 64, 0x38, 0x40}; // global=2, blocks=1,2
    payload.insert(payload.end(), 16, 0x22); // all FP4 values +1
    payload.push_back(0); payload.push_back(0);
    f.write(header, payload);
}
void compressed_nvfp4_rows() {
    Fixture f;
    auto header="{"+tensor("p.weight_global_scale","F32","[1]",0,4)+","+
      tensor("p.weight_scale","F8_E4M3","[2,1]",4,6)+","+
      tensor("p.weight_packed","U8","[2,8]",6,22)+"}";
    std::vector<std::uint8_t> bytes{0,0,0,64,0x38,0x40}; // inverse global=0.5
    bytes.insert(bytes.end(),16,0x22);
    f.write(header,bytes);auto q=budget();
    {Shard shard(f.root.c_str(),"model.safetensors",q);
     auto row=shard.load_compressed_nvfp4_rows("p",1,1,13);
     check(kadan::quantization::decode_rows(row.view(),0,1,64)==std::vector<float>(16,1));
     fails([&]{shard.load_compressed_nvfp4_rows("p",0,1,12);},"payload_budget");}
    check(q->used()==0);
    for(auto raw: {std::uint32_t(0),std::uint32_t(0xbf800000),std::uint32_t(0x7f800000)}) {
      for(unsigned i=0;i<4;++i)bytes[i]=std::uint8_t(raw>>(i*8));f.write(header,bytes);
      {Shard shard(f.root.c_str(),"model.safetensors",q);
       fails([&]{shard.load_compressed_nvfp4_rows("p",0,1,13);},"compressed_nvfp4_global_scale");}
      check(q->used()==0);
    }
}
void loading_and_lifetime() {
    Fixture f; nvfp4_fixture(f); auto quota = budget();
    {
        std::unique_ptr<Projection> retained;
        {
            Shard shard(f.root.c_str(), "model.safetensors", quota);
            check(shard.tensor_count() == 4);
            const auto metadata_bytes = quota->used();
            fails([&] { shard.load_modelopt_rows("p", 1, 1, 12); }, "payload_budget");
            check(quota->used() == metadata_bytes);
            auto projection = shard.load_modelopt_rows("p", 1, 1, 13);
            check(projection.view().rows == 1 && projection.view().columns == 16);
            check(kadan::quantization::decode_rows(projection.view(), 0, 1, 64) == std::vector<float>(16, 4));
            check(quota->used() == metadata_bytes + 13);
            retained = std::make_unique<Projection>(std::move(projection));
            fails([&] { shard.load_modelopt_rows("p", 2, 1, 13); }, "projection_shape");
            fails([&] { shard.load_modelopt_rows("p", 0, 0, 13); }, "projection_range_or_name");
            fails([&] { shard.load_modelopt_rows("absent", 0, 1, 13); }, "missing_tensor");
        }
        check(quota->used() == 13);
        check(retained->view().weights.size() == 8);
        check(kadan::quantization::decode_rows(retained->view(), 0, 1, 64)[0] == 4);
    }
    check(quota->used() == 0);
}
void fp8_rows() {
    Fixture f;
    f.write("{" + tensor("p.weight_scale", "F32", "[2,1]", 0, 8) + "," +
        tensor("p.weight", "F8_E4M3", "[2,2]", 8, 12) + "}",
        {0,0,0,64, 0,0,0,63, 0x38,0x40,0xc0,0x7e});
    auto q = budget();
    {
        Shard shard(f.root.c_str(), "model.safetensors", q);
        auto p = shard.load_modelopt_rows("p", 1, 1, 6);
        check(kadan::quantization::decode_rows(p.view(), 0, 1, 8) == std::vector<float>({-1,224}));
        const auto before = q->used();
        fails([&] { shard.load_modelopt_rows("p", 0, 2, 11); }, "payload_budget");
        check(q->used() == before);
    }
    check(q->used() == 0);
}
void rejection_cases() {
    Fixture f; auto q = budget();
    auto reject = [&](std::string header, std::vector<std::uint8_t> data, std::string_view error) {
        f.write(std::move(header), std::move(data));
        fails([&] { Shard s(f.root.c_str(), "model.safetensors", q); }, error);
        check(q->used() == 0);
    };
    reject("{" + tensor("x", "U8", "[1]", 0, 1) + "," + tensor("x", "U8", "[1]", 1, 2) + "}", {0,0}, "duplicate_tensor");
    reject("{" + tensor("x", "U8", "[1]", 1, 2) + "}", {0,0}, "tensor_gap_or_overlap");
    reject("{" + tensor("x", "U8", "[2]", 0, 2) + "," + tensor("y", "U8", "[1]", 1, 2) + "}", {0,0}, "tensor_gap_or_overlap");
    reject("{" + tensor("x", "U8", "[1]", 0, 1) + "}", {0,0}, "payload_coverage");
    reject("{" + tensor("x", "U8", "[2]", 0, 2) + "}", {0}, "payload_coverage");
    reject("{" + tensor("x", "U8", "[2]", 0, 1) + "}", {0}, "tensor_byte_size");
    reject("{" + tensor("x", "U8", "[18446744073709551615,2]", 0, 0) + "}", {}, "size_overflow");
    reject("{" + tensor("x", "U8", "[1,1,1,1,1,1,1,1,1]", 0, 1) + "}", {0}, "rank_limit");
    reject("{" + tensor("x", "U8", "[-1]", 0, 0) + "}", {}, "json_integer");
    reject("{" + tensor("x", "U8", "[01]", 0, 1) + "}", {0}, "json_integer");
    reject("{" + tensor("x", "U8", "[18446744073709551616]", 0, 0) + "}", {}, "json_integer");
    reject("{" + tensor("x", "I32", "[1]", 0, 4) + "}", {0,0,0,0}, "unsupported_dtype");
    reject("{\"x\":{\"dtype\":\"U8\",\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}", {0}, "duplicate_tensor_field");
    reject("{\"x\":{\"dtype\":\"U8\"}}", {}, "missing_tensor_field");
    reject("{\"x\":{\"bad\":[]}}", {}, "unknown_tensor_field");
    reject("{\"__metadata__\":{\"x\":{}}}", {}, "json_syntax");
    reject("{\"__metadata__\":{\"x\":\"a\",\"x\":\"b\"}}", {}, "metadata_key_limit_or_duplicate");
    reject("{\"__metadata__\":{},\"__metadata__\":{}}", {}, "duplicate_metadata");
    reject("{\"__metadata__\":{\"x\":\"\\u1234\"}}", {}, "non_ascii_json_string");
    reject("{}false", {}, "trailing_json");
    reject(std::string("{}\0", 3), {}, "trailing_json");
    reject("{\"" + std::string(513,'x') + "\":{}}", {}, "json_string_limit");
    reject(" " + std::string("{}"), {}, "header_start");
    f.write("{}");
    fails([&] { Shard s(f.root.c_str(), "model.safetensors", q, {1,10}); }, "header_size");
    check(q->used() == 0);
    f.write("{" + tensor("x", "U8", "[0]", 0, 0) + "," + tensor("y", "F32", "[]", 0, 4) + "}", {0,0,0,0});
    fails([&] { Shard s(f.root.c_str(), "model.safetensors", q, {1024,1}); }, "tensor_limit_or_name");
    check(q->used() == 0);
    { Shard valid_empty_tensor(f.root.c_str(), "model.safetensors", q); check(valid_empty_tensor.tensor_count() == 2); }
    auto tiny = budget(1);
    bool exhausted = false;
    try { Shard s(f.root.c_str(), "model.safetensors", tiny); } catch (const std::bad_alloc&) { exhausted = true; }
    check(exhausted && tiny->used() == 0);
}
void binding_and_budget_failures() {
    Fixture f; nvfp4_fixture(f); auto q = budget(8192);
    {
        Shard shard(f.root.c_str(), "model.safetensors", q);
        const auto metadata = q->used();
        const auto remaining = 8192 - metadata - 1;
        void* occupied = q->allocate(remaining);
        bool exhausted = false;
        try { shard.load_modelopt_rows("p", 0, 1, 13); }
        catch (const std::bad_alloc&) { exhausted = true; }
        check(exhausted && q->used() == metadata + remaining);
        q->deallocate(occupied, remaining);
        check(q->used() == metadata);
        { auto p = shard.load_modelopt_rows("p", 0, 1, 13); check(q->used() == metadata + 13); }
        check(q->used() == metadata);
    }
    check(q->used() == 0);
    auto fp8_reject = [&](std::vector<std::uint8_t> data, std::string_view error) {
        f.write("{" + tensor("p.weight_scale", "F32", "[]", 0, 4) + "," +
                tensor("p.weight", "F8_E4M3", "[1,1]", 4, 5) + "}", data);
        Shard s(f.root.c_str(), "model.safetensors", q);
        const auto before = q->used();
        fails([&] { s.load_modelopt_rows("p", 0, 1, 5); }, error);
        check(q->used() == before);
    };
    fp8_reject({0,0,128,63,0x7f}, "nonfinite_weight");
    fp8_reject({0,0,128,127,0x38}, "invalid_multiplier");
    fp8_reject({0,0,0,0,0x38}, "invalid_multiplier");
    auto layout_reject = [&](std::string_view dtype, std::string_view global_shape, std::string_view error) {
        f.write("{" + tensor("p.weight_scale_2", "F32", global_shape, 0, 4) + "," +
                tensor("p.weight_scale", dtype, "[1,1]", 4, 5) + "," +
                tensor("p.weight", "U8", "[1,8]", 5, 13) + "}",
                {0,0,128,63, 0x38, 0x22,0x22,0x22,0x22,0x22,0x22,0x22,0x22});
        Shard s(f.root.c_str(), "model.safetensors", q);
        fails([&] { s.load_modelopt_rows("p", 0, 1, 13); }, error);
    };
    layout_reject("U8", "[]", "nvfp4_scale_layout");
    layout_reject("F8_E4M3", "[1,1]", "nvfp4_global_layout");
    check(q->used() == 0);
}
void half_metadata_limits() {
    {
        Fixture signed_fixture;auto admitted=budget();
        signed_fixture.write("{"+tensor("signed", "I8", "[3]", 0, 3)+"}", {128,0,127});
        Shard shard(signed_fixture.root.c_str(), "model.safetensors", admitted);
        check(shard.tensor("signed").dtype==Dtype::i8 && shard.tensor("signed").bytes==3);
        std::array<std::uint8_t,3> raw{};shard.read_tensor("signed",0,raw);
        check(raw==std::array<std::uint8_t,3>{128,0,127});
    }

    Fixture f; auto q=budget();
    const auto header="{\"__metadata__\":{\"config\":\""+std::string(2013, 'a')+"\"},"+tensor("half", "F16", "[1]", 0, 2)+"}";
    f.write(header, {0, 60});
    fails([&]{Shard shard(f.root.c_str(), "model.safetensors", q);}, "json_string_limit");
    { Shard shard(f.root.c_str(), "model.safetensors", q, {8192, 4, 4096});
      check(shard.tensor("half").dtype==Dtype::fp16 && shard.tensor("half").bytes==2); }
    check(q->used()==0);
    f.write("{"+tensor("half", "F16", "[1]", 0, 1)+"}", {0});
    fails([&]{Shard shard(f.root.c_str(), "model.safetensors", q);}, "tensor_byte_size");
    f.write("{"+tensor(std::string(513,'a'), "F16", "[1]", 0, 2)+"}", {0,60});
    fails([&]{Shard shard(f.root.c_str(), "model.safetensors", q, {8192,4,4096});}, "json_string_limit");
    check(q->used()==0);
}
void filesystem_cases() {
    Fixture f; nvfp4_fixture(f); auto q = budget();
    for (const auto name : {"../model.safetensors", "/tmp/model.safetensors", ".", "..", "x/y", ""})
        fails([&] { Shard s(f.root.c_str(), name, q); }, "invalid_shard_name");
    std::filesystem::create_symlink(f.root / "model.safetensors", f.root / "link.safetensors");
    fails([&] { Shard s(f.root.c_str(), "link.safetensors", q); }, "open_failed");
    check(::mkfifo((f.root / "fifo").c_str(), 0600) == 0);
    fails([&] { Shard s(f.root.c_str(), "fifo", q); }, "not_safetensors_file");
    std::filesystem::create_directory(f.root / "directory");
    fails([&] { Shard s(f.root.c_str(), "directory", q); }, "not_safetensors_file");
    {
        Shard shard(f.root.c_str(), "model.safetensors", q);
        std::filesystem::resize_file(f.root / "model.safetensors", 8);
        fails([&] { shard.load_modelopt_rows("p", 0, 1, 13); }, "checkpoint_changed");
    }
    check(q->used() == 0);
    fails([&] { Shard s(f.root.c_str(), "model.safetensors", q); }, "header_size");
    check(q->used() == 0);
}
void inventory_cases(){
    Fixture f;nvfp4_fixture(f);auto q=budget();Shard shard(f.root.c_str(),"model.safetensors",q);
    const auto before=q->used();WeightInventory inventory;inventory.add(shard);
    check(q->used()==before&&inventory.stored_bytes==24&&inventory.floating_f32_bytes==8&&inventory.raw_nonfloating_bytes==18);
    check(shard.tensor_at(0).name=="p.weight"&&shard.tensor_at(3).name=="unused");
    fails([&]{shard.tensor_at(4);},"tensor_index");
    fails([&]{execution_weight_bytes(shard.tensor("p.weight"),WeightRepresentation::f32);},"weight_inventory_float_representation");
    TensorInfo t{"half",Dtype::fp16,{2,3},2,12};
    check(execution_weight_bytes(t,WeightRepresentation::f32)==24);
    t.bytes=13;fails([&]{inventory.add(t);},"weight_inventory_shape");check(inventory.stored_bytes==24);
    t.bytes=12;t.shape[0]=std::numeric_limits<std::uint64_t>::max();
    fails([&]{inventory.add(t);},"weight_inventory_overflow");check(inventory.stored_bytes==24);
    std::filesystem::resize_file(f.root/"model.safetensors",8);
    fails([&]{inventory.add(shard);},"checkpoint_changed");check(inventory.stored_bytes==24);
}
void converted_cache_cases(){
    Fixture f;
    f.write("{"+tensor("half","F16","[4]",0,8)+","+tensor("brain","BF16","[2]",8,12)+","+tensor("single","F32","[1]",12,16)+"}",
        {0,0x3c,0,0x80,1,0,0,0xc0,0x80,0x3f,0,0xc0,0,0,0x60,0x40});
    auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{4096});
    auto cache=std::make_shared<ReadCache>(resources,4096,kadan::Workload::video);std::atomic_bool cancel=false;
    {
        Shard shard(f.root.c_str(),"model.safetensors",budget());shard.cache_reads(cache,&cancel);
        std::array<std::uint8_t,2> raw{};shard.read_tensor("half",0,raw);
        std::array<float,4> values{};shard.read_float_tensor("half",0,values,cancel);
        check(values[0]==1&&std::bit_cast<std::uint32_t>(values[1])==0x80000000&&values[2]==std::ldexp(1.0f,-24)&&values[3]==-2);
        check(cache->stats().entries==2);const auto read_bytes=cache->stats().source_bytes;
        shard.read_float_tensor("half",3,{values.data(),1},cancel);check(values[0]==-2&&cache->stats().source_bytes==read_bytes);
        shard.read_float_tensor("brain",0,{values.data(),2},cancel);check(values[0]==1&&values[1]==-2);
        shard.read_float_tensor("single",0,{values.data(),1},cancel);check(values[0]==3.5f);
        cancel=true;fails([&]{shard.read_float_tensor("half",0,values,cancel);},"checkpoint_cache_cancelled");cancel=false;
        fails([&]{shard.read_float_tensor("half",4,{values.data(),1},cancel);},"checkpoint_float_range");
    }
    // Reopening (including after a GPU park) keeps decoded values, no source reads.
    const auto before=cache->stats().source_bytes;
    {Shard shard(f.root.c_str(),"model.safetensors",budget());shard.cache_reads(cache,&cancel);float value=0;shard.read_float_tensor("half",0,{&value,1},cancel);check(value==1&&cache->stats().source_bytes==before);}
    cache.reset();check(resources->snapshot().used[0]==0);
    cache=std::make_shared<ReadCache>(resources,1,kadan::Workload::video);
    {Shard shard(f.root.c_str(),"model.safetensors",budget());shard.cache_reads(cache,&cancel);float value=0;shard.read_float_tensor("half",3,{&value,1},cancel);check(value==-2&&cache->stats().entries==0);}
}
void cache_cases(){
    Fixture f;nvfp4_fixture(f);auto q=budget();
    auto resources=std::make_shared<kadan::Resources>(kadan::Footprint{4096});
    auto cache=std::make_shared<ReadCache>(resources,4096,kadan::Workload::video);
    std::atomic_bool cancel=false;std::array<std::uint8_t,4> out{};
    {
        Shard shard(f.root.c_str(),"model.safetensors",q);shard.cache_reads(cache,&cancel);
        shard.read_tensor("p.weight",0,out);check(out==std::array<std::uint8_t,4>{0x22,0x22,0x22,0x22});
        check(cache->stats().misses==1&&cache->stats().source_bytes==16&&cache->stats().entries==1);
    }
    {
        Shard shard(f.root.c_str(),"model.safetensors",q);shard.cache_reads(cache,&cancel);
        shard.read_tensor("p.weight",4,out);check(cache->stats().hits==1&&cache->stats().source_bytes==16);
        cancel=true;fails([&]{shard.read_tensor("p.weight",0,out);},"checkpoint_cache_cancelled");cancel=false;
        fails([&]{shard.read_tensor("p.weight",15,out);},"checkpoint_cache_range");
        std::filesystem::resize_file(f.root/"model.safetensors",8);
        fails([&]{shard.read_tensor("p.weight",0,out);},"checkpoint_changed");
    }
    cache.reset();check(resources->snapshot().used[0]==0);
    // A tensor larger than the cache streams into the caller's destination.
    cache=std::make_shared<ReadCache>(resources,1,kadan::Workload::video);
    std::size_t reads=0;ReadCache::Key key{};
    cache->read(key,4,0,out,[&](auto,std::span<std::uint8_t> dst){++reads;std::fill(dst.begin(),dst.end(),7);},&cancel);
    check(reads==1&&cache->stats().allocated==0&&cache->stats().entries==0&&out[0]==7);
    cache.reset();check(resources->snapshot().used[0]==0);
    cache=std::make_shared<ReadCache>(resources,4096,kadan::Workload::video);
    fails([&]{cache->read(key,4,0,out,[&](auto,auto){cancel=true;},&cancel);},"checkpoint_cache_cancelled");
    check(cache->stats().entries==0&&cache->stats().allocated==0);cancel=false;
    fails([&]{cache->read(key,4,0,out,[](auto,auto){throw std::runtime_error("source_failure");},&cancel);},"source_failure");
    check(cache->stats().entries==0&&cache->stats().allocated==0);
    cache.reset();
    // Payload fits exactly, but its map entry does not: stream and retain none.
    cache=std::make_shared<ReadCache>(resources,4,kadan::Workload::video);
    cache->read(key,4,0,out,[](auto,auto dst){std::fill(dst.begin(),dst.end(),3);},&cancel);
    check(cache->stats().allocated==0&&out[0]==3);cache.reset();
    resources=std::make_shared<kadan::Resources>(kadan::Footprint{3*ReadCache::chunk_bytes});
    cache=std::make_shared<ReadCache>(resources,3*ReadCache::chunk_bytes,kadan::Workload::video);reads=0;
    fails([&]{cache->read(key,2*ReadCache::chunk_bytes,0,out,[&](auto offset,auto dst){
        check(dst.size()<=ReadCache::chunk_bytes&&offset==reads*ReadCache::chunk_bytes);++reads;
        std::fill(dst.begin(),dst.end(),1);if(reads==2)cancel=true;
    },&cancel);},"checkpoint_cache_cancelled");
    check(reads==2&&cache->stats().allocated==0&&cache->stats().entries==0);cancel=false;
}
} // namespace
int main() {
    try { compressed_nvfp4_rows(); converted_cache_cases(); cache_cases(); inventory_cases(); half_metadata_limits(); loading_and_lifetime(); fp8_rows(); rejection_cases(); binding_and_budget_failures(); filesystem_cases(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
