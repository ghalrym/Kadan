#include "kadan/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <bit>
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
            check(kadan::quantization::matvec(projection.view(), std::vector<float>(16, 1), 4) == std::vector<float>({64}));
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
} // namespace
int main() {
    try { half_metadata_limits(); loading_and_lifetime(); fp8_rows(); rejection_cases(); binding_and_budget_failures(); filesystem_cases(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
