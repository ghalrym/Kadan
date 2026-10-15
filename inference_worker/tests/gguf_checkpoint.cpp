#include "kadan/gguf_checkpoint.hpp"

#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unistd.h>

using namespace kadan::checkpoint;

namespace {
void check(bool ok) {
    if (!ok)
        throw std::runtime_error("check_failed");
}

template <class F> void fails(F fn, std::string_view expected) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception& e) {
        if (e.what() != expected)
            throw std::runtime_error(std::string(expected) + ": " + e.what());
        rejected = true;
    }
    check(rejected);
}

using Bytes = std::vector<std::uint8_t>;

void append(Bytes& b, std::uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
        b.push_back(value >> (8 * i));
}

void text(Bytes& b, std::string_view value) {
    append(b, value.size(), 8);
    b.insert(b.end(), value.begin(), value.end());
}

struct Tensor {
    std::string name;
    std::uint32_t type;
    std::vector<std::uint64_t> shape;
    Bytes data;
};

class Fixture {
public:
    std::filesystem::path root;

    Fixture() {
        std::array<char, 40> pattern{};
        const std::string s = "/tmp/kadan-gguf-unit-XXXXXX";
        std::copy(s.begin(), s.end(), pattern.begin());
        const auto* p = mkdtemp(pattern.data());
        if (!p)
            throw std::runtime_error("mkdtemp");
        root = p;
    }

    ~Fixture() {
        std::error_code e;
        std::filesystem::remove_all(root, e);
    }

    std::filesystem::path file(unsigned split) const {
        return root
               / (split == 0 ? "muse-00001-of-00002.gguf" : "muse-00002-of-00002.gguf");
    }

    void
    write(unsigned split, const std::vector<Tensor>& tensors, std::uint32_t total = 4) {
        Bytes b;
        append(b, 0x46554747, 4);
        append(b, 3, 4);
        append(b, tensors.size(), 8);
        append(b, 10, 8);
        auto integer = [&](std::string_view key, std::uint32_t value) {
            text(b, key);
            append(b, 4, 4);
            append(b, value, 4);
        };
        integer("split.no", split);
        integer("split.count", 2);
        integer("split.tensors.count", total);
        text(b, "general.architecture");
        append(b, 8, 4);
        text(b, "muse-spark");
        text(b, "muse-spark.expert_weights_norm");
        append(b, 7, 4);
        append(b, 1, 1);
        text(b, "muse-spark.temperature");
        append(b, 6, 4);
        append(b, std::bit_cast<std::uint32_t>(.5f), 4);
        text(b, "muse-spark.attention_swa");
        append(b, 9, 4);
        append(b, 7, 4);
        append(b, 3, 8);
        b.insert(b.end(), {1, 0, 1});
        text(b, "tokenizer.ggml.tokens");
        append(b, 9, 4);
        append(b, 8, 4);
        append(b, 2, 8);
        text(b, "a");
        text(b, "b");
        text(b, "tokenizer.ggml.token_type");
        append(b, 9, 4);
        append(b, 5, 4);
        append(b, 2, 8);
        append(b, 1, 4);
        append(b, 3, 4);
        integer("general.alignment", 32);
        std::uint64_t offset = 0;
        for (const auto& t : tensors) {
            text(b, t.name);
            append(b, t.shape.size(), 4);
            for (auto d : t.shape)
                append(b, d, 8);
            append(b, t.type, 4);
            append(b, offset, 8);
            offset = (offset + t.data.size() + 31) / 32 * 32;
        }
        if (!tensors.empty())
            while (b.size() % 32)
                b.push_back(0);
        const auto begin = b.size();
        for (const auto& t : tensors) {
            b.insert(b.end(), t.data.begin(), t.data.end());
            while ((b.size() - begin) % 32)
                b.push_back(0);
        }
        std::ofstream out(file(split), std::ios::binary);
        out.write(reinterpret_cast<const char*>(b.data()), b.size());
    }
};

Bytes f32(std::initializer_list<float> values) {
    Bytes b;
    for (auto v : values)
        append(b, std::bit_cast<std::uint32_t>(v), 4);
    return b;
}

void reader_rows() {
    Fixture f;
    Bytes q3, q4, q8;
    // Literal GGML blocks, independent of the production dequantizer. Each
    // expert has two rows with a different positive FP16 multiplier.
    for (auto d : {0x3c00, 0x4000, 0x4200, 0x4400}) {
        Bytes b3(110, 0);
        b3[108] = d & 255;
        b3[109] = d >> 8;
        // Q3_K: zero six-bit subscales represent -32; zero hmask and
        // two-bit codes represent -4, so all weights are 128*d.
        q3.insert(q3.end(), b3.begin(), b3.end());
        Bytes b4(144, 0);
        b4[0] = d & 255;
        b4[1] = d >> 8;
        std::fill(b4.begin() + 4, b4.begin() + 16, 1);
        std::fill(b4.begin() + 16, b4.end(), 0x21);
        q4.insert(q4.end(), b4.begin(), b4.end());
        for (unsigned block = 0; block < 8; ++block) {
            append(q8, d, 2);
            for (int value = -16; value < 16; ++value)
                q8.push_back(std::uint8_t(value));
        }
    }
    f.write(0, {});
    f.write(
        1,
        {{"vector", 0, {4}, f32({1, -2, 3, -4})},
         {"q3", 11, {256, 2, 2}, q3},
         {"q4", 12, {256, 2, 2}, q4},
         {"q8", 8, {256, 2, 2}, q8}}
    );
    const std::atomic_bool cancel{false};
    GgufMetadata metadata(f.file(0), cancel);
    check(
        metadata.tensor_count() == 0
        && metadata.string("general.architecture") == "muse-spark"
    );
    check(
        metadata.integer("split.count") == 2
        && metadata.number("muse-spark.temperature") == .5
    );
    check(metadata.boolean("muse-spark.expert_weights_norm"));
    check(
        metadata.booleans("muse-spark.attention_swa")
        == std::vector<bool>({true, false, true})
    );
    check(
        metadata.strings("tokenizer.ggml.tokens")
        == std::vector<std::string>({"a", "b"})
    );
    check(
        metadata.integers("tokenizer.ggml.token_type")
        == std::vector<std::int64_t>({1, 3})
    );
    fails([&] { metadata.integer("general.architecture"); }, "gguf_metadata_integer");
    fails([&] { metadata.string("missing"); }, "gguf_missing_metadata");
    GgufCheckpoint checkpoint(f.root, cancel);
    check(checkpoint.tensor_count() == 4 && checkpoint.tensor("q3").bytes == 440);
    check(checkpoint.tensor("q3").shape == std::array<std::uint64_t, 3>{256, 2, 2});
    const auto id0 = checkpoint.identity("q3", 0), id1 = checkpoint.identity("q3", 1);
    check(
        id0.payload_end == id1.payload_begin
        && id0.payload_end - id0.payload_begin == 220
    );
    std::array<float, 256> output{};
    checkpoint.read_rows("q3", 1, 1, output, cancel);
    for (auto value : output)
        check(value == 512);
    checkpoint.read_rows("q4", 1, 0, output, cancel);
    for (std::size_t i = 0; i < output.size(); ++i)
        check(output[i] == (i % 64 < 32 ? 3.f : 6.f));
    checkpoint.read_rows("q8", 0, 1, output, cancel);
    for (std::size_t i = 0; i < output.size(); ++i)
        check(output[i] == 2.f * (int(i % 32) - 16));
    std::array<float, 4> vector{};
    checkpoint.read_vector("vector", vector, cancel);
    check(vector == std::array<float, 4>{1, -2, 3, -4});
    fails([&] { checkpoint.read_rows("q3", 2, 0, output, cancel); }, "gguf_row_range");
    fails([&] { checkpoint.read_rows("q3", 0, 2, output, cancel); }, "gguf_row_range");
    fails(
        [&] { checkpoint.read_rows("q3", 0, 0, std::span(output).first(255), cancel); },
        "gguf_row_range"
    );
    const std::atomic_bool cancelled{true};
    fails(
        [&] { checkpoint.read_rows("q3", 0, 0, output, cancelled); }, "gguf_cancelled"
    );
    fails([&] { GgufCheckpoint bad(f.root, cancelled); }, "gguf_cancelled");
    std::ofstream(f.file(1), std::ios::app).put(0);
    fails([&] { checkpoint.check_unchanged(); }, "gguf_checkpoint_changed");
}

void reader_batched_rows() {
    Fixture f;
    Bytes dense, quantized;
    for (unsigned expert = 0; expert < 2; ++expert) {
        for (unsigned row = 0; row < 260; ++row)
            for (unsigned column = 0; column < 1024; ++column)
                append(
                    dense,
                    std::bit_cast<std::uint32_t>(
                        float(expert * 1000 + row + column % 16)
                    ),
                    4
                );
        for (unsigned row = 0; row < 600; ++row) {
            Bytes block(110, 0);
            const auto scale = (expert + row) % 2 ? 0x4000 : 0x3c00;
            block[108] = scale & 255;
            block[109] = scale >> 8;
            for (unsigned i = 0; i < 16; ++i)
                quantized.insert(quantized.end(), block.begin(), block.end());
        }
    }
    f.write(0, {}, 2);
    f.write(
        1,
        {{"dense", 0, {1024, 260, 2}, dense},
         {"quantized", 11, {4096, 600, 2}, quantized}},
        2
    );
    const std::atomic_bool cancel{false};
    GgufCheckpoint checkpoint(f.root, cancel);
    // Both selections cross a 1 MiB packed-read boundary and leave a partial
    // final batch. Values also distinguish the selected expert and first row.
    std::vector<float> dense_output(258 * 1024);
    checkpoint.read_rows("dense", 1, 1, dense_output, cancel);
    for (unsigned row = 0; row < 258; ++row)
        for (unsigned column = 0; column < 1024; ++column)
            check(dense_output[row * 1024 + column] == 1001.f + row + column % 16);
    std::vector<float> quantized_output(598 * 4096);
    checkpoint.read_rows("quantized", 1, 1, quantized_output, cancel);
    for (unsigned row = 0; row < 598; ++row)
        for (unsigned column = 0; column < 4096; ++column)
            check(quantized_output[row * 4096 + column] == (row % 2 ? 256.f : 128.f));
}

void metadata_failures() {
    Fixture f;
    const std::atomic_bool cancel{false};
    f.write(0, {}, 0);
    fails([&] { GgufCheckpoint bad(f.root, cancel); }, "gguf_split_name");
    f.write(1, {{"vector", 0, {4}, f32({1, 2, 3, 4})}}, 1);
    fails([&] { GgufCheckpoint bad(f.root, cancel); }, "gguf_split_metadata");
    f.write(0, {}, 1);
    f.write(1, {{"unsupported", 1, {4}, f32({1, 2, 3, 4})}}, 1);
    fails([&] { GgufCheckpoint bad(f.root, cancel); }, "gguf_tensor_type");
    f.write(1, {{"unaligned", 11, {255, 1}, Bytes(110)}}, 1);
    fails([&] { GgufCheckpoint bad(f.root, cancel); }, "gguf_tensor_block_alignment");
    f.write(1, {{"rank", 0, {1, 1, 1, 1}, f32({1})}}, 1);
    fails([&] { GgufCheckpoint bad(f.root, cancel); }, "gguf_tensor_rank");
    f.write(
        1,
        {{"vector", 0, {4}, f32({1, 2, 3, std::numeric_limits<float>::infinity()})}},
        1
    );
    // Metadata construction succeeds without interpreting any tensor payload.
    {
        GgufCheckpoint checkpoint(f.root, cancel);
        std::array<float, 4> output{};
        fails(
            [&] { checkpoint.read_vector("vector", output, cancel); },
            "gguf_nonfinite_weight"
        );
    }
    // A corrupted tensor count is rejected before upstream allocations.
    {
        std::fstream out(f.file(0), std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(8);
        const char bytes[8] = {-1, -1, -1, -1, -1, -1, -1, 127};
        out.write(bytes, 8);
    }
    fails([&] { GgufMetadata bad(f.file(0), cancel); }, "gguf_header_count");
}
}  // namespace

int main() {
    try {
        reader_rows();
        reader_batched_rows();
        metadata_failures();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
