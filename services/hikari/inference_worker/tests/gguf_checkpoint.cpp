#include "kadan/gguf_checkpoint.hpp"

#include <array>
#include <bit>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <utility>

namespace {
struct MetadataReads {
    std::uint64_t header_end = 0, requested_end = 0;
    std::size_t calls = 0, windows = 0, maximum_request = 0;
    off_t trigger_offset = 0;
    bool crossed_header = false, triggered = false;
    std::atomic_bool* cancel_after_refill = nullptr;
    const char* append_after_refill = nullptr;
    int mutation_error = 0;
};

thread_local MetadataReads* active_metadata_reads = nullptr;
}  // namespace

extern "C" ssize_t original_pread(int, void*, size_t, off_t) asm("__real_pread");
extern "C" ssize_t observed_pread(int, void*, size_t, off_t) asm("__wrap_pread");

extern "C" ssize_t observed_pread(int fd, void* output, size_t count, off_t offset) {
    auto* observed = active_metadata_reads;
    if (!observed)
        return original_pread(fd, output, count, offset);
    ++observed->calls;
    observed->maximum_request = std::max(observed->maximum_request, count);
    if (offset < 0 || std::uint64_t(offset) > observed->header_end
        || count > observed->header_end - std::uint64_t(offset)) {
        observed->crossed_header = true;
    } else {
        observed->requested_end =
            std::max(observed->requested_end, std::uint64_t(offset) + count);
        if (std::uint64_t(offset) % 65536 == 0 && count > 0
            && count
                   == std::min<std::uint64_t>(
                       65536, observed->header_end - std::uint64_t(offset)
                   ))
            ++observed->windows;
    }
    const auto result = original_pread(fd, output, count, offset);
    const int saved_errno = errno;
    if (result > 0 && offset == observed->trigger_offset && count == 65536
        && !observed->triggered) {
        observed->triggered = true;
        if (observed->cancel_after_refill)
            observed->cancel_after_refill->store(true);
        if (observed->append_after_refill) {
            const int changed =
                ::open(observed->append_after_refill, O_WRONLY | O_APPEND | O_CLOEXEC);
            if (changed < 0) {
                observed->mutation_error = errno;
            } else {
                const char byte = 0;
                if (::write(changed, &byte, 1) != 1)
                    observed->mutation_error = errno ? errno : EIO;
                if (::close(changed) != 0 && !observed->mutation_error)
                    observed->mutation_error = errno;
            }
        }
    }
    errno = saved_errno;
    return result;
}

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

    // An unsplit file has no split keys and its own, unnumbered name.
    void write(
        unsigned split,
        const std::vector<Tensor>& tensors,
        std::uint32_t total = 4,
        bool single = false
    ) {
        Bytes b;
        append(b, 0x46554747, 4);
        append(b, 3, 4);
        append(b, tensors.size(), 8);
        append(b, single ? 8 : 11, 8);
        auto integer = [&](std::string_view key, std::uint32_t value) {
            text(b, key);
            append(b, 4, 4);
            append(b, value, 4);
        };
        if (!single) {
            integer("split.no", split);
            integer("split.count", 2);
            integer("split.tensors.count", total);
        }
        text(b, "muse-spark.hash_offsets");
        append(b, 9, 4);
        append(b, 10, 4);
        append(b, 2, 8);
        append(b, 0, 8);
        append(b, 300001275, 8);
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
        std::ofstream out(single ? root / "model.gguf" : file(split), std::ios::binary);
        out.write(reinterpret_cast<const char*>(b.data()), b.size());
    }
};

Bytes f32(std::initializer_list<float> values) {
    Bytes b;
    for (auto v : values)
        append(b, std::bit_cast<std::uint32_t>(v), 4);
    return b;
}

class ObserveMetadataReads {
public:
    explicit ObserveMetadataReads(MetadataReads& reads) {
        check(active_metadata_reads == nullptr);
        active_metadata_reads = &reads;
    }

    ~ObserveMetadataReads() {
        active_metadata_reads = nullptr;
    }

    ObserveMetadataReads(const ObserveMetadataReads&) = delete;
    ObserveMetadataReads& operator=(const ObserveMetadataReads&) = delete;
};

struct LargeMetadata {
    Bytes header;
    std::string large_string;
    std::vector<std::string> strings;
    std::vector<std::int64_t> integers;
};

LargeMetadata large_metadata(bool has_tensor = true) {
    LargeMetadata fixture;
    auto& data = fixture.header;
    append(data, 0x46554747, 4);
    append(data, 3, 4);
    append(data, has_tensor ? 1 : 0, 8);
    append(data, 4, 8);
    text(data, "padding");
    append(data, 8, 4);
    // The following key's eight-byte length straddles the first 64 KiB boundary.
    text(data, std::string(65532 - data.size() - 8, 'p'));
    check(data.size() == 65532);
    text(data, "crossing");
    append(data, 8, 4);
    fixture.large_string.resize(2 * 65536 + 37);
    for (std::size_t i = 0; i < fixture.large_string.size(); ++i)
        fixture.large_string[i] = char('a' + i % 26);
    text(data, fixture.large_string);
    text(data, "numbers");
    append(data, 9, 4);
    append(data, 5, 4);
    append(data, 20003, 8);
    for (std::int32_t i = 0; i < 20003; ++i) {
        const auto value = i % 97 - 48;
        fixture.integers.push_back(value);
        append(data, std::bit_cast<std::uint32_t>(value), 4);
    }
    text(data, "strings");
    append(data, 9, 4);
    append(data, 8, 4);
    fixture.strings = {"left", std::string(65536 + 19, 'z'), "right"};
    append(data, fixture.strings.size(), 8);
    for (const auto& value : fixture.strings)
        text(data, value);
    if (has_tensor) {
        text(data, "described_but_unread");
        append(data, 1, 4);
        append(data, 4, 8);
        append(data, 0, 4);
        append(data, 0, 8);
    }
    check(data.size() % 65536 != 0);
    return fixture;
}

void write_bytes(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    check(bool(output));
}

void buffered_metadata_boundaries() {
    Fixture fixture;
    const auto serialized = large_metadata();
    const auto header_end = serialized.header.size();
    std::atomic_bool cancel = false;
    auto header_only = serialized.header;
    // GGUF seeks to the aligned data offset even with no payload allocation.
    // Keep its padding available while bounding actual reads at header_end.
    while (header_only.size() % 32)
        header_only.push_back(0);
    for (const bool payload_present : {false, true}) {
        auto bytes = header_only;
        if (payload_present)
            bytes.resize(bytes.size() + 128 * 1024, 0xa5);
        write_bytes(fixture.file(0), bytes);
        MetadataReads reads{.header_end = header_end};
        {
            ObserveMetadataReads observe(reads);
            GgufMetadata metadata(fixture.file(0), cancel);
            check(metadata.tensor_count() == 1);
            check(metadata.string("crossing") == serialized.large_string);
            check(metadata.integers("numbers") == serialized.integers);
            check(metadata.strings("strings") == serialized.strings);
        }
        // A real payload follows one variant, but even the final short refill
        // must stop exactly at the preflight header end rather than file EOF.
        check(!reads.crossed_header && reads.requested_end == header_end);
        check(reads.windows == (header_end + 65535) / 65536);
        check(reads.calls > reads.windows);
    }
    write_bytes(fixture.file(0), header_only);
    MetadataReads cancelled{.header_end = header_end};
    cancelled.cancel_after_refill = &cancel;
    {
        ObserveMetadataReads observe(cancelled);
        fails(
            [&] { GgufMetadata metadata(fixture.file(0), cancel); }, "gguf_cancelled"
        );
    }
    check(cancelled.triggered && cancelled.windows == 1 && !cancelled.crossed_header);
    cancel = false;
    const auto path = fixture.file(0).string();
    MetadataReads changed{.header_end = header_end};
    changed.append_after_refill = path.c_str();
    {
        ObserveMetadataReads observe(changed);
        fails([&] { GgufMetadata metadata(path, cancel); }, "gguf_checkpoint_changed");
    }
    check(changed.triggered && !changed.mutation_error && !changed.crossed_header);
    auto truncated = serialized.header;
    truncated.resize(65536 - 1);
    write_bytes(fixture.file(0), truncated);
    fails([&] { GgufMetadata metadata(path, cancel); }, "gguf_read_bounds");
}

void zero_tensor_metadata_buffering() {
    Fixture fixture;
    const auto serialized = large_metadata(false);
    const auto header_end = serialized.header.size();
    const auto path = fixture.file(0).string();
    write_bytes(path, serialized.header);
    std::atomic_bool cancel = false;
    MetadataReads reads{.header_end = header_end};
    {
        ObserveMetadataReads observe(reads);
        GgufMetadata metadata(path, cancel);
        check(metadata.tensor_count() == 0);
        check(metadata.string("crossing") == serialized.large_string);
        check(metadata.integers("numbers") == serialized.integers);
        check(metadata.strings("strings") == serialized.strings);
    }
    // There is no tensor payload, and this file ends at its exact metadata end.
    // Four fixed preflight reads precede two independent bounded window passes.
    const auto maximum_windows = (header_end + 65535) / 65536;
    check(!reads.crossed_header && reads.requested_end == header_end);
    check(reads.maximum_request <= 65536 && reads.windows == maximum_windows);
    check(reads.calls <= 4 + 2 * maximum_windows);

    MetadataReads cancelled{.header_end = header_end};
    cancelled.trigger_offset = 24;
    cancelled.cancel_after_refill = &cancel;
    {
        ObserveMetadataReads observe(cancelled);
        fails([&] { GgufMetadata metadata(path, cancel); }, "gguf_cancelled");
    }
    check(cancelled.triggered && cancelled.calls == 5 && !cancelled.crossed_header);
    cancel = false;
    MetadataReads changed{.header_end = header_end};
    changed.trigger_offset = 24;
    changed.append_after_refill = path.c_str();
    {
        ObserveMetadataReads observe(changed);
        fails([&] { GgufMetadata metadata(path, cancel); }, "gguf_checkpoint_changed");
    }
    check(changed.triggered && !changed.mutation_error && !changed.crossed_header);
    check(changed.maximum_request <= 65536);
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
    for (const auto& [name, bytes] :
         std::array<std::pair<std::string_view, const Bytes*>, 3>{
             {{"q3", &q3}, {"q4", &q4}, {"q8", &q8}}
         }) {
        const auto row_bytes = bytes->size() / 4;
        std::vector<std::byte> packed(row_bytes * 2);
        checkpoint.read_packed_rows(name, 1, 0, packed, cancel);
        for (std::size_t i = 0; i < packed.size(); ++i)
            check(
                std::to_integer<std::uint8_t>(packed[i]) == (*bytes)[2 * row_bytes + i]
            );
        packed.resize(row_bytes);
        checkpoint.read_packed_rows(name, 0, 1, packed, cancel);
        for (std::size_t i = 0; i < packed.size(); ++i)
            check(std::to_integer<std::uint8_t>(packed[i]) == (*bytes)[row_bytes + i]);
        checkpoint.read_packed_rows(name, 0, 2, {}, cancel);
        fails(
            [&] { checkpoint.read_packed_rows(name, 2, 0, packed, cancel); },
            "gguf_row_range"
        );
        fails(
            [&] { checkpoint.read_packed_rows(name, 0, 2, packed, cancel); },
            "gguf_row_range"
        );
        fails(
            [&] { checkpoint.read_packed_rows(name, 0, SIZE_MAX, {}, cancel); },
            "gguf_row_range"
        );
        fails(
            [&] {
                checkpoint.read_packed_rows(
                    name, 0, 0, std::span(packed).first(row_bytes - 1), cancel
                );
            },
            "gguf_row_range"
        );
    }
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
    std::array<std::byte, 110> packed{};
    fails(
        [&] { checkpoint.read_packed_rows("q3", 0, 0, packed, cancelled); },
        "gguf_cancelled"
    );
    fails(
        [&] { checkpoint.read_packed_rows("vector", 0, 0, {}, cancel); },
        "gguf_row_range"
    );
    fails(
        [&] { checkpoint.read_rows("q3", 0, 0, output, cancelled); }, "gguf_cancelled"
    );
    fails([&] { GgufCheckpoint bad(f.root, cancelled); }, "gguf_cancelled");
    std::ofstream(f.file(1), std::ios::app).put(0);
    fails([&] { checkpoint.check_unchanged(); }, "gguf_checkpoint_changed");
    fails(
        [&] { checkpoint.read_packed_rows("q3", 0, 0, packed, cancel); },
        "gguf_checkpoint_changed"
    );
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
    for (const auto& [name, bytes, row_bytes, rows] : std::array<
             std::tuple<std::string_view, const Bytes*, std::size_t, std::size_t>,
             2>{{{"dense", &dense, 4096, 260}, {"quantized", &quantized, 1760, 600}}}) {
        std::vector<std::byte> packed((rows - 2) * row_bytes);
        checkpoint.read_packed_rows(name, 1, 1, packed, cancel);
        const auto begin = (rows + 1) * row_bytes;
        for (std::size_t i = 0; i < packed.size(); ++i)
            check(std::to_integer<std::uint8_t>(packed[i]) == (*bytes)[begin + i]);
    }
    std::array<std::vector<std::byte>, 2> concurrent{
        std::vector<std::byte>(dense.size() / 2),
        std::vector<std::byte>(dense.size() / 2)
    };
    auto first = std::async(std::launch::async, [&] {
        checkpoint.read_packed_rows("dense", 0, 0, concurrent[0], cancel);
    });
    checkpoint.read_packed_rows("dense", 1, 0, concurrent[1], cancel);
    first.get();
    for (std::size_t expert = 0; expert < concurrent.size(); ++expert)
        for (std::size_t i = 0; i < concurrent[expert].size(); ++i)
            check(
                std::to_integer<std::uint8_t>(concurrent[expert][i])
                == dense[expert * concurrent[expert].size() + i]
            );
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

void single_file_tensors() {
    Fixture f;
    // Q6_K, IQ4_NL and BF16 rows; payload values are decoded by ggml.
    Bytes q6(210), iq4(18), bf16;
    q6[208] = 0x00;
    q6[209] = 0x3c;  // d = 1
    q6[192] = 2;     // scales for elements 0-15 and 64-79
    q6[196] = 2;
    q6[0] = 0x21;  // codes 1 and 2 before the -32 offset
    iq4[0] = 0x00;
    iq4[1] = 0x3c;
    for (float v : {1.f, -2.f})
        append(bf16, std::bit_cast<std::uint32_t>(v) >> 16, 2);
    f.write(
        0,
        {{"q6", 14, {256, 1}, q6},
         {"iq4", 20, {32, 1}, iq4},
         {"bf16", 30, {2, 1}, bf16},
         {"conv", 0, {2, 1, 2}, f32({1, 2, 3, 4})}},
        4,
        true
    );
    const std::atomic_bool cancel{false};
    GgufCheckpoint checkpoint(f.root, cancel);
    check(checkpoint.tensor_count() == 4);
    check(checkpoint.has("muse-spark.hash_offsets") && !checkpoint.has("split.no"));
    const auto offsets = checkpoint.integers("muse-spark.hash_offsets");
    check(offsets.size() == 2 && offsets[1] == 300001275);
    std::array<float, 256> q6_row{};
    checkpoint.read_rows("q6", 0, 0, q6_row, cancel);
    check(q6_row[0] == 2.f * float(1 - 32) && q6_row[64] == 2.f * float(2 - 32));
    std::array<float, 32> iq4_row{};
    checkpoint.read_rows("iq4", 0, 0, iq4_row, cancel);
    check(iq4_row[0] == -127.f);
    std::array<float, 2> bf16_row{};
    checkpoint.read_rows("bf16", 0, 0, bf16_row, cancel);
    check(bf16_row[0] == 1.f && bf16_row[1] == -2.f);
    // Small F32 tensors of any rank are read whole.
    std::array<float, 4> conv{};
    checkpoint.read_vector("conv", conv, cancel);
    check(conv[0] == 1.f && conv[3] == 4.f);
    std::array<float, 3> wrong{};
    fails([&] { checkpoint.read_vector("conv", wrong, cancel); }, "gguf_vector_shape");
}

void packed_f32_payload() {
    Fixture f;
    const auto bytes = f32({1, -2, std::numeric_limits<float>::infinity(), 4});
    f.write(0, {}, 1);
    f.write(1, {{"dense", 0, {2, 2}, bytes}}, 1);
    const std::atomic_bool cancel{false};
    GgufCheckpoint checkpoint(f.root, cancel);
    std::array<std::byte, 16> packed{};
    checkpoint.read_packed_rows("dense", 0, 0, packed, cancel);
    for (std::size_t i = 0; i < packed.size(); ++i)
        check(std::to_integer<std::uint8_t>(packed[i]) == bytes[i]);
    std::array<float, 4> decoded{};
    fails(
        [&] { checkpoint.read_rows("dense", 0, 0, decoded, cancel); },
        "gguf_nonfinite_weight"
    );
}
}  // namespace

int main() {
    try {
        buffered_metadata_boundaries();
        zero_tensor_metadata_buffering();
        reader_rows();
        reader_batched_rows();
        packed_f32_payload();
        single_file_tensors();
        metadata_failures();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
