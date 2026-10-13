// Resident H3 block layouts and session orchestration with a fake device; no
// CUDA, no projection arithmetic and no model execution.
#include "../src/models/video/h3_resident_block.hpp"

#include <array>
#include <cstring>
#include <map>
#include <source_location>
#include <string>
#include <vector>

#include "kadan/cleanup_error.hpp"

void check(bool b, std::source_location at = std::source_location::current()) {
    if (!b)
        throw std::runtime_error("test_failed:" + std::to_string(at.line()));
}

template <class F> void rejects(F f, std::string_view why) {
    try {
        f();
    } catch (const std::exception& e) {
        check(std::string_view(e.what()) == why);
        return;
    }
    check(false);
}

using kadan::DeviceDType;
using kadan::MediaDeviceSpan;
namespace detail = kadan::video::detail;

// Host bytes stand in for device memory; uploads and downloads copy metadata only.
struct Device final : kadan::MediaDeviceOperations {
    kadan::Bytes available = 64 * 1024 * 1024;
    std::map<void*, kadan::Bytes> allocations;
    std::map<int, kadan::Bytes> remaining;
    std::size_t next_event = 1, peers = 0, downloads = 0;
    std::uint32_t status = 0;
    std::size_t synchronizations = 0;

    std::size_t allocation_alignment() const override {
        return 256;
    }

    // Distinct per-device stream handles; launchers only compare them.
    std::array<char, 2> streams{};

    void* stream(int d) override {
        return &streams.at(std::size_t(d));
    }

    kadan::Bytes free_memory(int d) override {
        return remaining.try_emplace(d, available).first->second;
    }

    bool direct_peer(int, int) override {
        return true;
    }

    void* allocate(int d, kadan::Bytes bytes, std::size_t alignment) override {
        void* p = ::operator new(bytes, std::align_val_t(alignment));
        allocations.emplace(p, bytes);
        remaining.try_emplace(d, available).first->second -= bytes;
        return p;
    }

    void free(int d, void* p) override {
        remaining.at(d) += allocations.at(p);
        allocations.erase(p);
        ::operator delete(p, std::align_val_t(256));
    }

    void wait(int, kadan::MediaEvent) override {
    }

    kadan::MediaEvent record(int d) override {
        return {d, next_event++};
    }

    void synchronize(int) override {
        ++synchronizations;
    }

    void release_events(int) override {
    }

    void upload(int, void* p, std::span<const std::byte> raw) override {
        std::memcpy(p, raw.data(), raw.size());
    }

    void download(int, std::span<std::byte> raw, const void* p) override {
        ++downloads;
        std::memcpy(raw.data(), p, raw.size());
    }

    void peer(int, const void*, int, void*, kadan::Bytes) override {
        ++peers;
    }

    std::uint32_t read_status(int, const void*) override {
        return status;
    }
};

// Validates every launch with the shared layout checks; never reads tensor data.
// Calls: c convrot, s SiLU gate, r residual, n normalize, q q/k rotary, a attention.
struct Launcher final : kadan::video::H3ResidentLauncher {
    Device& backend;
    int selected;
    bool fail_silu = false, fail_release = false;
    std::size_t releases = 0;
    std::size_t key_tile = 0;
    std::string calls;
    std::vector<std::size_t> residual_index, normalize_shift;

    Launcher(Device& d, int id) : backend(d), selected(id) {
    }

    int device() const override {
        return selected;
    }

    void release_stream() override {
        ++releases;
        if (fail_release)
            throw std::runtime_error("release_failed");
    }

    void convrot(
        void* stream,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t group,
        float scale
    ) override {
        check(stream == backend.stream(selected) && group == 64 && scale == 0.5f);
        const auto shape =
            kadan::video::validate_h3_convrot(selected, read, write, group);
        check(shape.rank == 8 && shape.tile >= 4);
        calls.push_back('c');
        backend.status = 0;
    }

    void silu_gate(
        void*,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write
    ) override {
        kadan::video::validate_h3_silu_gate(selected, read, write);
        calls.push_back('s');
        backend.status = fail_silu ? 1 : 0;
    }

    void gated_residual(
        void*,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t first,
        std::size_t index
    ) override {
        kadan::video::validate_h3_gated_residual(selected, read, write, first, index);
        residual_index.push_back(index);
        calls.push_back('r');
        backend.status = 0;
    }

    void normalize(
        void*,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t first,
        std::size_t shift
    ) override {
        kadan::video::validate_h3_normalize(selected, read, write, first, shift);
        normalize_shift.push_back(shift);
        calls.push_back('n');
        backend.status = 0;
    }

    void qk_rotary(
        void*,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t first
    ) override {
        kadan::video::validate_h3_qk_rotary(selected, read, write, first);
        calls.push_back('q');
        backend.status = 0;
    }

    void attention(
        void*,
        std::span<const MediaDeviceSpan> read,
        std::span<const MediaDeviceSpan> write,
        std::size_t tile
    ) override {
        const auto shape =
            kadan::video::validate_h3_attention(selected, read, write, tile);
        check(shape.keys == tokens && shape.heads >= 1);
        key_tile = tile;
        calls.push_back('a');
        backend.status = 0;
    }

    static constexpr std::size_t tokens = 17;
};

constexpr std::size_t hidden = 64, width = 128, rank = 8, tokens = Launcher::tokens,
                      groups = 3, heads = 2, head_dim = 128, inner = heads * head_dim;

struct Identities {
    std::uint64_t inode = 1;

    kadan::WeightIdentity operator()(kadan::Bytes bytes) {
        return {1, inode++, 1, 0, 0, 0, 0, 0, bytes};
    }
};

detail::H3Projection projection(
    Identities& identity,
    const std::string& prefix,
    std::size_t rows,
    std::size_t columns,
    bool lora
) {
    detail::H3Projection p;
    p.weight = prefix + ".weight";
    p.scale = prefix + ".weight_scale";
    p.weight_id = identity(rows * columns);
    p.scale_id = identity(rows * 4);
    p.group = 64;
    if (!lora)
        return p;
    p.lora_a = "diffusion_model." + prefix + ".lora_A.weight";
    p.lora_b = "diffusion_model." + prefix + ".lora_B.weight";
    p.lora_a_id = identity(rank * columns * 2);
    p.lora_b_id = identity(rows * rank * 2);
    p.rank = rank;
    p.lora_scale = 0.5f;
    return p;
}

detail::H3FeedForward feed_forward(Identities& identity, bool lora) {
    detail::H3FeedForward ff;
    ff.hidden = hidden;
    ff.width = width;
    ff.fc[0] = projection(identity, "blocks.0.mlp.fc1", 2 * width, hidden, lora);
    ff.fc[1] = projection(identity, "blocks.0.mlp.fc2", hidden, width, lora);
    return ff;
}

detail::H3Block block(bool lora) {
    Identities identity;
    detail::H3Block b;
    b.heads = heads;
    b.head_dim = head_dim;
    b.ff = feed_forward(identity, lora);
    b.qkv = projection(identity, "blocks.0.attn.qkv_proj", 3 * inner, hidden, lora);
    b.out = projection(identity, "blocks.0.attn.out_proj", hidden, inner, lora);
    const std::array<const char*, 4> names{
        "norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight"
    };
    for (std::size_t i = 0; i < names.size(); ++i) {
        b.norm[i] = std::string("blocks.0.") + names[i];
        b.norm_id[i] = identity((i < 2 ? hidden : head_dim) * 2);
    }
    return b;
}

void layouts() {
    std::vector<std::byte> memory(4 << 20);
    std::size_t used = 0;
    auto span = [&](DeviceDType type,
                    std::vector<std::size_t> dimensions,
                    std::size_t element) {
        std::size_t count = 1;
        for (auto d : dimensions)
            count *= d;
        std::vector<std::size_t> strides(dimensions.size(), 1);
        if (dimensions.size() == 2)
            strides[0] = dimensions[1];
        MediaDeviceSpan s{
            memory.data() + used,
            0,
            type,
            std::move(dimensions),
            strides,
            count * element
        };
        used += (count * element + 255) / 256 * 256;
        return s;
    };
    const std::size_t rows = 3, in = 64, out = 8;
    const auto conversion_bytes = kadan::Bytes(16 * 1024);
    auto input = span(DeviceDType::f32, {rows, in}, 4);
    auto weight = span(DeviceDType::i8, {out, in}, 1);
    auto scale = span(DeviceDType::f32, {out, 1}, 4);
    auto a = span(DeviceDType::bf16, {rank, in}, 2);
    auto b = span(DeviceDType::bf16, {out, rank}, 2);
    auto output = span(DeviceDType::f32, {rows, out}, 4);
    auto conversion = span(DeviceDType::u8, {conversion_bytes}, 1);
    auto low = span(DeviceDType::f64, {rows, rank}, 8);
    auto widened = span(DeviceDType::f64, {rank * in + out * rank + rows * in}, 8);
    std::vector<std::byte> workspace(kadan::video::h3_resident_blas_bytes);
    MediaDeviceSpan blas{
        workspace.data(), 0, DeviceDType::u8, {workspace.size()}, {1}, workspace.size()
    };
    auto status = span(DeviceDType::u32, {1}, 4);

    // Scratch: aligned rotated F32, INT8 codes and row scales, then column tiles.
    check(kadan::video::h3_convrot_tile(3 * 256, 1, 64) == 0);
    check(kadan::video::h3_convrot_tile(3 * 256 + 64, 1, 64) == 16);
    check(kadan::video::h3_convrot_tile(conversion_bytes, rows, in) >= out);

    const std::vector plain_read{input, weight, scale};
    const std::vector plain_write{output, conversion, blas, status};
    auto shape = kadan::video::validate_h3_convrot(0, plain_read, plain_write, 64);
    check(shape.rows == rows && shape.in == in && shape.out == out && !shape.rank);
    check(shape.tile == out);
    const std::vector lora_read{input, weight, scale, a, b};
    const std::vector lora_write{output, conversion, low, widened, blas, status};
    shape = kadan::video::validate_h3_convrot(0, lora_read, lora_write, 64);
    check(shape.rank == rank);
    check(kadan::video::h3_convrot_tile(3 * 256 + 64, 1, 64, true) == 8);
    auto narrow_low = low;
    narrow_low.dtype = DeviceDType::f32;
    rejects(
        [&] {
            const std::vector write{
                output, conversion, narrow_low, widened, blas, status
            };
            kadan::video::validate_h3_convrot(0, lora_read, write, 64);
        },
        "h3_convrot_lora"
    );

    rejects(
        [&] { kadan::video::validate_h3_convrot(0, plain_read, plain_write, 32); },
        "h3_convrot_shape"
    );
    rejects(
        [&] { kadan::video::validate_h3_convrot(1, plain_read, plain_write, 64); },
        "h3_convrot_layout"
    );
    auto short_widened = widened;
    short_widened.dimensions = {rank * in};
    rejects(
        [&] {
            const std::vector write{
                output, conversion, low, short_widened, blas, status
            };
            kadan::video::validate_h3_convrot(0, lora_read, write, 64);
        },
        "h3_convrot_lora"
    );
    auto tiny = conversion;
    tiny.bytes = 256;
    tiny.dimensions = {256};
    rejects(
        [&] {
            const std::vector write{output, tiny, blas, status};
            kadan::video::validate_h3_convrot(0, plain_read, write, 64);
        },
        "h3_convrot_workspace"
    );
    // Writing into the input is an alias, never an in-place projection.
    auto aliased = output;
    aliased.data = input.data;
    rejects(
        [&] {
            const std::vector write{aliased, conversion, blas, status};
            kadan::video::validate_h3_convrot(0, plain_read, write, 64);
        },
        "h3_convrot_alias"
    );

    auto gate = span(DeviceDType::f32, {rows, out}, 4);
    auto value = span(DeviceDType::f32, {rows, out}, 4);
    const std::vector silu_read{gate, value};
    const std::vector silu_write{gate, status};
    check(kadan::video::validate_h3_silu_gate(0, silu_read, silu_write) == rows * out);
    rejects(
        [&] {
            const std::vector write{value, status};
            kadan::video::validate_h3_silu_gate(0, silu_read, write);
        },
        "h3_silu_layout"
    );

    // Residual and normalization share the [groups * 6, hidden] modulation.
    auto rows_hidden = span(DeviceDType::f32, {rows, hidden}, 4);
    auto projected = span(DeviceDType::f32, {rows, hidden}, 4);
    auto modulation = span(DeviceDType::f32, {groups * 6, hidden}, 4);
    auto token_groups = span(DeviceDType::u32, {tokens}, 4);
    const std::vector residual_read{rows_hidden, projected, modulation, token_groups};
    const std::vector residual_write{rows_hidden, status};
    check(
        kadan::video::validate_h3_gated_residual(
            0, residual_read, residual_write, 14, 5
        )
        == rows * hidden
    );
    rejects(
        [&] {
            kadan::video::validate_h3_gated_residual(
                0, residual_read, residual_write, 15, 5
            );
        },
        "h3_modulation_layout"
    );
    rejects(
        [&] {
            kadan::video::validate_h3_gated_residual(
                0, residual_read, residual_write, 0, 6
            );
        },
        "h3_residual_layout"
    );
    auto partial = span(DeviceDType::f32, {groups * 6 - 1, hidden}, 4);
    rejects(
        [&] {
            const std::vector read{rows_hidden, projected, partial, token_groups};
            kadan::video::validate_h3_gated_residual(0, read, residual_write, 0, 2);
        },
        "h3_modulation_layout"
    );

    auto norm = span(DeviceDType::bf16, {hidden, 1}, 2);
    auto normalized = span(DeviceDType::f32, {rows, hidden}, 4);
    const std::vector normalize_read{rows_hidden, norm, modulation, token_groups};
    const std::vector normalize_write{normalized, status};
    check(
        kadan::video::validate_h3_normalize(0, normalize_read, normalize_write, 0, 3)
        == rows
    );
    rejects(
        [&] {
            kadan::video::validate_h3_normalize(
                0, normalize_read, normalize_write, 0, 1
            );
        },
        "h3_normalize_layout"
    );
    rejects(
        [&] {
            const std::vector write{rows_hidden, status};
            kadan::video::validate_h3_normalize(0, normalize_read, write, 0, 0);
        },
        "h3_normalize_layout"
    );

    auto qk = span(DeviceDType::f32, {rows, inner}, 4);
    auto head_norm = span(DeviceDType::bf16, {head_dim, 1}, 2);
    auto table = span(DeviceDType::f64, {tokens, 96}, 8);
    const std::vector rotary_read{qk, head_norm, table};
    const std::vector rotary_write{qk, status};
    check(
        kadan::video::validate_h3_qk_rotary(0, rotary_read, rotary_write, 14)
        == rows * heads
    );
    rejects(
        [&] { kadan::video::validate_h3_qk_rotary(0, rotary_read, rotary_write, 15); },
        "h3_rotary_layout"
    );
    auto narrow = span(DeviceDType::f32, {rows, 96}, 4);
    rejects(
        [&] {
            const std::vector read{narrow, head_norm, table};
            const std::vector write{narrow, status};
            kadan::video::validate_h3_qk_rotary(0, read, write, 0);
        },
        "h3_rotary_layout"
    );

    auto keys = span(DeviceDType::f32, {tokens, inner}, 4);
    auto values = span(DeviceDType::f32, {tokens, inner}, 4);
    auto attended = span(DeviceDType::f32, {rows, inner}, 4);
    const auto score_count = heads * rows * 8;
    auto scores = span(DeviceDType::f64, {score_count}, 8);
    auto state = span(DeviceDType::f64, {2 * heads * rows}, 8);
    const auto wide_count =
        kadan::video::h3_attention_wide_bytes(heads, rows, 8) / sizeof(double);
    auto wide = span(DeviceDType::f64, {wide_count}, 8);
    const std::vector attention_read{qk, keys, values};
    const std::vector attention_write{attended, scores, state, wide, blas, status};
    const auto attention =
        kadan::video::validate_h3_attention(0, attention_read, attention_write, 8);
    check(
        attention.queries == rows && attention.keys == tokens
        && attention.heads == heads
    );
    rejects(
        [&] {
            kadan::video::validate_h3_attention(0, attention_read, attention_write, 9);
        },
        "h3_attention_layout"
    );
    rejects(
        [&] {
            const std::vector write{qk, scores, state, wide, blas, status};
            kadan::video::validate_h3_attention(0, attention_read, write, 8);
        },
        "h3_attention_alias"
    );
    for (std::size_t index : {1u, 2u, 3u}) {
        auto narrow_write = attention_write;
        narrow_write[index].dtype = DeviceDType::f32;
        rejects(
            [&] {
                kadan::video::validate_h3_attention(0, attention_read, narrow_write, 8);
            },
            "h3_attention_layout"
        );
    }
    auto short_write = attention_write;
    short_write[3].bytes -= sizeof(double);
    rejects(
        [&] { kadan::video::validate_h3_attention(0, attention_read, short_write, 8); },
        "h3_attention_layout"
    );
}

kadan::MediaPlanResult planned(
    const kadan::StageManifest& m, const kadan::MediaShape& shape, kadan::Bytes budget
) {
    const std::vector<kadan::DeviceBudget> budgets{{0, budget, budget}};
    return kadan::plan_media_stage(m, shape, budgets, {});
}

void manifests() {
    for (bool lora : {false, true}) {
        Identities identity;
        const auto ff = feed_forward(identity, lora);
        const auto m = detail::h3_feed_forward_manifest(ff, tokens, groups);
        check(
            m.schedule == kadan::MediaSchedule::feed_forward && m.workspaces_complete
        );
        check(m.weights.size() == (lora ? 8u : 4u));
        const auto& fc1 = m.weights[0];
        check(
            fc1.storage == kadan::TensorStorage::i8_convrot && fc1.planes == 2
            && fc1.rows == 2 * width && fc1.columns == hidden && fc1.alignment == 4
            && fc1.precision == kadan::ProjectionPrecision::convrot
            && fc1.rotation_group == 64
        );
        if (lora) {
            for (const auto& weight : m.weights)
                if (weight.storage == kadan::TensorStorage::bf16_exact)
                    check(weight.precision == kadan::ProjectionPrecision::precise_f64);
            for (const auto& tensor : m.tensors)
                if (tensor.name == "lora/low")
                    check(tensor.bytes == tokens * rank * sizeof(double));
        }
        std::size_t lora_tensors = 0;
        for (const auto& t : m.tensors)
            lora_tensors += t.name.starts_with("lora/");
        check(lora_tensors == (lora ? 2u : 0u));
        const auto plan =
            planned(m, detail::h3_feed_forward_shape(ff, tokens), 64 << 20);
        const auto* resident = std::get_if<kadan::MediaResidencyPlan>(&plan);
        check(resident && resident->devices.size() == 1 && resident->rows == tokens);

        const auto b = block(lora);
        const auto full = detail::h3_block_manifest(b, tokens, groups);
        check(full.schedule == kadan::MediaSchedule::full_block);
        for (const auto& tensor : full.tensors) {
            if (tensor.name == "attention/scores")
                check(tensor.bytes == heads * tokens * tokens * sizeof(double));
            if (tensor.name == "attention/state")
                check(tensor.bytes == 2 * heads * tokens * sizeof(double));
            if (tensor.name == "attention/wide")
                check(tensor.bytes == 4 * tokens * inner * sizeof(double));
        }
        check(full.weights.size() == 4 + (lora ? 16u : 8u));
        const auto qkv =
            std::find_if(full.weights.begin(), full.weights.end(), [&](auto& w) {
                return w.name == b.qkv.weight;
            });
        check(
            qkv != full.weights.end() && qkv->planes == 3
            && qkv->partition == kadan::WeightPartition::heads
        );
        const auto block_plan =
            planned(full, detail::h3_block_shape(b, tokens), 64 << 20);
        const auto* whole = std::get_if<kadan::MediaResidencyPlan>(&block_plan);
        check(
            whole && whole->devices.size() == 1 && whole->rows == tokens
            && whole->queries == tokens && whole->keys == tokens
        );
    }
}

struct Run {
    std::size_t rows = 0, queries = 0, devices = 0, key_tile = 0;
    std::vector<std::string> calls;
    std::vector<std::vector<std::size_t>> residual_index, normalize_shift;
    std::size_t peers = 0, downloads = 0;
};

Run run(kadan::Bytes budget, std::vector<int> ids, bool full, bool fail = false) {
    std::atomic_bool cancel = false;
    auto resources = std::make_shared<kadan::Resources>(
        kadan::Footprint{1ULL << 30, budget, budget}
    );
    auto backend = std::make_unique<Device>();
    auto& device = *backend;
    device.available = budget;
    std::vector<kadan::MediaDeviceLimit> limits;
    limits.reserve(ids.size());
    for (int id : ids)
        limits.push_back({id});
    kadan::MediaSession session(
        resources,
        std::make_shared<kadan::MediaSessionActivity>(),
        std::move(backend),
        kadan::Workload::video,
        limits,
        cancel
    );
    const auto b = block(true);
    check(
        !(full ? session.admit(
                     detail::h3_block_manifest(b, tokens, groups),
                     detail::h3_block_shape(b, tokens)
                 )
               : session.admit(
                     detail::h3_feed_forward_manifest(b.ff, tokens, groups),
                     detail::h3_feed_forward_shape(b.ff, tokens)
                 ))
    );
    for (const auto& slice : session.plan().weights) {
        auto view = session.weight(slice);
        std::vector<std::byte> raw(slice.stored_bytes);
        session.upload_weight(view, raw);
    }
    std::vector<float> normalized(tokens * hidden), state(tokens * hidden, 1.f),
        modulation(groups * 6 * hidden);
    std::vector<double> rotary(tokens * 96);
    std::vector<std::uint32_t> token_groups(tokens);
    for (std::size_t t = 0; t < tokens; ++t)
        token_groups[t] = std::uint32_t(t % groups);
    if (full)
        detail::upload_h3_block_inputs(
            session, b, state, modulation, token_groups, rotary
        );
    else
        detail::upload_h3_feed_forward_inputs(
            session, b.ff, normalized, state, modulation, token_groups
        );
    std::vector<std::unique_ptr<Launcher>> owned;
    std::vector<kadan::video::H3ResidentLauncher*> launchers;
    for (int id : ids) {
        owned.push_back(std::make_unique<Launcher>(device, id));
        owned.back()->fail_silu = fail && id == ids.front();
        launchers.push_back(owned.back().get());
    }
    Run result;
    result.rows = session.plan().rows;
    result.queries = session.plan().queries;
    result.devices = session.plan().devices.size();
    try {
        if (full)
            detail::execute_h3_block(session, launchers, b, tokens, state, cancel);
        else
            detail::execute_h3_feed_forward(
                session, launchers, b.ff, tokens, state, cancel
            );
    } catch (...) {
        session.abort();
        throw;
    }
    session.finish();
    for (const auto& l : owned) {
        result.calls.push_back(l->calls);
        result.residual_index.push_back(l->residual_index);
        result.normalize_shift.push_back(l->normalize_shift);
        result.key_tile = l->key_tile;
    }
    result.peers = device.peers;
    result.downloads = device.downloads;
    // The downloaded hidden is the uploaded baseline; launchers never wrote it.
    for (float v : state)
        check(v == 1.f);
    return result;
}

std::string repeat(const char* calls, std::size_t tile) {
    std::string result;
    for (std::size_t first = 0; first < tokens; first += tile)
        result += calls;
    return result;
}

// Lower the budget until the planner splits the stage over two devices.
Run split(bool full) {
    for (kadan::Bytes budget = 8 * 1024 * 1024 + 2048 * 1024; budget >= 8 * 1024 * 1024;
         budget -= 1024) {
        Run two;
        try {
            two = run(budget, {0, 1}, full);
        } catch (const std::exception&) {
            continue;
        }
        if (two.devices == 2)
            return two;
    }
    check(false);
    return {};
}

void orchestration() {
    // FF only: normalized input arrives from the host.
    auto single = run(64 << 20, {0}, false);
    check(single.devices == 1 && single.rows == tokens);
    check(single.calls == std::vector<std::string>{"ccscr"});
    check(single.residual_index[0] == std::vector<std::size_t>{5});
    check(single.peers == 0 && single.downloads == 1);

    // Two devices split FF channels and gather through direct peer copies.
    auto two = split(false);
    check(two.calls == std::vector<std::string>(2, repeat("ccscr", two.rows)));
    check(two.peers > 0 && two.downloads == 1);

    // Whole block: K/V tiles (norm1, K, k-norm/RoPE, V), query tiles (norm1, Q,
    // q-norm/RoPE, attention, out_proj, residual 2), then FF tiles (norm2, FF).
    auto whole = run(64 << 20, {0}, true);
    check(whole.devices == 1 && whole.rows == tokens && whole.queries == tokens);
    check(whole.key_tile == tokens);
    check(whole.calls == std::vector<std::string>{"ncqcncqacrnccscr"});
    check(whole.residual_index[0] == (std::vector<std::size_t>{2, 5}));
    check(whole.normalize_shift[0] == (std::vector<std::size_t>{0, 0, 3}));
    check(whole.peers == 0 && whole.downloads == 1);

    // Two devices split heads for Q/K/V and attention, gathering attention
    // outputs before out_proj and every channel-split projection after it.
    auto heads_split = split(true);
    const auto expected = repeat("ncqc", heads_split.rows)
                          + repeat("ncqacr", heads_split.queries)
                          + repeat("nccscr", heads_split.rows);
    check(heads_split.calls == std::vector<std::string>(2, expected));
    check(heads_split.peers > 0 && heads_split.downloads == 1);

    rejects([] { run(64 << 20, {0}, false, true); }, "media_session_nonfinite_output");
    rejects([] { run(64 << 20, {0}, true, true); }, "media_session_nonfinite_output");
}

// Exercise the same cleanup boundary used by the denoiser, without inference.
void cleanup() {
    for (bool full : {false, true})
        for (bool abort : {false, true}) {
            std::atomic_bool cancel = false;
            auto resources = std::make_shared<kadan::Resources>(
                kadan::Footprint{1ULL << 30, 64ULL << 20, 64ULL << 20}
            );
            auto activity = std::make_shared<kadan::MediaSessionActivity>();
            auto backend = std::make_unique<Device>();
            auto& device = *backend;
            kadan::MediaSession session(
                resources,
                activity,
                std::move(backend),
                kadan::Workload::video,
                {{0}, {1}},
                cancel
            );
            const auto b = block(true);
            check(!session.admit(
                full ? detail::h3_block_manifest(b, tokens, groups)
                     : detail::h3_feed_forward_manifest(b.ff, tokens, groups),
                full ? detail::h3_block_shape(b, tokens)
                     : detail::h3_feed_forward_shape(b.ff, tokens)
            ));
            check(!device.allocations.empty());
            Launcher first(device, 0), second(device, 1);
            first.fail_release = true;
            const std::vector<kadan::video::H3ResidentLauncher*> launchers{
                &first, &second
            };
            if (abort) {
                const auto original =
                    std::make_exception_ptr(std::runtime_error("operation_failed"));
                try {
                    kadan::rethrow_after_cleanup(original, [&] {
                        kadan::video::detail::close_h3_resident(
                            session, launchers, true
                        );
                    });
                } catch (const kadan::OperationCleanupError& error) {
                    check(error.operation == original);
                    check(kadan::exception_message(error.cleanup) == "release_failed");
                }
            } else {
                rejects(
                    [&] {
                        kadan::video::detail::close_h3_resident(session, launchers);
                    },
                    "release_failed"
                );
            }
            check(first.releases == 1 && second.releases == 1);
            check(device.synchronizations > 0 && device.allocations.empty());
            check(activity->owners.load() == 0);
        }
}

int main() {
    layouts();
    manifests();
    orchestration();
    cleanup();
}
