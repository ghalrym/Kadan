// Resident H3 feed-forward layouts and session orchestration with a fake device;
// no CUDA, no projection arithmetic and no model execution.
#include "../src/models/video/h3_feed_forward.hpp"

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

// Host bytes stand in for device memory; uploads and downloads copy metadata only.
struct Device final : kadan::MediaDeviceOperations {
    kadan::Bytes available = 64 * 1024 * 1024;
    std::map<void*, kadan::Bytes> allocations;
    std::map<int, kadan::Bytes> remaining;
    std::map<int, int> streams;
    std::size_t next_event = 1, peers = 0, downloads = 0;
    std::uint32_t status = 0;
    std::size_t synchronizations = 0;

    std::size_t allocation_alignment() const override {
        return 256;
    }

    void* stream(int d) override {
        return &streams[d];
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
struct Launcher final : kadan::video::H3ResidentLauncher {
    Device& backend;
    int selected;
    bool fail_silu = false, fail_release = false;
    std::string calls;
    std::vector<std::size_t> residual_first, released;

    Launcher(Device& d, int id) : backend(d), selected(id) {
    }

    int device() const override {
        return selected;
    }

    void release_stream() override {
        released.push_back(calls.size());
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
        std::size_t first
    ) override {
        kadan::video::validate_h3_gated_residual(selected, read, write, first);
        residual_first.push_back(first);
        calls.push_back('r');
        backend.status = 0;
    }
};

constexpr std::size_t hidden = 64, width = 128, rank = 8, tokens = 17, groups = 3;

kadan::video::detail::H3FeedForward contract(bool lora) {
    kadan::video::detail::H3FeedForward ff;
    ff.hidden = hidden;
    ff.width = width;
    std::uint64_t inode = 1;
    auto identity = [&](kadan::Bytes bytes) {
        return kadan::WeightIdentity{1, inode++, 1, 0, 0, 0, 0, 0, bytes};
    };
    for (std::size_t fc = 0; fc < 2; ++fc) {
        const auto rows = fc ? hidden : 2 * width, columns = fc ? width : hidden;
        const auto p = "blocks.0.mlp.fc" + std::to_string(fc + 1);
        ff.weight[fc] = p + ".weight";
        ff.scale[fc] = p + ".weight_scale";
        ff.weight_id[fc] = identity(rows * columns);
        ff.scale_id[fc] = identity(rows * 4);
        ff.group[fc] = 64;
        if (!lora)
            continue;
        ff.lora_a[fc] = "diffusion_model." + p + ".lora_A.weight";
        ff.lora_b[fc] = "diffusion_model." + p + ".lora_B.weight";
        ff.lora_a_id[fc] = identity(rank * columns * 2);
        ff.lora_b_id[fc] = identity(rows * rank * 2);
        ff.rank[fc] = rank;
        ff.lora_scale[fc] = 0.5f;
    }
    return ff;
}

void layouts() {
    std::vector<std::byte> memory(1 << 20);
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
    auto blas = span(DeviceDType::u8, {1}, 1);
    blas.bytes = kadan::video::h3_resident_blas_bytes;
    blas.dimensions = {kadan::video::h3_resident_blas_bytes};
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
    auto f32_weight = weight;
    f32_weight.dtype = DeviceDType::f32;
    rejects(
        [&] {
            const std::vector read{input, f32_weight, scale};
            kadan::video::validate_h3_convrot(0, read, plain_write, 64);
        },
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

    auto rows_hidden = span(DeviceDType::f32, {rows, hidden}, 4);
    auto projected = span(DeviceDType::f32, {rows, hidden}, 4);
    auto gates = span(DeviceDType::f32, {groups, hidden}, 4);
    auto token_groups = span(DeviceDType::u32, {tokens}, 4);
    const std::vector residual_read{rows_hidden, projected, gates, token_groups};
    const std::vector residual_write{rows_hidden, status};
    check(
        kadan::video::validate_h3_gated_residual(0, residual_read, residual_write, 14)
        == rows * hidden
    );
    rejects(
        [&] {
            kadan::video::validate_h3_gated_residual(
                0, residual_read, residual_write, 15
            );
        },
        "h3_residual_layout"
    );
    auto ten_groups = span(DeviceDType::f32, {10, hidden}, 4);
    rejects(
        [&] {
            const std::vector read{rows_hidden, projected, ten_groups, token_groups};
            kadan::video::validate_h3_gated_residual(0, read, residual_write, 0);
        },
        "h3_residual_layout"
    );
}

void manifest() {
    for (bool lora : {false, true}) {
        const auto m = kadan::video::detail::h3_feed_forward_manifest(
            contract(lora), tokens, groups
        );
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
                if (tensor.name == "ff/lora-low")
                    check(tensor.bytes == tokens * rank * sizeof(double));
        }
        std::size_t lora_tensors = 0;
        for (const auto& t : m.tensors)
            lora_tensors += t.name.starts_with("ff/lora-");
        check(lora_tensors == (lora ? 2u : 0u));
        const std::vector<kadan::DeviceBudget> budgets{
            {0, 64 * 1024 * 1024, 64 * 1024 * 1024}
        };
        const auto planned =
            kadan::plan_media_stage(m, {tokens, hidden, 0, 0, 0, width}, budgets, {});
        const auto* plan = std::get_if<kadan::MediaResidencyPlan>(&planned);
        check(plan && plan->devices.size() == 1 && plan->rows == tokens);
    }
}

struct Run {
    std::size_t rows = 0, devices = 0;
    std::vector<std::string> calls;
    std::vector<std::vector<std::size_t>> residual_first;
    std::size_t peers = 0, downloads = 0;
};

Run run(kadan::Bytes budget, std::vector<int> ids, bool fail = false) {
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
    const auto ff = contract(true);
    check(!session.admit(
        kadan::video::detail::h3_feed_forward_manifest(ff, tokens, groups),
        {tokens, hidden, 0, 0, 0, width}
    ));
    for (const auto& slice : session.plan().weights) {
        auto view = session.weight(slice);
        std::vector<std::byte> raw(slice.stored_bytes);
        session.upload_weight(view, raw);
    }
    std::vector<float> normalized(tokens * hidden), state(tokens * hidden, 1.f),
        gates(groups * hidden);
    std::vector<std::uint32_t> token_groups(tokens);
    for (std::size_t t = 0; t < tokens; ++t)
        token_groups[t] = std::uint32_t(t % groups);
    kadan::video::detail::upload_h3_feed_forward_inputs(
        session, ff, normalized, state, gates, token_groups
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
    result.devices = session.plan().devices.size();
    try {
        kadan::video::detail::execute_h3_feed_forward(
            session, launchers, ff, tokens, groups, state, cancel
        );
    } catch (...) {
        session.abort();
        throw;
    }
    session.finish();
    for (const auto& l : owned) {
        result.calls.push_back(l->calls);
        result.residual_first.push_back(l->residual_first);
    }
    result.peers = device.peers;
    result.downloads = device.downloads;
    // The downloaded hidden is the uploaded baseline; launchers never wrote it.
    for (float v : state)
        check(v == 1.f);
    return result;
}

void orchestration() {
    // One device with room for every token: one tile, four projections.
    auto single = run(64 * 1024 * 1024, {0});
    check(single.devices == 1 && single.rows == tokens);
    check(single.calls == std::vector<std::string>{"ccscr"});
    check(single.residual_first[0] == std::vector<std::size_t>{0});
    check(single.peers == 0 && single.downloads == 1);

    // A budget below one device's need splits channels over two devices and
    // gathers fc1 activations and fc2 outputs through direct peer copies.
    bool split = false;
    for (kadan::Bytes budget = 8 * 1024 * 1024 + 512 * 1024;
         !split && budget >= 8 * 1024 * 1024;
         budget -= 1024) {
        Run two;
        try {
            two = run(budget, {0, 1});
        } catch (const std::exception&) {
            continue;
        }
        if (two.devices != 2)
            continue;
        split = true;
        const auto tiles = (tokens + two.rows - 1) / two.rows;
        std::string expected;
        for (std::size_t i = 0; i < tiles; ++i)
            expected += "ccscr";
        check(two.calls == std::vector<std::string>(2, expected));
        check(two.residual_first[0] == two.residual_first[1]);
        check(two.peers > 0 && two.downloads == 1);
    }
    check(split);

    rejects([] { run(64 * 1024 * 1024, {0}, true); }, "media_session_nonfinite_output");
}

// Exercise the same cleanup boundary used by the denoiser, without inference.
void cleanup() {
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
        check(!session.admit(
            kadan::video::detail::h3_feed_forward_manifest(
                contract(true), tokens, groups
            ),
            {tokens, hidden, 0, 0, 0, width}
        ));
        check(!device.allocations.empty());
        Launcher first(device, 0), second(device, 1);
        first.fail_release = true;
        const std::vector<kadan::video::H3ResidentLauncher*> launchers{&first, &second};
        if (abort) {
            const auto original =
                std::make_exception_ptr(std::runtime_error("operation_failed"));
            try {
                kadan::rethrow_after_cleanup(original, [&] {
                    kadan::video::detail::close_h3_feed_forward(
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
                    kadan::video::detail::close_h3_feed_forward(session, launchers);
                },
                "release_failed"
            );
        }
        check(first.released.size() == 1 && second.released.size() == 1);
        check(device.synchronizations > 0 && device.allocations.empty());
        check(activity->owners.load() == 0);
    }
}

int main() {
    layouts();
    manifest();
    orchestration();
    cleanup();
}
