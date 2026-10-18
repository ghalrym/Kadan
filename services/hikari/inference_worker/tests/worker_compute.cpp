#include "kadan/worker_compute.hpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <limits>

#include "kadan/image_attention.hpp"
#include "kadan/video_progress.hpp"
#include "kadan/weight_progress.hpp"
using namespace kadan;

void check(bool b) {
    if (!b)
        throw std::runtime_error("compute_plan_test");
}

template <class F> void fails(F f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected_rejection");
}

struct FakeContext final : DenseCompute {
    std::vector<DeviceWeightPlacement> placement;
    mutable unsigned snapshots = 0;

    std::vector<DeviceWeightPlacement> weight_placement() const override {
        ++snapshots;
        return placement;
    }

    bool fail = false, scratch_fail = false;
    std::size_t releases = 0, scratch_releases = 0;
    std::shared_ptr<Resources> resources;
    Handle scratch = 0;

    void release_scratch() override {
        ++scratch_releases;
        if (scratch_fail)
            throw std::runtime_error("scratch_cleanup_unconfirmed");
        if (scratch) {
            resources->released(scratch);
            scratch = 0;
        }
    }

    void dense(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
    }

    void release_devices() override {
        ++releases;
        if (fail)
            throw std::runtime_error("cleanup_unconfirmed");
    }
};

void projection_output_classification() {
    for (std::uint32_t bits :
         {0U,
          0x80000000U,
          1U,
          0x80000001U,
          0x007fffffU,
          0x807fffffU,
          0x00800000U,
          0x80800000U,
          0x7f7fffffU,
          0xff7fffffU,
          0x7f800000U,
          0xff800000U,
          0x7f800001U,
          0xff800001U,
          0x7fc00000U,
          0xffc00000U,
          0x7fffffffU,
          0xffffffffU}) {
        check(
            projection_output_finite(bits) == std::isfinite(std::bit_cast<float>(bits))
        );
    }
}

void image_attention_precision() {
    using image::AttentionPrecision;
    check(image::attention_precision_name(AttentionPrecision{}) == "pedantic_fp32");
    // Fake nested handle scopes restore the caller's math mode on unwind.
    for (bool enabled : {false, true})
        for (bool interrupted : {false, true}) {
            int mode = 23;
            try {
                with_projection_math(
                    true,
                    17,
                    [&] { return mode; },
                    [&](int next) { mode = next; },
                    [&] {
                        check(mode == 17);
                        with_projection_math(
                            enabled,
                            23,
                            [&] { return mode; },
                            [&](int next) { mode = next; },
                            [&] {
                                check(mode == (enabled ? 23 : 17));
                                if (interrupted)
                                    throw std::runtime_error("cancelled_or_gemm_error");
                            }
                        );
                        check(mode == 17);
                    }
                );
                check(!interrupted);
            } catch (const std::runtime_error& error) {
                check(
                    interrupted
                    && std::string_view(error.what()) == "cancelled_or_gemm_error"
                );
            }
            check(mode == 23);
        }
}

void projection_math() {
    for (auto workload :
         {Workload::llm,
          Workload::image,
          Workload::video,
          Workload::speech,
          Workload::tts,
          Workload::decision})
        for (int major : {7, 8, 9})
            for (bool precise : {false, true}) {
                check(!projection_uses_tf32(
                    ProjectionPolicy::standard, workload, major, precise
                ));
                check(
                    projection_uses_tf32(
                        ProjectionPolicy::image_feed_forward, workload, major, precise
                    )
                    == (workload == Workload::image && major >= 8 && !precise)
                );
            }

    // Fake handle state only: no device, tensor, projection or model execution.
    for (bool enabled : {false, true}) {
        int mode = 17, gets = 0, sets = 0, calls = 0;
        with_projection_math(
            enabled,
            23,
            [&] {
                ++gets;
                return mode;
            },
            [&](int next) {
                ++sets;
                mode = next;
            },
            [&] {
                ++calls;
                check(mode == (enabled ? 23 : 17));
            }
        );
        check(
            mode == 17 && calls == 1 && gets == int(enabled) && sets == 2 * int(enabled)
        );
    }
    // Preserve the exact previous mode, including an already-fast handle.
    for (int previous : {17, 23}) {
        int mode = previous;
        try {
            with_projection_math(
                true,
                23,
                [&] { return mode; },
                [&](int next) { mode = next; },
                [] { throw std::runtime_error("cancelled_or_gemm_error"); }
            );
            check(false);
        } catch (const std::runtime_error& e) {
            check(std::string(e.what()) == "cancelled_or_gemm_error");
        }
        check(mode == previous);
    }
    // Even a failing enable may have changed external state; attempt restoration.
    {
        int mode = 17, calls = 0;
        try {
            with_projection_math(
                true,
                23,
                [&] { return mode; },
                [&](int next) {
                    mode = next;
                    if (next == 23)
                        throw std::runtime_error("set_failed");
                },
                [&] { ++calls; }
            );
            check(false);
        } catch (const std::runtime_error& e) {
            check(std::string(e.what()) == "set_failed");
        }
        check(mode == 17 && calls == 0);
    }
    // Restoration failures must propagate to existing workspace cleanup, never
    // allow a contaminated handle to be silently reused or hide the first error.
    for (bool operation_fails : {false, true}) {
        int mode = 17;
        try {
            with_projection_math(
                true,
                23,
                [&] { return mode; },
                [&](int next) {
                    if (next == 17)
                        throw std::runtime_error("restore_failed");
                    mode = next;
                },
                [&] {
                    if (operation_fails)
                        throw std::runtime_error("gemm_failed");
                }
            );
            check(false);
        } catch (const OperationCleanupError& e) {
            check(operation_fails);
            check(exception_message(e.operation) == "gemm_failed");
            check(exception_message(e.cleanup) == "restore_failed");
        } catch (const std::runtime_error& e) {
            check(!operation_fails && std::string(e.what()) == "restore_failed");
        }
    }
}

void vae_normalization_planning() {
    constexpr Bytes limit = 2ULL * 1024 * 1024 * 1024;
    const auto small = image_vae_normalization_plan(3, 5, 1024);
    check(small && small->points == 5 && small->bytes == 1024);
    check(!image_vae_normalization_plan(3, 5, 1023));
    const auto largest = image_vae_normalization_plan(1152, 2048 * 2048, limit);
    check(largest && largest->points == 65536 && largest->bytes == 603984640);
    for (std::size_t channels : {1u, 144u, 1152u})
        for (std::size_t points : {1u, 17u, 65535u, 65537u, 2048u * 2048u})
            for (Bytes budget : std::array<Bytes, 4>{1024, 8192, 1024 * 1024, limit}) {
                const auto plan =
                    image_vae_normalization_plan(channels, points, budget);
                if (!plan)
                    continue;
                check(
                    plan->points > 0 && plan->points <= points && plan->points <= 65536
                    && plan->bytes <= budget
                );
                check(
                    image_vae_normalization_plan(channels, points, plan->bytes)->points
                    == plan->points
                );
                std::size_t covered = 0;
                for (std::size_t first = 0; first < points; first += plan->points)
                    covered += std::min(plan->points, points - first);
                check(covered == points);
            }
    check(!image_vae_normalization_plan(SIZE_MAX, 1, SIZE_MAX));
    check(!image_vae_normalization_plan(1, SIZE_MAX, SIZE_MAX));
    check(!image_vae_normalization_plan(1, 2147483647ULL / 4 + 1, SIZE_MAX));
    fails([] { image_vae_normalization_plan(0, 1, limit); });
    fails([] { image_vae_normalization_plan(1, 0, limit); });
    FakeContext unsupported;
    std::atomic_bool cancel = false;
    std::array<float, 1> input{1}, gamma{1}, output{17};
    check(!unsupported.image_vae_normalize(input, gamma, 1, output, false, cancel));
    check(output[0] == 17);
}

void image_pair_planning() {
    constexpr Bytes limit = 2ULL * 1024 * 1024 * 1024;
    auto aligned = [](Bytes n) {
        return (n + 255) / 256 * 256;
    };
    for (std::size_t in : {1u, 17u, 4096u, 12288u, 2147483647u})
        for (std::size_t out : {1u, 3u, 341u, 12288u}) {
            if (in > 16 * 1024 * 1024 / 2) {
                fails([&] { image_silu_pair_plan(in, out, limit); });
                continue;
            }
            const auto plan = image_silu_pair_plan(in, out, limit);
            check(plan.columns && plan.columns <= out && plan.bytes <= limit);
            check(plan.columns * in * 2 <= 16 * 1024 * 1024);
            const auto layout = aligned(plan.rows * in * 4)
                                + aligned(plan.rows * in * 2)
                                + 2 * aligned(plan.columns * in * 2)
                                + 2 * aligned(plan.rows * plan.columns * 4)
                                + aligned(8 * 1024 * 1024) + aligned(4);
            check(layout <= plan.bytes);
            // Column tails are complete; neither cached slot changes scratch admission.
            std::size_t visited = 0;
            for (std::size_t first = 0; first < out; first += plan.columns)
                visited += std::min(plan.columns, out - first);
            check(visited == out);
            const auto rows = plan.rows + 17;
            const auto exact = plan.bytes + 17 * in * 6;
            check(image_silu_pair_input_rows(rows, in, plan, exact) == rows);
            check(image_silu_pair_input_rows(rows, in, plan, exact - 1) == plan.rows);
            check(image_silu_pair_input_rows(SIZE_MAX, in, plan, limit) == plan.rows);
        }
    fails([] { image_silu_pair_plan(0, 1, limit); });
    fails([] { image_silu_pair_plan(SIZE_MAX, 1, limit); });
    fails([] { image_silu_pair_plan(1, SIZE_MAX, limit); });
    fails([] { image_silu_pair_plan(4096, 12288, 8 * 1024 * 1024); });
    const auto small = image_silu_pair_plan(4096, 12288, 9 * 1024 * 1024);
    check(small.bytes <= 9 * 1024 * 1024 && small.rows < 256);
    const auto plan = image_silu_pair_plan(4096, 12288, limit);
    fails([&] { image_silu_pair_input_rows(0, 4096, plan, limit); });
    fails([&] { image_silu_pair_input_rows(1, 0, plan, limit); });
    fails([&] { image_silu_pair_input_rows(1, 4096, plan, plan.bytes - 1); });
    FakeContext unsupported;
    std::atomic_bool cancel = false;
    unsigned reads = 0;
    DenseCompute::Bf16WeightSource source = [&](std::size_t, std::span<std::uint16_t>) {
        ++reads;
    };
    std::array<float, 1> input{1}, output{17};
    check(!unsupported.image_silu_pair_bf16_source(
        {}, source, {}, source, input, 1, 1, output, cancel
    ));
    check(reads == 0 && output[0] == 17);
}

int main() {
    projection_output_classification();
    image_pair_planning();
    vae_normalization_planning();
    image_attention_precision();
    projection_math();
    // Exercise only progress events, without a model, tensor or inference call.
    std::vector<std::pair<std::string, std::size_t>> updates;
    video::VideoProgress progress([&](const char* phase, std::size_t count) {
        updates.emplace_back(phase, count);
    });
    for (std::size_t head = 1; head <= 363040; ++head)
        progress.update("rope", head);
    progress.update("block_completed", 1);
    check(updates.size() == 91);
    check(updates.front() == std::pair<std::string, std::size_t>{"rope", 1});
    check(updates[89] == std::pair<std::string, std::size_t>{"rope", 363040});
    check(updates.back() == std::pair<std::string, std::size_t>{"block_completed", 1});
    updates.clear();
    progress.update("rope", 1);
    progress.update("rope", 100);
    progress.reset();
    progress.update("rope", 1);
    progress.update("block_loaded", 2);
    check(
        updates.size() == 3 && updates[1].second == 1
        && updates.back().first == "block_loaded"
    );
    updates.clear();
    progress.update("rope", 1);
    progress.update("rope", 50);
    progress.update("rope", 1);
    check(updates.size() == 3 && updates[1].second == 50 && updates.back().second == 1);
    progress.update("gpu_0_weight_allocated_bytes", 123);
    progress.update("gpu_0_weight_allocated_bytes", 456);
    check(updates.back().second == 456);

    // Existing progress wire format: exact byte counts, physical device identity,
    // and distinct inventory/planned/allocated values; never executes a model.
    checkpoint::WeightInventory inventory;
    inventory.stored_bytes = 31581ULL * 1024 * 1024 + 17;
    inventory.floating_f32_bytes = 61874ULL * 1024 * 1024 + 33;
    std::vector<std::pair<std::string, Bytes>> events;
    std::function<void(const char*, std::size_t)> hook = [&](const char* name,
                                                             std::size_t bytes) {
        events.emplace_back(name, bytes);
    };
    FakeContext placement;
    placement.placement = {{1, 9000000000ULL, 1024}, {0, 5000000000ULL, 0}};
    report_weight_inventory(inventory, hook);
    report_weight_placement(placement, hook);
    check(
        events
        == std::vector<std::pair<std::string, Bytes>>(
            {{"checkpoint_tensor_bytes", inventory.stored_bytes},
             {"expanded_f32_inventory_bytes", inventory.floating_f32_bytes},
             {"nonfloating_inventory_bytes", 0},
             {"gpu_1_weight_planned_bytes", 9000000000ULL},
             {"gpu_1_weight_allocated_bytes", 1024},
             {"gpu_0_weight_planned_bytes", 5000000000ULL},
             {"gpu_0_weight_allocated_bytes", 0}}
        )
    );
    events.clear();
    placement.placement[0].allocated_bytes = 2048;
    report_weight_placement(placement, hook);
    check(events[1].second == 2048 && events[0].second == 9000000000ULL);
    const auto snapshots = placement.snapshots;
    std::function<void(const char*, std::size_t)> absent;
    report_weight_placement(placement, absent);
    check(placement.snapshots == snapshots);
    events.clear();
    placement.placement.clear();
    report_weight_placement(placement, hook);
    check(events.empty());

    check(projection_split_columns(1, 2));
    check(!projection_split_columns(2, 2));
    check(!projection_split_columns(1, 1));
    check(attention_visible(0, 0, 2, 0, 0));
    check(!attention_visible(0, 1, 2, 0, 0));
    check(
        attention_visible(2, 3, 2, 0, 0)
    );  // Image rows see all keys after causal text.
    check(attention_visible(0, 4, 1, 0, 4));
    check(!attention_visible(0, 5, 1, 0, 4));  // Cached last query.
    check(!attention_visible(4, 1, 5, 3, 0));
    check(attention_visible(4, 2, 5, 3, 0));
    check(!attention_visible(4, 5, 5, 3, 0));

    check(projection_columns(3072, 96768, 4, true, 1) < 96768);
    check(projection_columns(3072, 96768, 1, false, 4) % 4 == 0);
    check(projection_columns(24, 24, 4, false, 1) == 24);
    fails([] { projection_columns(2147483647, 4, 4, true, 1); });
    // Pure workspace arithmetic: independently include every aligned CUDA slice.
    const auto aligned = [](std::initializer_list<Bytes> sizes) {
        Bytes n = 0;
        for (auto size : sizes) {
            n = (n + 255) / 256 * 256;
            n += size;
        }
        return (n + 255) / 256 * 256;
    };
    for (Bytes limit :
         {16ULL * 1024 * 1024, 64ULL * 1024 * 1024, 2ULL * 1024 * 1024 * 1024}) {
        for (std::size_t width : {1U, 2U, 4U})
            for (bool precise : {false, true})
                for (std::size_t in : {24U, 3072U, 4096U, 16384U}) {
                    const auto plan = projection_plan(
                        in, 96768, width, precise, width == 1 ? 4 : 1, limit
                    );
                    const Bytes t = plan.rows, o = plan.columns, w = in * o;
                    const auto bytes = aligned(
                        {sizeof(unsigned),
                         w * width,
                         o * 4,
                         o * 4,
                         t * in * 4,
                         t * in * 4,
                         t * in,
                         t * 4,
                         t * o * 4,
                         t * o * 4,
                         8 * 1024 * 1024,
                         precise ? w * 8 : 0,
                         precise ? t * in * 8 : 0,
                         precise ? t * o * 8 : 0}
                    );
                    check(
                        bytes <= plan.bytes && plan.bytes <= limit && o > 0 && t > 0
                        && t <= 256 && (width != 1 || o % 4 == 0)
                    );
                }
        for (std::size_t width : {4U, 8U})
            for (std::size_t n : {1U, 129U, 18432U})
                for (std::size_t keys : {3U, 4097U, 18432U}) {
                    const auto plan = attention_plan(n, keys, 2048, width, limit);
                    const Bytes q = plan.queries, k = plan.keys;
                    const Bytes qs = (plan.full_heads ? n : q) * 2048,
                                ks = (plan.full_heads ? keys : k) * 2048;
                    check(
                        aligned(
                            {qs * width,
                             ks * width,
                             ks * width,
                             q * 2048 * width,
                             q * k * width,
                             q * width,
                             q * width,
                             8 * 1024 * 1024}
                        ) <= plan.bytes
                        && plan.bytes <= limit
                    );
                    std::size_t queries_seen = 0, keys_seen = 0;
                    for (std::size_t start = 0; start < n; start += q)
                        queries_seen += std::min<std::size_t>(q, n - start);
                    for (std::size_t start = 0; start < keys; start += k)
                        keys_seen += std::min<std::size_t>(k, keys - start);
                    check(queries_seen == n && keys_seen == keys);
                }
    }
    check(projection_status_bytes == aligned({sizeof(unsigned)}));
    // Stable prefix and admitted high-water size, including retained weights and
    // short row/column tails; no projection, device or model execution.
    const auto tail_plan = projection_plan(24, 96769, 2, false, 1, 16ULL * 1024 * 1024);
    const std::size_t row_count = tail_plan.rows + 1;
    const auto input_rows =
        projection_input_rows(row_count, 24, tail_plan, 16ULL * 1024 * 1024);
    const auto reserved = tail_plan.bytes + (input_rows - tail_plan.rows) * 24 * 4;
    for (bool retained : {false, true}) {
        std::size_t columns_seen = 0;
        for (std::size_t column = 0; column < 96769; column += tail_plan.columns) {
            const auto out = std::min(tail_plan.columns, 96769 - column);
            columns_seen += out;
            const auto layout = aligned(
                {sizeof(unsigned),
                 input_rows * 24 * 4,
                 retained ? 0 : out * 24 * 2,
                 out * 4,
                 out * 4,
                 tail_plan.rows * 24 * 4,
                 tail_plan.rows * 24,
                 tail_plan.rows * 4,
                 tail_plan.rows * out * 4,
                 tail_plan.rows * out * 4,
                 8 * 1024 * 1024}
            );
            check(layout <= reserved);
            std::size_t rows_seen = 0;
            for (std::size_t row = 0; row < row_count; row += tail_plan.rows)
                rows_seen += std::min(tail_plan.rows, row_count - row);
            check(rows_seen == row_count);
        }
        check(columns_seen == 96769 && tail_plan.columns < 96769);
    }
    // Packed GGUF retains compressed blocks but always admits a decoded tile.
    for (auto encoding :
         {gguf::Encoding::q3_k, gguf::Encoding::q4_k, gguf::Encoding::q8_0}) {
        const auto block = gguf::block_values(encoding);
        const auto width = gguf::block_bytes(encoding);
        check(gguf_projection_row_bytes(block * 3, encoding) == width * 3);
        for (bool precise : {false, true}) {
            constexpr Bytes limit = 64ULL * 1024 * 1024;
            constexpr std::size_t in = 8192;
            const auto plan = gguf_projection_plan(in, 96768, encoding, precise, limit);
            check(plan.bytes <= limit);
            check(plan.columns * (in / block) * width <= 16ULL * 1024 * 1024);
            const Bytes t = plan.rows, o = plan.columns, w = in * o;
            const auto bytes = aligned(
                {sizeof(unsigned),
                 t * in * 4,
                 0,
                 sizeof(unsigned),
                 16 * 1024 * 1024,
                 w * 4,
                 o * 4,
                 o * 4,
                 t * in * 4,
                 t * in,
                 t * 4,
                 t * o * 4,
                 t * o * 4,
                 8 * 1024 * 1024,
                 precise ? w * 8 : 0,
                 precise ? t * in * 8 : 0,
                 precise ? t * o * 8 : 0}
            );
            check(bytes <= plan.bytes);
            const auto rows = plan.rows + 5;
            const auto needed = plan.bytes + 5 * in * sizeof(float);
            check(projection_input_rows(rows, in, plan, needed) == rows);
            check(projection_input_rows(rows, in, plan, needed - 1) == plan.rows);
        }
        fails([&] { gguf_projection_plan(0, 1, encoding, false, SIZE_MAX); });
        fails([&] { gguf_projection_plan(block - 1, 1, encoding, false, SIZE_MAX); });
        fails([&] { gguf_projection_plan(SIZE_MAX, 1, encoding, true, SIZE_MAX); });
        // Valid block alignment cannot admit a packed row above 16 MiB.
        const auto too_wide = (16 * 1024 * 1024 / width + 1) * block;
        fails([&] { gguf_projection_plan(too_wide, 1, encoding, false, SIZE_MAX); });
        fails([&] { gguf_projection_plan(block, 0, encoding, false, SIZE_MAX); });
        fails([&] {
            gguf_projection_plan(block, SIZE_MAX, encoding, false, SIZE_MAX);
        });
        fails([&] {
            gguf_projection_plan(
                block, 1, encoding, false, gguf_projection_reserve_bytes
            );
        });
        fails([&] {
            gguf_projection_plan(block, 1, encoding, false, 24 * 1024 * 1024);
        });
    }
    fails([] {
        // Deliberately malformed caller value exercises preflight rejection.
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        gguf_projection_plan(256, 1, static_cast<gguf::Encoding>(99), false, SIZE_MAX);
    });
    // Smallest admitted plan: the aligned scalar still fits at the exact limit.
    const auto minimum = projection_plan(1, 1, 4, false, 1, 8 * 1024 * 1024 + 8192);
    check(projection_plan(1, 1, 4, false, 1, minimum.bytes).bytes == minimum.bytes);
    fails([] { projection_plan(SIZE_MAX, 1, 4, false, 1, SIZE_MAX); });
    fails([] { projection_plan(1, SIZE_MAX, 4, false, 1, SIZE_MAX); });
    const auto normal =
        projection_plan(4096, 4096, 4, false, 1, 2ULL * 1024 * 1024 * 1024);
    check(normal.rows == 256 && normal.columns == 4096);
    // Long image/video inputs fit without duplicating their upload per column chunk.
    constexpr Bytes scratch = 2ULL * 1024 * 1024 * 1024;
    const auto projection = projection_plan(12288, 341, 4, false, 1, scratch);
    check(projection_input_rows(16394, 12288, projection, scratch) == 16394);
    const auto needed = projection.bytes + (16394 - projection.rows) * 12288 * 4;
    check(projection_input_rows(16394, 12288, projection, needed) == 16394);
    check(
        projection_input_rows(16394, 12288, projection, needed - 1) == projection.rows
    );
    check(projection_input_rows(1, 12288, projection, scratch) == projection.rows);
    check(
        projection_input_rows(SIZE_MAX, 12288, projection, scratch) == projection.rows
    );
    fails([&] { projection_input_rows(0, 12288, projection, scratch); });
    fails([&] { projection_input_rows(1, 0, projection, scratch); });
    fails([&] { projection_input_rows(1, 12288, projection, projection.bytes - 1); });
    const auto small = projection_plan(4096, 4096, 4, false, 1, 16ULL * 1024 * 1024);
    check(small.bytes <= 16ULL * 1024 * 1024 && small.columns < 4096);
    check(attention_plan(18432, 18432, 1024, 4, 2ULL * 1024 * 1024 * 1024).full_heads);
    check(!attention_plan(18432, 18432, 1024, 4, 16ULL * 1024 * 1024).full_heads);
    fails([] { attention_plan(1, 1, 1, 4, 8 * 1024 * 1024); });
    fails([] { projection_plan(4096, 4096, 4, false, 1, 8 * 1024 * 1024); });
    unsigned attempts = 0, pauses = 0;
    bool cancellation = false;
    wait_for_device_allocation(
        [&] { return cancellation; }, [&] { return ++attempts == 3; }, [&] { ++pauses; }
    );
    check(attempts == 3 && pauses == 2);
    attempts = 0;
    pauses = 0;
    bool interrupted = false;
    try {
        wait_for_device_allocation(
            [&] { return cancellation; },
            [&] {
                ++attempts;
                return false;
            },
            [&] {
                ++pauses;
                cancellation = true;
            }
        );
    } catch (const std::runtime_error&) {
        interrupted = true;
    }
    check(interrupted && attempts == 1 && pauses == 1);
    cancellation = false;
    interrupted = false;
    try {
        wait_for_device_allocation(
            [] { return false; },
            [] {
                throw std::runtime_error("device_fault");
                return false;
            },
            [&] { ++pauses; }
        );
    } catch (const std::runtime_error&) {
        interrupted = true;
    }
    check(interrupted && pauses == 1);
    auto cpu = compute_plan(1024, "", "", false);
    check(cpu.capacity == Footprint{1024} && cpu.devices.empty());
    const std::string budget =
        std::to_string(2ULL * 1024 * 1024 * 1024 + compute_context_bytes);
    auto dual = compute_plan(1024, "0,1", budget, true);
    check(
        dual.devices == std::vector<int>({0, 1})
        && dual.capacity == Footprint({1024, std::stoull(budget), std::stoull(budget)})
    );
    const auto large = std::to_string(8ULL * 1024 * 1024 * 1024);
    auto unequal =
        compute_plan(1024, "1,0", budget, true, "0:" + budget + ",1:" + large);
    check(
        unequal.capacity[1] == std::stoull(budget)
        && unequal.capacity[2] == std::stoull(large)
    );
    for (auto mapping :
         {"0:4294967296",
          "0:4294967296,0:4294967296",
          "0:4294967296,1:4294967296,",
          "2:4294967296,0:4294967296",
          "0:1,1:4294967296"})
        fails([&] { compute_plan(1024, "0,1", budget, true, mapping); });
    fails([&] { compute_plan(1024, "0", budget, true, "0:4294967296,1:4294967296"); });
    const std::array<Bytes, 2> capacities{1, 3}, reverse_capacities{3, 1}, zero{0, 0};
    check(
        weight_boundary(100, capacities, 1) == 25
        && weight_boundary(100, reverse_capacities, 1) == 75
    );
    check(
        weight_boundary(100, zero, 1) == 50
        && weight_boundary(100, capacities, 2) == 100
    );
    for (std::size_t n = 0; n < 100; ++n) {
        auto mid = weight_boundary(n, capacities, 1);
        check(mid <= n && mid * 4 + (n - mid) * 4 == n * 4);
    }
    auto reverse = compute_plan(1024, "1,0", budget, true);
    check(reverse.devices == std::vector<int>({1, 0}));
    auto second = compute_plan(1024, "1", budget, true);
    check(second.capacity[1] == 0 && second.capacity[2] == std::stoull(budget));
    for (auto devices : {"0,0", "2", "-1", "0,1,2", "cpu", "0,"})
        fails([&] { compute_plan(1024, devices, budget, true); });
    for (auto bytes :
         {"",
          "0",
          "268435456",
          "-1",
          "99999999999999999999999999",
          "30000000000",
          "536870912x"})
        fails([&] { compute_plan(1024, "0,1", bytes, true); });
    fails([&] { compute_plan(1024, "0,1", budget, false); });
    fails([&] { compute_plan(0, "", "", true); });
    WorkerCompute owned(cpu, Workload::image);
    auto h = owned.resources->reserve(
        Workload::image, host_footprint(*owned.resources, 1024)
    );
    fails([&] { compute_plan(1024, "1", "", true); });
    owned.resources->released(h);
    owned.idle();
    check(owned.resources->snapshot().used == Footprint{0});
    WorkerCompute parked(ComputePlan{{1024, 1024}, {}}, Workload::tts);
    auto context = std::make_shared<FakeContext>();
    parked.compute = context;
    parked.context = {0, 512};
    parked.resume();
    check(parked.resources->snapshot().used == Footprint({0, 512}));
    context->resources = parked.resources;
    context->scratch = parked.resources->reserve(Workload::tts, {0, 128});
    context->scratch_fail = true;
    bool scratch_failed = false;
    try {
        parked.park();
    } catch (const std::runtime_error&) {
        scratch_failed = true;
    }
    check(
        scratch_failed && context->releases == 0 && parked.context_handle
        && parked.resources->snapshot().used[1] == 640
    );
    context->scratch_fail = false;
    parked.idle();
    check(!context->scratch && parked.resources->snapshot().used[1] == 512);
    context->fail = true;
    bool failed = false;
    try {
        parked.park();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    check(
        failed && parked.context_handle && parked.resources->snapshot().used[1] == 512
    );
    context->fail = false;
    parked.park();
    check(!parked.context_handle && parked.resources->snapshot().used[1] == 0);
    parked.resume();
    auto active = parked.resources->reserve(Workload::tts, {0, 128});
    const auto calls = context->releases;
    failed = false;
    try {
        parked.park();
    } catch (const std::runtime_error&) {
        failed = true;
    }
    check(
        failed && context->releases == calls
        && parked.resources->snapshot().used[1] == 640
    );
    // Explicit image-cache reservations survive idle, but cannot disappear from
    // accounting or permit park before their physical retirement.
    parked.idle({0, 128});
    for (const Footprint& incorrect : {Footprint{0, 127}, Footprint{0, 129}}) {
        bool rejected = false;
        try {
            parked.idle(incorrect);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected && parked.resources->snapshot().used[1] == 640);
    }
    fails([&] { parked.idle({128}); });
    parked.resources->released(active);
    parked.park();
    check(parked.resources->snapshot().residents == 0);
    std::cout << "CPU device-selection, budget and fail-closed contracts passed\n";
}
