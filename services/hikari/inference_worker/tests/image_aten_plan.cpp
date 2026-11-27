// Admission/placement arithmetic only; no ATen tensors, CUDA calls or inference.
#include "kadan/image_aten_plan.hpp"

#include <iostream>
#include <source_location>

#include "kadan/image_input.hpp"

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("image_aten_plan_test:" + std::to_string(at.line()));
}

std::int64_t floor_half(std::int64_t value) {
    return value >= 0 ? value / 2 : (value - 1) / 2;
}

void tile_geometry_units() {
    using namespace kadan::image;
    for (const auto extent : {2, 16, 18, 96, 128, 192}) {
        std::int64_t covered = 0;
        for (std::int64_t begin = 0; begin < extent; begin += vae_tile_core) {
            const auto decoder = vae_tile_axis(begin, extent, vae_decode_halo);
            const auto encoder = vae_tile_axis(begin, extent, vae_encode_halo);
            check(decoder.begin == covered && decoder.end == encoder.end);
            check(
                decoder.input_end - decoder.input_begin
                <= vae_tile_core + 2 * vae_decode_halo
            );
            check(decoder.input_begin + decoder.crop_begin() == decoder.begin);
            for (auto pixel = decoder.begin * 16; pixel < decoder.end * 16; ++pixel) {
                auto first = pixel - 1, last = pixel + 1;
                for (int block = 4; block >= 0; --block) {
                    if (block < 4) {
                        first = floor_half(first - 1);
                        last = floor_half(last + 1);
                    }
                    first -= 6;
                    last += 6;
                }
                check(std::max<std::int64_t>(0, first) >= decoder.input_begin);
                check(std::min<std::int64_t>(extent - 1, last) < decoder.input_end);
            }
            covered = decoder.end;
        }
        check(covered == extent);
    }
    bool rejected = false;
    try {
        (void)vae_tile_axis(16, 16, vae_decode_halo);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected);
}

void reference_budget_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    // Strip generation (1536x512 = 3072 cells) with 2 references (3072 + 2496 = 5568 cells).
    constexpr Bytes points = 96 * 32, ref_points = 96 * 32 + 48 * 52, text_tokens = 512;
    const auto plain =
        aten_stage_workspace(AtenImageStage::denoiser, points, text_tokens);
    const auto reference = aten_stage_workspace(
        AtenImageStage::denoiser, points, text_tokens, false, ref_points
    );
    check(reference > plain);
    Snapshot constrained{{32 * gib, plain}, {0, 0}, 0};
    check(aten_stage_device(constrained, {0}, plain) == 0);
    bool rejected = false;
    try {
        (void)aten_stage_device(constrained, {0}, reference);
    } catch (const ResourceExhausted&) {
        rejected = true;
    }
    check(rejected);
    const auto host = aten_reference_host_bytes(ref_points, text_tokens);
    check(host == ref_points * 128 * 4 + text_tokens * 7680 * 4);
    Resources ledger({host, 8 * gib});
    const auto retained = ledger.reserve(Workload::image, {host, 0});
    check(ledger.snapshot().used[0] == host);
    ledger.released(retained);
    check(ledger.snapshot().used == Footprint({0, 0}));
}

void batch_budget_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    const auto one = aten_batch_workspace(3072, 512, 5568, 1);
    const auto two = aten_batch_workspace(3072, 512, 5568, 2);
    check(
        one == aten_stage_workspace(AtenImageStage::denoiser, 3072, 512, false, 5568)
    );
    check(two - one == one - 2 * gib);
    // FLUX.2 Klein 4B denoiser weights (~7.75 GiB) fit comfortably in 24 GiB VRAM.
    const Snapshot card24{{64 * gib, 24 * gib}, {0, 0}, 0};
    const auto strip_batch = aten_stage_plan(
        card24, {0}, 8 * gib, gib / 8, aten_batch_workspace(3072, 512, 5568, 4)
    );
    check(!strip_batch.stream_weights && strip_batch.device_bytes <= 24 * gib);

    bool rejected = false;
    try {
        (void)aten_batch_workspace(1024, 512, 1024, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected);
}

void denoiser_lane_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    constexpr Bytes weights = 8 * gib, largest = gib / 8;
    constexpr Bytes host = 2 * largest + 256ULL * 1024 * 1024;
    const auto workspace = [](std::size_t images) {
        return aten_batch_workspace(3072, 512, 5568, images);
    };
    const auto plan = [&](const Snapshot& snapshot, std::size_t images) {
        return aten_denoiser_plan(
            snapshot, {0, 1}, weights, largest, 3072, 512, 5568, images
        );
    };
    const Snapshot ample{{64 * gib, 24 * gib, 24 * gib}, {0, 0, 0}, 0};
    const auto balanced = plan(ample, 4);
    check(balanced.size() == 2);
    check(balanced[0].begin == 0 && balanced[0].images == 2);
    check(balanced[1].begin == 2 && balanced[1].images == 2);
    check(balanced[0].stage.device != balanced[1].stage.device);
    Resources ledger(ample.capacity);
    std::vector<Handle> handles;
    for (const auto& lane : balanced) {
        check(!lane.stage.stream_weights && !lane.stage.tile_vae);
        check(lane.stage.device_bytes == weights + workspace(2));
        Footprint bytes(3);
        bytes[0] = lane.stage.host_bytes;
        bytes[lane.stage.device + 1] = lane.stage.device_bytes;
        handles.push_back(ledger.reserve(Workload::image, bytes));
    }
    check(ledger.snapshot().used[0] == 2 * host);
    for (const auto handle : handles)
        ledger.released(handle);
    check(ledger.snapshot().used == Footprint({0, 0, 0}));

    const auto streamed = plan(Snapshot{{64 * gib, 8 * gib, 8 * gib}, {0, 0, 0}, 0}, 4);
    check(streamed.size() == 1 && streamed[0].stage.stream_weights);
    check(streamed[0].images == 4);
    const auto singleton = plan(ample, 1);
    check(singleton.size() == 1 && singleton[0].images == 1);
}

void vae_budget_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    for (const auto gpu : {8 * gib, 16 * gib}) {
        Resources ledger({32 * gib, gpu});
        const auto baseline =
            ledger.reserve(Workload::image, {256 * 1024 * 1024, gib / 2});
        for (const auto points : {4096, 16384}) {
            for (const auto stage :
                 {AtenImageStage::vae, AtenImageStage::vae_encoder}) {
                const auto plan = aten_stage_plan(
                    ledger.snapshot(),
                    {0},
                    512ULL * 1024 * 1024,
                    512ULL * 512 * 9 * 4,
                    aten_stage_workspace(stage, points, 0),
                    aten_tiled_vae_workspace(stage, points)
                );
                check(plan.device == 0 && plan.device_bytes <= gpu - gib / 2);
                const auto handle = ledger.reserve(
                    Workload::image, {plan.host_bytes, plan.device_bytes}
                );
                ledger.released(handle);
                check(
                    ledger.snapshot().used == Footprint({256 * 1024 * 1024, gib / 2})
                );
            }
        }
        ledger.released(baseline);
    }
}
}  // namespace

int main() {
    try {
        tile_geometry_units();
        vae_budget_units();
        reference_budget_units();
        batch_budget_units();
        denoiser_lane_units();

        using namespace kadan;
        using namespace kadan::image;
        constexpr Bytes gib = 1024ULL * 1024 * 1024;
        Snapshot snapshot{{32 * gib, 24 * gib, 24 * gib}, {0, 8 * gib, 2 * gib}, 0};
        check(aten_stage_device(snapshot, {0, 1}, 20 * gib) == 1);
        check(aten_stage_device(snapshot, {1, 0}, 20 * gib) == 1);
        snapshot.used[1] = gib;
        check(aten_stage_device(snapshot, {1, 0}, 20 * gib) == 0);

        const auto text = aten_stage_workspace(AtenImageStage::text, 4096, 512);
        const auto denoiser =
            aten_stage_workspace(AtenImageStage::denoiser, 4096, 512);
        const auto vae = aten_stage_workspace(AtenImageStage::vae, 4096, 512);
        check(text > 2 * gib && 8 * gib + text < 16 * gib);
        check(denoiser > 2 * gib && 8 * gib + denoiser < 16 * gib);
        check(vae > 2 * gib && gib + vae < 8 * gib);

        const auto edited =
            aten_stage_workspace(AtenImageStage::denoiser, 4096, 512, true);
        check(edited > denoiser);

        std::atomic_bool cancelled = false;
        std::array<float, 8> first{}, repeated{}, next{};
        seed_image_latent(first, 1138803809, cancelled);
        seed_image_latent(repeated, 1138803809, cancelled);
        seed_image_latent(next, 1138803810, cancelled);
        check(first == repeated && first != next);
        cancelled = true;
        bool rejected = false;
        try {
            seed_image_latent(first, 0, cancelled);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected);
        std::cout << "image ATen admission units passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
