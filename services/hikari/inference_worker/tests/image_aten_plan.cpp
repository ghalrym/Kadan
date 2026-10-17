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

// Walk only integer dependency intervals, not tensor values or model forwards.
// A stride-one 3x3 convolution extends both ends by one; nearest upsampling
// maps each output index to floor(index / 2).
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
                auto first = pixel - 1, last = pixel + 1;  // conv_out
                for (int block = 4; block >= 0; --block) {
                    if (block < 4) {  // upsampler convolution, then nearest upsample
                        first = floor_half(first - 1);
                        last = floor_half(last + 1);
                    }
                    first -= 6;  // three residuals, each with two convolutions
                    last += 6;
                }
                check(std::max<std::int64_t>(0, first) >= decoder.input_begin);
                check(std::min<std::int64_t>(extent - 1, last) < decoder.input_end);
            }
            for (auto cell = encoder.begin; cell < encoder.end; ++cell) {
                auto first = cell, last = cell;
                for (int block = 4; block >= 0; --block) {
                    if (block < 4) {  // right-padded stride-two convolution
                        first *= 2;
                        last = last * 2 + 2;
                    }
                    first -= 4;  // two residuals
                    last += 4;
                }
                --first;  // conv_in
                ++last;
                check(std::max<std::int64_t>(0, first) >= encoder.input_begin * 16);
                check(
                    std::min<std::int64_t>(extent * 16 - 1, last)
                    < encoder.input_end * 16
                );
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

void vae_budget_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    for (const auto gpu : {8 * gib, 16 * gib}) {
        Resources ledger({32 * gib, gpu});
        const auto baseline =
            ledger.reserve(Workload::image, {256 * 1024 * 1024, gib / 2});
        for (const auto points : {16384, 18432}) {
            for (const auto stage :
                 {AtenImageStage::vae, AtenImageStage::vae_encoder}) {
                // A conservative 4 GiB component, largest 1152x1152x3x3 FP32
                // kernel. Actual checkpoint inventory supplies these at runtime.
                const auto plan = aten_stage_plan(
                    ledger.snapshot(),
                    {0},
                    4 * gib,
                    1152ULL * 1152 * 9 * 4,
                    aten_stage_workspace(stage, points, 0),
                    aten_tiled_vae_workspace(stage, points)
                );
                check(plan.device == 0 && plan.device_bytes <= gpu - gib / 2);
                if (stage == AtenImageStage::vae || gpu == 8 * gib)
                    check(plan.tile_vae);
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
    bool rejected = false;
    try {
        (void)aten_stage_plan(
            Snapshot{{32 * gib, 4 * gib}, {0, 0}, 0},
            {0},
            4 * gib,
            1152ULL * 1152 * 9 * 4,
            aten_stage_workspace(AtenImageStage::vae, 18432, 0),
            aten_tiled_vae_workspace(AtenImageStage::vae, 18432)
        );
    } catch (const ResourceExhausted&) {
        rejected = true;
    }
    check(rejected);
}
}  // namespace

int main() {
    try {
        tile_geometry_units();
        vae_budget_units();
        using namespace kadan;
        using namespace kadan::image;
        constexpr Bytes gib = 1024ULL * 1024 * 1024;
        Snapshot snapshot{{32 * gib, 24 * gib, 24 * gib}, {0, 8 * gib, 2 * gib}, 0};
        check(aten_stage_device(snapshot, {0, 1}, 20 * gib) == 1);
        check(aten_stage_device(snapshot, {1, 0}, 20 * gib) == 1);
        snapshot.used[1] = gib;
        check(aten_stage_device(snapshot, {1, 0}, 20 * gib) == 0);
        bool rejected = false;
        try {
            (void)aten_stage_device(snapshot, {1}, 23 * gib);
        } catch (const ResourceExhausted&) {
            rejected = true;
        }
        check(rejected);
        rejected = false;
        try {
            (void)aten_stage_device(snapshot, {2}, gib);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected);
        // The same planner drives prepare() and Stage's actual residency.
        // Lower tiers stream text/denoiser weights rather than admitting only
        // a smaller number while still loading the full checkpoint.
        for (const auto gpu : {8 * gib, 16 * gib}) {
            Snapshot tier{{32 * gib, gpu}, {256 * 1024 * 1024, gib / 2}, 0};
            const auto text_plan = aten_stage_plan(
                tier,
                {0},
                15 * gib,
                151936ULL * 4096 * 2,
                aten_stage_workspace(AtenImageStage::text, 16384, 2048)
            );
            check(text_plan.stream_weights && text_plan.device == 0);
            check(text_plan.device_bytes <= gpu - gib / 2);
            const auto denoiser_plan = aten_stage_plan(
                tier,
                {0},
                14 * gib,
                16384ULL * 4096 * 2,
                aten_stage_workspace(AtenImageStage::denoiser, 18432, 2034)
            );
            check(denoiser_plan.stream_weights && denoiser_plan.device == 0);
            check(denoiser_plan.device_bytes <= gpu - gib / 2);
        }
        const auto resident = aten_stage_plan(
            Snapshot{{32 * gib, 24 * gib}, {0, 0}, 0}, {0}, 15 * gib, gib, 3 * gib
        );
        check(!resident.stream_weights && resident.device_bytes == 18 * gib);
        rejected = false;
        try {
            (void)aten_stage_plan(
                Snapshot{{gib, 24 * gib}, {0, 0}, 0}, {0}, 15 * gib, gib, 3 * gib
            );
        } catch (const ResourceExhausted&) {
            rejected = true;
        }
        check(rejected);
        const auto text = aten_stage_workspace(AtenImageStage::text, 16384, 2048);
        const auto denoiser =
            aten_stage_workspace(AtenImageStage::denoiser, 16384, 2034);
        const auto vae = aten_stage_workspace(AtenImageStage::vae, 16384, 2034);
        check(text > 2 * gib && 17 * gib + text < 24 * gib);
        check(denoiser > 4 * gib && 14 * gib + denoiser < 24 * gib);
        check(vae == 20 * gib && 2 * gib + vae < 24 * gib);
        // Regression: the observed 7.74 GiB live + 2.31 GiB cached + next
        // 4.50 GiB convolution exceeded the old 14.22 GiB weight+stage cap.
        check(gib + vae > 15 * gib);
        snapshot.used = {0, 8 * gib, 2 * gib};
        check(aten_stage_device(snapshot, {0, 1}, gib + vae) == 1);
        rejected = false;
        try {
            (void)aten_stage_device(snapshot, {0}, gib + vae);
        } catch (const ResourceExhausted&) {
            rejected = true;
        }
        check(rejected);
        check(aten_stage_workspace(AtenImageStage::vae, 18432, 0) > vae);
        const auto encoder =
            aten_stage_workspace(AtenImageStage::vae_encoder, 16384, 0);
        check(encoder == 11 * gib);
        const auto edited =
            aten_stage_workspace(AtenImageStage::denoiser, 16384, 2034, true);
        check(edited > denoiser);
        // The extra source/noise/mask retention is admitted before editing starts.
        Snapshot constrained{{32 * gib, 14 * gib + denoiser}, {0, 0}, 0};
        check(aten_stage_device(constrained, {0}, 14 * gib + denoiser) == 0);
        rejected = false;
        try {
            (void)aten_stage_device(constrained, {0}, 14 * gib + edited);
        } catch (const ResourceExhausted&) {
            rejected = true;
        }
        check(rejected);
        std::atomic_bool cancelled = false;
        std::array<float, 8> first{}, repeated{}, next{};
        seed_image_latent(first, 1138803809, cancelled);
        seed_image_latent(repeated, 1138803809, cancelled);
        seed_image_latent(next, 1138803810, cancelled);
        check(first == repeated && first != next);
        std::array<float, 3> odd{};
        seed_image_latent(odd, 1138803809, cancelled);
        check(std::equal(odd.begin(), odd.end(), first.begin()));
        cancelled = true;
        rejected = false;
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
