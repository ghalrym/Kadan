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

void reference_budget_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    constexpr Bytes points = 24 * 26, vision_tokens = points / 4, text_tokens = 100;
    const auto vision = aten_stage_workspace(
        AtenImageStage::vision, points, 0, false, points, vision_tokens
    );
    check(vision > 2 * gib && vision < 3 * gib);
    const auto plan = aten_stage_plan(
        Snapshot{{32 * gib, 8 * gib}, {0, 0}, 0},
        {0},
        1152776672ULL,
        4608ULL * 4096 * 2,
        vision
    );
    check(!plan.stream_weights && plan.device_bytes < 8 * gib);
    const auto plain =
        aten_stage_workspace(AtenImageStage::denoiser, points, text_tokens);
    const auto reference = aten_stage_workspace(
        AtenImageStage::denoiser, points, text_tokens, false, points, vision_tokens
    );
    // 624 reference cells alone retain 312 MiB of K/V, beyond projection work.
    check(reference - plain > 312ULL * 1024 * 1024);
    check(reference - plain < 512ULL * 1024 * 1024);
    check(
        aten_stage_workspace(AtenImageStage::denoiser, points, text_tokens * 2) - plain
        >= text_tokens * 32 * 2 * 4096 * 2
    );
    Snapshot constrained{{32 * gib, plain}, {0, 0}, 0};
    check(aten_stage_device(constrained, {0}, plain) == 0);
    bool rejected = false;
    try {
        (void)aten_stage_device(constrained, {0}, reference);
    } catch (const ResourceExhausted&) {
        rejected = true;
    }
    check(rejected);
    const auto host =
        aten_reference_host_bytes(points, vision_tokens, text_tokens + vision_tokens);
    check(
        host
        == points * 64 * 4
               + (4 * vision_tokens + text_tokens + vision_tokens) * 4096 * 4
    );
    check(host < 16 * 1024 * 1024);
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
    const auto one = aten_batch_workspace(1024, 100, 1024, 1);
    const auto two = aten_batch_workspace(1024, 100, 1024, 2);
    check(
        one == aten_stage_workspace(AtenImageStage::denoiser, 1024, 100, false, 1024)
    );
    check(two - one == one - 2 * gib);
    check(aten_batch_workspace(1024, 200, 1024, 2) > two);
    // A short head can provisionally admit four images. A later long prompt
    // must trigger the full-batch preflight and reduce the batch before running.
    const Snapshot pose_device{{64 * gib, 24 * gib}, {0, 0}, 0};
    const auto pose_batch = aten_stage_plan(
        pose_device, {0}, 14 * gib, gib / 8, aten_batch_workspace(624, 250, 624, 4)
    );
    check(!pose_batch.stream_weights);
    bool longer_rejected = false;
    try {
        (void)aten_stage_plan(
            pose_device, {0}, 14 * gib, gib / 8, aten_batch_workspace(624, 8192, 624, 4)
        );
    } catch (const ResourceExhausted&) {
        longer_rejected = true;
    }
    check(longer_rejected);
    const auto reduced = aten_stage_plan(
        pose_device, {0}, 14 * gib, gib / 8, aten_batch_workspace(624, 8192, 624, 2)
    );
    check(reduced.device_bytes <= 24 * gib);
    // Two individually adequate devices cannot pool memory for one larger batch.
    Snapshot snapshot{{64 * gib, 8 * gib, 8 * gib}, {0, 0, 0}, 0};
    const auto fitting = aten_stage_plan(snapshot, {0, 1}, 4 * gib, gib / 2, two);
    check(fitting.device_bytes <= 8 * gib);
    bool rejected = false;
    try {
        (void)aten_stage_plan(
            snapshot, {0, 1}, 4 * gib, gib / 2, aten_batch_workspace(1024, 100, 1024, 8)
        );
    } catch (const ResourceExhausted&) {
        rejected = true;
    }
    check(rejected);
    rejected = false;
    try {
        (void)aten_batch_workspace(1024, 100, 1024, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected);
}

void denoiser_lane_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    constexpr Bytes weights = 14 * gib, largest = gib / 8;
    constexpr Bytes host = 2 * largest + 256ULL * 1024 * 1024;
    const auto workspace = [](std::size_t images) {
        return aten_batch_workspace(624, 250, 624, images);
    };
    const auto plan = [&](const Snapshot& snapshot, std::size_t images) {
        return aten_denoiser_plan(
            snapshot, {0, 1}, weights, largest, 624, 250, 624, images
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
    const auto uneven = plan(
        Snapshot{
            {64 * gib, weights + workspace(1), weights + workspace(4)}, {0, 0, 0}, 0
        },
        4
    );
    check(uneven.size() == 2);
    check(uneven[0].stage.device == 1 && uneven[0].images == 3);
    check(uneven[1].stage.device == 0 && uneven[1].images == 1);
    check(uneven[1].begin == 3);
    // Physical headroom on one device never increases the other's capacity.
    const auto pressured =
        plan(Snapshot{{64 * gib, 24 * gib, 24 * gib}, {0, 0, 20 * gib}, 0}, 4);
    check(pressured.size() == 1 && pressured[0].stage.device == 0);
    // Staging reservations coexist while replicas execute; admit their sum.
    const auto host_limited =
        plan(Snapshot{{2 * host, 24 * gib, 24 * gib}, {host, 0, 0}, 0}, 4);
    check(host_limited.size() == 1 && host_limited[0].images == 4);
    const auto streamed = plan(Snapshot{{64 * gib, 8 * gib, 8 * gib}, {0, 0, 0}, 0}, 4);
    check(streamed.size() == 1 && streamed[0].stage.stream_weights);
    check(streamed[0].images == 4);
    const auto singleton = plan(ample, 1);
    check(singleton.size() == 1 && singleton[0].images == 1);
    bool rejected = false;
    try {
        (void)plan(Snapshot{{64 * gib, 3 * gib, 3 * gib}, {0, 0, 0}, 0}, 4);
    } catch (const ResourceExhausted&) {
        rejected = true;
    }
    check(rejected);
    rejected = false;
    try {
        (void)aten_denoiser_plan(ample, {0, 0}, weights, largest, 624, 250, 624, 4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected);
}

void heterogeneous_denoiser_units() {
    using namespace kadan;
    using namespace kadan::image;
    constexpr Bytes gib = 1024ULL * 1024 * 1024;
    constexpr Bytes weights = 14 * gib, largest = gib / 8;
    constexpr Bytes host = 2 * largest + 256ULL * 1024 * 1024;
    const auto workspace = [](std::size_t images) {
        return aten_batch_workspace(624, 250, 624, images);
    };
    const auto plan = [&](const Snapshot& snapshot,
                          const std::vector<int>& devices,
                          std::size_t images) {
        return aten_denoiser_plan(
            snapshot, devices, weights, largest, 624, 250, 624, images
        );
    };
    // Sparse CUDA IDs and independent background usage must survive placement.
    Snapshot three{Footprint(7), Footprint(7), 0};
    three.capacity[0] = 64 * gib;
    for (const auto& [device, images] :
         {std::pair{1, 1}, std::pair{3, 2}, std::pair{5, 5}}) {
        three.used[device + 1] = device * gib;
        three.capacity[device + 1] =
            three.used[device + 1] + weights + workspace(images);
    }
    const auto uneven = plan(three, {3, 1, 5}, 6);
    check(uneven.size() == 3);
    check(uneven[0].stage.device == 5 && uneven[0].images == 3);
    check(uneven[1].stage.device == 3 && uneven[1].images == 2);
    check(uneven[2].stage.device == 1 && uneven[2].images == 1);
    check(uneven[0].begin == 0 && uneven[1].begin == 3 && uneven[2].begin == 5);
    for (const auto& lane : uneven)
        check(
            lane.stage.device_bytes
            <= three.capacity[lane.stage.device + 1] - three.used[lane.stage.device + 1]
        );

    Snapshot four{Footprint(9), Footprint(9), 0};
    four.capacity[0] = 64 * gib;
    const std::vector<int> devices{0, 2, 5, 7};
    for (const auto device : devices)
        four.capacity[device + 1] = weights + workspace(8);
    const auto balanced = plan(four, devices, 8);
    check(balanced.size() == 4);
    Resources ledger(four.capacity);
    std::vector<Handle> handles;
    for (std::size_t i = 0; i < balanced.size(); ++i) {
        const auto& lane = balanced[i];
        check(
            lane.stage.device == devices[i] && lane.images == 2 && lane.begin == 2 * i
        );
        Footprint bytes(9);
        bytes[0] = lane.stage.host_bytes;
        bytes[lane.stage.device + 1] = lane.stage.device_bytes;
        handles.push_back(ledger.reserve(Workload::image, bytes));
    }
    check(ledger.snapshot().used[0] == 4 * host);
    for (const auto handle : handles)
        ledger.released(handle);
    check(ledger.snapshot().used == Footprint(9));

    for (std::size_t i = 0; i < devices.size(); ++i)
        four.capacity[devices[i] + 1] = weights + workspace(std::size_t(1) << i);
    const auto four_uneven = plan(four, devices, 10);
    check(four_uneven.size() == 4);
    check(four_uneven[0].stage.device == 7 && four_uneven[0].images == 4);
    check(four_uneven[1].stage.device == 5 && four_uneven[1].images == 3);
    check(four_uneven[2].stage.device == 2 && four_uneven[2].images == 2);
    check(four_uneven[3].stage.device == 0 && four_uneven[3].images == 1);
    check(four_uneven[3].begin == 9);

    // Do not create empty lanes when more devices than images are available.
    const auto two_images = plan(four, devices, 2);
    check(two_images.size() == 2);
    check(two_images[0].images == 1 && two_images[1].images == 1);
    four.capacity[0] = 3 * host;
    four.used[0] = host;
    const auto host_limited = plan(four, devices, 8);
    check(host_limited.size() == 2);
    check(host_limited[0].images == 4 && host_limited[1].images == 4);
    four.used[0] = 2 * host;
    const auto one_lane = plan(four, devices, 8);
    check(one_lane.size() == 1 && one_lane[0].images == 8);

    // Four independent 8 GiB devices cannot pool their memory into a replica.
    four.capacity[0] = 64 * gib;
    four.used[0] = 0;
    for (const auto device : devices)
        four.capacity[device + 1] = 8 * gib;
    const auto streamed = plan(four, devices, 4);
    check(streamed.size() == 1 && streamed[0].stage.stream_weights);
    check(streamed[0].stage.device_bytes <= 8 * gib);
    // If even one streamed image batch cannot fit, aggregate free VRAM is irrelevant.
    for (const auto device : devices)
        four.capacity[device + 1] = 3 * gib;
    bool rejected = false;
    try {
        (void)plan(four, devices, 4);
    } catch (const ResourceExhausted&) {
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
        reference_budget_units();
        batch_budget_units();
        denoiser_lane_units();
        heterogeneous_denoiser_units();
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
