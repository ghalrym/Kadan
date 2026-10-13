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
}  // namespace

int main() {
    try {
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
        const auto text = aten_stage_workspace(AtenImageStage::text, 16384, 2048);
        const auto denoiser =
            aten_stage_workspace(AtenImageStage::denoiser, 16384, 2034);
        const auto vae = aten_stage_workspace(AtenImageStage::vae, 16384, 2034);
        check(text > 2 * gib && 17 * gib + text < 24 * gib);
        check(denoiser > 4 * gib && 14 * gib + denoiser < 24 * gib);
        check(vae > 12 * gib && 2 * gib + vae < 24 * gib);
        check(aten_stage_workspace(AtenImageStage::vae, 18432, 0) > vae);
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
