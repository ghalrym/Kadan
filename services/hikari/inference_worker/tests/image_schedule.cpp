// Scalar schedule and canvas arithmetic only; no model or inference.
#include "kadan/image_schedule.hpp"

#include <array>
#include <iostream>
#include <source_location>
#include <string>
#include <string_view>

namespace {
void check(bool ok, std::source_location at = std::source_location::current()) {
    if (!ok)
        throw std::runtime_error("image_schedule_test:" + std::to_string(at.line()));
}

void supported_canvases() {
    struct Canvas {
        std::size_t width, height;
    };

    for (auto canvas : std::array{
             Canvas{1024, 1024},
             Canvas{1152, 864},
             Canvas{864, 1152},
             Canvas{1344, 768},
             Canvas{1536, 512},
             Canvas{768, 832},
             Canvas{3072, 1536},
         }) {
        check(canvas.width * canvas.height <= 4608 * 1024);
        const auto points = canvas.width / 16 * (canvas.height / 16);
        for (std::size_t steps : {2u, 4u, 8u, 40u, 100u}) {
            const auto schedule = kadan::image::schedule(points, steps);
            check(schedule.steps == steps && schedule.sigma[0] == 1.f);
            check(schedule.sigma[steps] == 0.f);
            for (std::size_t i = 1; i <= steps; ++i)
                check(
                    std::isfinite(schedule.sigma[i])
                    && schedule.sigma[i] < schedule.sigma[i - 1]
                );
        }
    }
}

void klein_four_step_golden_values() {
    // Verified directly against Diffusers FlowMatchEulerDiscreteScheduler + compute_empirical_mu.
    {
        const auto s = kadan::image::schedule(1024, 4);
        check(std::abs(s.sigma[0] - 1.0f) < 1e-6f);
        check(std::abs(s.sigma[1] - 0.95808536f) < 1e-5f);
        check(std::abs(s.sigma[2] - 0.88398188f) < 1e-5f);
        check(std::abs(s.sigma[3] - 0.71749657f) < 1e-5f);
        check(s.sigma[4] == 0.0f);
    }
    {
        const auto s = kadan::image::schedule(3072, 4);
        check(std::abs(s.sigma[1] - 0.96453035f) < 1e-5f);
        check(std::abs(s.sigma[2] - 0.90063941f) < 1e-5f);
        check(std::abs(s.sigma[3] - 0.75133359f) < 1e-5f);
    }
    {
        const auto s = kadan::image::schedule(4096, 4);
        check(std::abs(s.sigma[1] - 0.96738404f) < 1e-5f);
        check(std::abs(s.sigma[2] - 0.90814394f) < 1e-5f);
        check(std::abs(s.sigma[3] - 0.76719993f) < 1e-5f);
    }
}

void rejected_bounds() {
    for (const auto& values : std::array{
             std::array<std::size_t, 2>{0, 2},
             std::array<std::size_t, 2>{18433, 2},
             std::array<std::size_t, 2>{1, 1},
             std::array<std::size_t, 2>{1, 101},
         }) {
        bool rejected = false;
        try {
            (void)kadan::image::schedule(values[0], values[1]);
        } catch (const std::invalid_argument& e) {
            rejected = std::string_view(e.what()) == "image_schedule_bounds";
        }
        check(rejected);
    }
    check(kadan::image::schedule(1, 2).steps == 2);
}
}  // namespace

int main() {
    supported_canvases();
    klein_four_step_golden_values();
    rejected_bounds();
    std::cout << "image schedule scalar units passed (no inference)\n";
}
