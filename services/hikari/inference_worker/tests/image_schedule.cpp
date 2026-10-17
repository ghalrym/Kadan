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

    // Published API aspect sizes plus the generator's admitted pixel boundary.
    for (auto canvas : std::array{
             Canvas{2048, 2048},
             Canvas{2400, 1792},
             Canvas{1792, 2400},
             Canvas{2752, 1536},
             Canvas{3072, 1536},
         }) {
        check(canvas.width * canvas.height <= 4608 * 1024);
        const auto points = canvas.width / 16 * (canvas.height / 16);
        for (std::size_t steps : {2u, 3u, 4u, 40u, 100u}) {
            const auto schedule = kadan::image::schedule(points, steps);
            check(schedule.steps == steps && schedule.sigma[0] == 1.f);
            check(std::abs(schedule.sigma[steps - 1] - .02f) < 1e-6f);
            check(schedule.sigma[steps] == 0.f);
            for (std::size_t i = 1; i <= steps; ++i)
                check(
                    std::isfinite(schedule.sigma[i])
                    && schedule.sigma[i] < schedule.sigma[i - 1]
                );
        }
    }
}

void square_schedule_values() {
    // Checkpoint parameters: exponential shift mu=1.3129032258064515,
    // terminal .02. Golden values were evaluated in double precision offline.
    const auto schedule = kadan::image::schedule(16384, 40);
    check(std::abs(schedule.sigma[1] - .9926459799f) < 1e-6f);
    check(std::abs(schedule.sigma[19] - .7898675096f) < 1e-6f);
    check(std::abs(schedule.sigma[38] - .1022295822f) < 1e-6f);
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
    square_schedule_values();
    rejected_bounds();
    std::cout << "image schedule scalar units passed (no inference)\n";
}
