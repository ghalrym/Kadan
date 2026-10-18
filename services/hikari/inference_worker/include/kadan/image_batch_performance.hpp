#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <span>

#include "kadan/resources.hpp"

namespace kadan::image {
struct ImageBatchCost {
    // Current uploads plus restoring component banks displaced by this plan.
    double loading_seconds = 0;
    std::optional<double> denoise_seconds;
    bool streamed_weights = false;
};

struct ImageBatchPlan {
    Footprint memory;
    ImageBatchCost cost;
};

// One geometry/steps/reference shape, owned and retired with the image model.
// Loading is removed from observations and priced from today's cache instead:
// a cold start or a costly replica probe must not inflate the warm baseline.
// Operator streaming stays in execution time: it repeats during denoising, unlike
// a retained upload. Keep its observations separate from resident execution.
class ImageBatchPerformance {
    struct Sample {
        double compute_seconds, denoise_seconds;
    };

    std::array<std::map<std::size_t, Sample>, 2> samples_;

public:
    void observe(
        std::size_t images,
        double seconds,
        double loading_seconds,
        double denoise_seconds,
        bool streamed_weights = false
    ) {
        if (!images || images > 32 || !std::isfinite(seconds)
            || !std::isfinite(loading_seconds) || !std::isfinite(denoise_seconds)
            || loading_seconds < 0 || denoise_seconds <= 0 || seconds <= loading_seconds
            || denoise_seconds > seconds - loading_seconds)
            return;
        samples_[streamed_weights][images] = {
            seconds - loading_seconds, denoise_seconds
        };
    }

    std::size_t capacity(std::span<const ImageBatchPlan> plans) const {
        const auto limit = std::min<std::size_t>(32, plans.size());
        std::size_t best = 1;
        double best_rate = std::numeric_limits<double>::infinity();
        for (std::size_t images = 1; images <= limit; ++images) {
            const auto& cost = plans[images - 1].cost;
            if (!std::isfinite(cost.loading_seconds) || cost.loading_seconds < 0
                || (cost.denoise_seconds
                    && (!std::isfinite(*cost.denoise_seconds)
                        || *cost.denoise_seconds <= 0)))
                continue;
            const auto& samples = samples_[cost.streamed_weights];
            if (samples.empty() || images > samples.rbegin()->first + 1)
                continue;
            auto sample = samples.upper_bound(images);
            if (sample != samples.begin())
                --sample;
            const auto denoise_rate =
                cost.denoise_seconds
                    ? *cost.denoise_seconds / double(images)
                    : sample->second.denoise_seconds / double(sample->first);
            const auto other_rate =
                (sample->second.compute_seconds - sample->second.denoise_seconds)
                / double(sample->first);
            const auto rate =
                cost.loading_seconds / double(images) + denoise_rate + other_rate;
            // Ties permit one bounded growth probe; successful execution is the
            // only operation that earns another size. Measured regressions fall
            // back to the faster size rather than perpetually increasing count.
            if (std::nextafter(rate, 0.0) <= best_rate) {
                best = images;
                best_rate = rate;
            }
        }
        return best;
    }
};
}  // namespace kadan::image
