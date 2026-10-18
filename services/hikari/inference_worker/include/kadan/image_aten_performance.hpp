#pragma once
#include <cmath>
#include <optional>
#include <span>

#include "kadan/image_aten_plan.hpp"

namespace kadan::image {

// The owner clears this model-scoped history when changing checkpoints. Cached
// host and GPU weights change loading cost; keep their observation regimes separate.
struct AtenDenoiserWorkload {
    Bytes points;
    Bytes tokens;
    Bytes reference_points;
    std::size_t steps;
    bool weights_cached;
    bool gpu_weights_cached = false;

    bool operator==(const AtenDenoiserWorkload&) const = default;
};

struct AtenDenoiserMeasurement {
    int device;
    std::size_t images;
    bool replica;
    double load_seconds;
    double denoise_seconds;
};

// Native cache planning supplies current hits/misses and the measured cost of
// restoring other component banks that this candidate would evict. An empty
// override vector uses historical loading; otherwise provide one entry per lane.
struct AtenDenoiserCost {
    std::vector<std::optional<double>> load_seconds;
    double deferred_reload_seconds = 0;
};

class AtenDenoiserPerformance {
    struct Record {
        AtenDenoiserWorkload workload;
        std::vector<AtenDenoiserMeasurement> single;
        std::vector<AtenDenoiserMeasurement> parallel;
    };

    static constexpr std::size_t history_limit = 64;
    std::vector<Record> history_;

    static bool valid_cost(const AtenDenoiserCost& current, std::size_t lanes) {
        const auto valid = [](double seconds) {
            return std::isfinite(seconds) && seconds >= 0;
        };
        return valid(current.deferred_reload_seconds)
               && (current.load_seconds.empty() || current.load_seconds.size() == lanes)
               && std::all_of(
                   current.load_seconds.begin(),
                   current.load_seconds.end(),
                   [&](const auto& seconds) { return !seconds || valid(*seconds); }
               );
    }

    static std::optional<double> cost(
        const Record& record,
        std::span<const AtenDenoiserLane> lanes,
        const AtenDenoiserCost& current
    ) {
        double loading = current.deferred_reload_seconds, denoising = 0;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            const auto& lane = lanes[i];
            const auto load =
                current.load_seconds.empty() ? std::nullopt : current.load_seconds[i];
            const AtenDenoiserMeasurement* sample = nullptr;
            // Prefer a measured matching lane size; otherwise interpolate the
            // observed per-image rate. Upload and replica copy are distinct.
            for (const auto observations :
                 {std::span(record.single), std::span(record.parallel)}) {
                for (const auto& candidate : observations) {
                    if (candidate.device != lane.stage.device
                        || (!load && candidate.replica != (i > 0)))
                        continue;
                    if (!sample || candidate.images == lane.images
                        || sample->images != lane.images)
                        sample = &candidate;
                }
            }
            if (!sample)
                return std::nullopt;
            loading += load.value_or(sample->load_seconds);
            denoising = std::max(
                denoising,
                sample->denoise_seconds * double(lane.images) / double(sample->images)
            );
        }
        // Stage loading is sequential; resident denoising overlaps across lanes.
        return loading + denoising;
    }

public:
    // Candidates already satisfy shared memory limits. The caller provides
    // useful device subsets, rather than assuming every available GPU is faster.
    // Planning is read-only; only successful observations consume calibration.
    std::vector<AtenDenoiserLane> choose_candidates(
        const AtenDenoiserWorkload& workload,
        const std::vector<std::vector<AtenDenoiserLane>>& candidates,
        const std::vector<AtenDenoiserCost>& costs = {}
    ) const {
        std::vector<std::size_t> resident;
        std::size_t fallback = candidates.size();
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].empty())
                continue;
            if (fallback == candidates.size())
                fallback = i;
            if (std::none_of(
                    candidates[i].begin(), candidates[i].end(), [](const auto& lane) {
                        return lane.stage.stream_weights;
                    }
                ))
                resident.push_back(i);
        }
        if (fallback == candidates.size())
            return {};
        if (resident.empty())
            return candidates[fallback];
        const auto fewer_lanes = [&](auto a, auto b) {
            return candidates[a].size() < candidates[b].size();
        };
        const auto baseline =
            *std::min_element(resident.begin(), resident.end(), fewer_lanes);
        if (!costs.empty() && costs.size() != candidates.size())
            return candidates[baseline];
        const AtenDenoiserCost unknown_cost;
        const auto current_cost = [&](std::size_t i) -> const AtenDenoiserCost& {
            return costs.empty() ? unknown_cost : costs[i];
        };
        if (!valid_cost(current_cost(baseline), candidates[baseline].size()))
            return candidates[baseline];
        const auto found =
            std::find_if(history_.begin(), history_.end(), [&](const auto& r) {
                return r.workload == workload;
            });
        if (found == history_.end())
            return candidates[baseline];
        auto best = baseline;
        auto best_seconds = cost(*found, candidates[baseline], current_cost(baseline));
        const auto unmeasured_device = [&](const auto& lane) {
            return std::none_of(
                found->parallel.begin(),
                found->parallel.end(),
                [&](const auto& sample) { return sample.device == lane.stage.device; }
            );
        };
        // Probe the largest worthwhile unmeasured subset. Even assuming free
        // compute and unknown uploads, known loading and displaced-bank reloads
        // can already outweigh the baseline. Keep narrower subsets eligible.
        auto probe = baseline;
        for (auto i : resident) {
            if (candidates[i].size() <= candidates[probe].size()
                || !valid_cost(current_cost(i), candidates[i].size())
                || std::none_of(
                    candidates[i].begin(), candidates[i].end(), unmeasured_device
                ))
                continue;
            double loading = current_cost(i).deferred_reload_seconds;
            for (const auto& seconds : current_cost(i).load_seconds)
                loading += seconds.value_or(0);
            if (!best_seconds || loading < *best_seconds)
                probe = i;
        }
        if (probe != baseline)
            return candidates[probe];
        if (!best_seconds)
            return candidates[baseline];
        for (auto i : resident) {
            if (!valid_cost(current_cost(i), candidates[i].size()))
                continue;
            const auto seconds = cost(*found, candidates[i], current_cost(i));
            if (seconds && *seconds < *best_seconds) {
                best = i;
                best_seconds = seconds;
            }
        }
        return candidates[best];
    }

    // Call only after a complete successful batch, with synchronized device
    // timings. Invalid/incomplete samples cannot consume the calibration run.
    void observe(
        const AtenDenoiserWorkload& workload,
        std::span<const AtenDenoiserMeasurement> measurements
    ) {
        if (measurements.empty() || measurements.size() > history_limit)
            return;
        for (std::size_t i = 0; i < measurements.size(); ++i) {
            const auto& sample = measurements[i];
            if (sample.device < 0 || !sample.images || sample.replica != (i > 0)
                || !std::isfinite(sample.load_seconds) || sample.load_seconds < 0
                || !std::isfinite(sample.denoise_seconds)
                || sample.denoise_seconds <= 0)
                return;
        }
        auto found = std::find_if(history_.begin(), history_.end(), [&](const auto& r) {
            return r.workload == workload;
        });
        if (found == history_.end()) {
            if (history_.size() == history_limit)
                history_.erase(history_.begin());
            history_.push_back({workload, {}, {}});
            found = std::prev(history_.end());
        }
        auto& target = measurements.size() == 1 ? found->single : found->parallel;
        // Keep the latest measurement per device and loading role. Narrower
        // subsequent batches must not erase evidence about other GPUs.
        for (const auto& sample : measurements) {
            const auto prior =
                std::find_if(target.begin(), target.end(), [&](const auto& existing) {
                    return existing.device == sample.device
                           && existing.replica == sample.replica;
                });
            if (prior != target.end())
                target.erase(prior);
            if (target.size() == history_limit)
                target.erase(target.begin());
            target.push_back(sample);
        }
    }
};

}  // namespace kadan::image
