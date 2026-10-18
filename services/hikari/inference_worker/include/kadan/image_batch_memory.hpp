#pragma once

#include <algorithm>
#include <map>
#include <span>

#include "kadan/resources.hpp"

namespace kadan::serving {
// Per-shape measurements. A new shape earns a singleton baseline before the
// next batch can grow by one image; physical headroom bounds every device.
class ImageBatchMemory {
public:
    void observe(
        std::size_t images,
        const Footprint& incremental_peak,
        const Footprint& planned_increment
    ) {
        if (!images || incremental_peak.empty()
            || planned_increment.size() != incremental_peak.size())
            throw std::invalid_argument("image_batch_measurement");
        if (planned_increment_.empty())
            planned_increment_.resize(planned_increment.size());
        if (planned_increment_.size() != planned_increment.size())
            throw std::invalid_argument("image_batch_memory_shape");
        for (std::size_t index = 0; index < planned_increment.size(); ++index)
            planned_increment_[index] =
                std::max(planned_increment_[index], planned_increment[index]);
        auto& measured = peaks_[images];
        if (measured.empty())
            measured.resize(incremental_peak.size());
        if (measured.size() != incremental_peak.size())
            throw std::invalid_argument("image_batch_memory_shape");
        for (std::size_t index = 0; index < measured.size(); ++index)
            measured[index] = std::max(measured[index], incremental_peak[index]);
    }

    std::size_t
    capacity(const Footprint& available, std::span<const Footprint> plans) const {
        const auto planned = plans.size();
        if (!planned || peaks_.empty())
            return std::min<std::size_t>(1, planned);
        Footprint increment = planned_increment_;
        for (auto current = peaks_.begin(); current != peaks_.end(); ++current) {
            if (current->second.size() != available.size())
                throw std::invalid_argument("image_batch_memory_shape");
            for (std::size_t index = 0; index < available.size(); ++index) {
                if (current == peaks_.begin())
                    continue;
                const auto previous = std::prev(current);
                // A newly used GPU adds a whole weight replica. That fixed
                // cost is already in its measured peak, not a per-image slope.
                if (index != 0 && previous->second[index] == 0)
                    continue;
                if (current->second[index] > previous->second[index])
                    increment[index] = std::max(
                        increment[index],
                        (current->second[index] - previous->second[index]
                         + current->first - previous->first - 1)
                            / (current->first - previous->first)
                    );
            }
        }
        auto count = std::min(planned, peaks_.rbegin()->first + 1);
        for (; count > 1; --count) {
            const auto& plan = plans[count - 1];
            if (plan.size() != available.size())
                throw std::invalid_argument("image_batch_memory_shape");
            bool fits = true;
            Footprint peak(available.size());
            std::size_t measured_count = 0;
            for (const auto& [images, sample] : peaks_) {
                if (images > count)
                    break;
                measured_count = images;
                for (std::size_t index = 0; index < available.size(); ++index)
                    peak[index] = std::max(peak[index], sample[index]);
            }
            for (std::size_t index = 0; index < available.size(); ++index) {
                // Historical replicas on an unused GPU cannot veto today's
                // placement. Native plans include all sequential stages too.
                if (!plan[index])
                    continue;
                const Bytes reserve = (index == 0 ? 1024ULL : 512ULL) * 1024 * 1024;
                // Weight staging is fixed, not multiplied by image count. The
                // native stage planner bounds the first 1->2 probe; afterward
                // measured differences also constrain each additional image.
                const auto extra = count - measured_count;
                const Bytes estimate =
                    std::max(plan[index], peak[index] + extra * increment[index]);
                if (available[index] < reserve || estimate > available[index] - reserve)
                    fits = false;
            }
            if (fits)
                break;
        }
        // Singleton admission remains the native planner's responsibility.
        return std::max<std::size_t>(1, count);
    }

private:
    std::map<std::size_t, Footprint> peaks_;
    Footprint planned_increment_;
};
}  // namespace kadan::serving
