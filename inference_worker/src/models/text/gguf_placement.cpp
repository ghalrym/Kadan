#include "gguf_placement.hpp"

#include <algorithm>
#include <cmath>

namespace kadan::text {
double Placement::score(const Slot& slot) const {
    return slot.score * std::exp2(-double(clock_ - slot.stamp) / half_life_);
}

bool Placement::admit(
    const SliceKey& key, Slot& slot, bool device, std::vector<SliceKey>& evicted
) {
    auto& used = device ? device_used_ : ram_used_;
    const auto budget = device ? device_budget_ : ram_budget_;
    if (slot.bytes > budget)
        return false;
    if (used + slot.bytes <= budget) {
        used += slot.bytes;
        return true;
    }
    // Displace the coldest residents only if every one is clearly colder.
    std::vector<std::pair<double, const SliceKey*>> residents;
    for (const auto& [other, candidate] : slots_)
        if (other != key && (device ? candidate.device : candidate.ram))
            residents.emplace_back(score(candidate), &other);
    std::sort(residents.begin(), residents.end());
    const auto threshold = score(slot) / margin_;
    Bytes freed = 0;
    std::size_t count = 0;
    while (used - freed + slot.bytes > budget && count < residents.size()
           && residents[count].first < threshold)
        freed += slots_.at(*residents[count++].second).bytes;
    if (used - freed + slot.bytes > budget)
        return false;
    for (std::size_t i = 0; i < count; ++i) {
        auto& victim = slots_.at(*residents[i].second);
        (device ? victim.device : victim.ram) = false;
        evicted.push_back(*residents[i].second);
    }
    used = used - freed + slot.bytes;
    return true;
}

Placement::Decision Placement::use(const SliceKey& key, Bytes bytes) {
    auto& slot = slots_[key];
    slot.score = score(slot) + 1;
    slot.stamp = clock_;
    slot.bytes = bytes;
    Decision decision;
    if (!slot.device && admit(key, slot, true, decision.device_evicted))
        slot.device = true;
    if (slot.device) {
        // A device-resident slice no longer needs its host copy.
        if (slot.ram) {
            slot.ram = false;
            ram_used_ -= slot.bytes;
            decision.ram_evicted.push_back(key);
        }
    } else if (!slot.ram && admit(key, slot, false, decision.ram_evicted))
        slot.ram = true;
    decision.device = slot.device;
    decision.ram = slot.ram;
    return decision;
}

Placement::Stats Placement::stats() const {
    Stats result{device_used_, ram_used_, 0, 0};
    for (const auto& [key, slot] : slots_) {
        result.device_slices += slot.device;
        result.ram_slices += slot.ram;
    }
    return result;
}
}  // namespace kadan::text
