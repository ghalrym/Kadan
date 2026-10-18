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
    auto& residents = order(device);
    if (used + slot.bytes > budget) {
        // Displace the coldest unpinned residents only if every one is clearly
        // colder; a pinned slice displaces any of them.
        const auto threshold = slot.priority - std::log2(margin_);
        Bytes freed = 0;
        auto end = residents.begin();
        while (used - freed + slot.bytes > budget && end != residents.end()
               && (slot.pinned || end->first < threshold))
            freed += slots_.at((end++)->second).bytes;
        if (used - freed + slot.bytes > budget)
            return false;
        for (auto it = residents.begin(); it != end; ++it) {
            auto& victim = slots_.at(it->second);
            (device ? victim.device : victim.ram) = false;
            evicted.push_back(it->second);
        }
        residents.erase(residents.begin(), end);
        used -= freed;
    }
    used += slot.bytes;
    if (!slot.pinned)
        residents.emplace(slot.priority, key);
    return true;
}

Placement::Decision Placement::use(const SliceKey& key, Bytes bytes, bool pinned) {
    auto& slot = slots_[key];
    // Resident entries are keyed by the previous priority.
    for (const bool device : {true, false})
        if ((device ? slot.device : slot.ram) && !slot.pinned)
            order(device).erase({slot.priority, key});
    slot.score = score(slot) + 1;
    slot.stamp = clock_;
    slot.priority = std::log2(slot.score) + double(slot.stamp) / half_life_;
    slot.bytes = bytes;
    slot.pinned = pinned;
    for (const bool device : {true, false})
        if ((device ? slot.device : slot.ram) && !slot.pinned)
            order(device).emplace(slot.priority, key);
    Decision decision;
    if (!slot.device && admit(key, slot, true, decision.device_evicted))
        slot.device = true;
    if (slot.device) {
        // A device-resident slice no longer needs its host copy.
        if (slot.ram) {
            slot.ram = false;
            ram_used_ -= slot.bytes;
            if (!slot.pinned)
                ram_order_.erase({slot.priority, key});
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
