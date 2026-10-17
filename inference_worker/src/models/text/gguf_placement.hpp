#pragma once
// Frequency-based weight placement across VRAM, host RAM and disk. A slice is
// one expert of an expert stack or one whole matrix. Each use adds one to a
// score that halves every `half_life` tokens. Slices stay in a tier while they
// fit; a resident slice is displaced only by one whose score exceeds it by
// `margin`, so routinely routed experts stay put and rare ones stream.
#include <compare>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kadan/resources.hpp"

namespace kadan::text {
struct SliceKey {
    std::string tensor;
    std::size_t expert = 0;
    auto operator<=>(const SliceKey&) const = default;
};

class Placement {
public:
    struct Decision {
        bool device = false, ram = false;
        // Slices that lost their tier; the caller releases their storage.
        std::vector<SliceKey> device_evicted, ram_evicted;
    };

    struct Stats {
        Bytes device = 0, ram = 0;
        std::size_t device_slices = 0, ram_slices = 0;
    };

    explicit Placement(double half_life = 512, double margin = 1.5)
        : half_life_(half_life), margin_(margin) {
    }

    void budgets(Bytes device, Bytes ram) {
        device_budget_ = device;
        ram_budget_ = ram;
    }

    // Tokens processed since the last call; ages every score.
    void advance(std::size_t tokens) {
        clock_ += tokens;
    }

    Decision use(const SliceKey& key, Bytes bytes);
    Stats stats() const;

private:
    struct Slot {
        double score = 0;
        std::uint64_t stamp = 0;
        Bytes bytes = 0;
        bool device = false, ram = false;
    };

    double score(const Slot&) const;
    // Admit `key` to a tier when it fits or displaces colder slices.
    bool
    admit(const SliceKey& key, Slot& slot, bool device, std::vector<SliceKey>& evicted);

    double half_life_, margin_;
    std::uint64_t clock_ = 0;
    Bytes device_budget_ = 0, ram_budget_ = 0, device_used_ = 0, ram_used_ = 0;
    std::map<SliceKey, Slot> slots_;
};
}  // namespace kadan::text
