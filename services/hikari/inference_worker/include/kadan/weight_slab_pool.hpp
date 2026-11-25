#pragma once
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "kadan/resources.hpp"

namespace kadan {
// Sub-allocates 256-byte aligned weight slices from fixed-size device slabs so
// thousands of sub-2MiB routed expert slices do not each consume a 2MiB CUDA
// VMM page. Slabs are returned immediately via raw free once their last live
// slice is freed, preserving exact device-memory release guarantees.
class WeightSlabPool {
public:
    static constexpr Bytes alignment = 256;
    static constexpr Bytes default_slab_bytes = 32ULL * 1024 * 1024;
    static constexpr Bytes default_max_pooled_bytes = 16ULL * 1024 * 1024;

    using RawAlloc = std::function<void*(Bytes)>;
    using RawFree = std::function<void(void*)>;

    WeightSlabPool(
        RawAlloc alloc,
        RawFree free,
        Bytes slab_bytes = default_slab_bytes,
        Bytes max_pooled_bytes = default_max_pooled_bytes
    )
        : alloc_(std::move(alloc)), free_(std::move(free)), slab_bytes_(slab_bytes),
          max_pooled_bytes_(std::min(max_pooled_bytes, slab_bytes)) {
        if (!alloc_ || !free_ || !slab_bytes_ || slab_bytes_ % alignment != 0)
            throw std::invalid_argument("weight_slab_pool_config");
    }

    WeightSlabPool(const WeightSlabPool&) = delete;
    WeightSlabPool& operator=(const WeightSlabPool&) = delete;

    void* allocate(Bytes bytes) {
        if (!bytes)
            return nullptr;
        const auto aligned = align_up(bytes);
        if (!aligned)
            return nullptr;
        if (aligned > max_pooled_bytes_)
            return allocate_direct(bytes);

        // Best-fit across open slabs minimizes fragmentation between mixed
        // gate_up / down / dense slice sizes.
        std::size_t best_slab = 0;
        Bytes best_offset = 0;
        Bytes best_size = std::numeric_limits<Bytes>::max();
        bool found = false;
        for (const auto& [id, slab] : slabs_) {
            if (slab.largest_free < aligned)
                continue;
            for (const auto& [offset, size] : slab.free_blocks) {
                if (size >= aligned && size < best_size) {
                    best_slab = id;
                    best_offset = offset;
                    best_size = size;
                    found = true;
                    if (size == aligned)
                        break;
                }
            }
            if (found && best_size == aligned)
                break;
        }

        if (found) {
            auto& slab = slabs_.at(best_slab);
            void* ptr = static_cast<std::byte*>(slab.base) + best_offset;
            // Insert tracking entry before mutating slab free-list so bad_alloc
            // leaves the slab untouched.
            auto [it, inserted] =
                allocs_.emplace(ptr, Allocation{best_slab, best_offset, aligned});
            if (!inserted)
                throw std::logic_error("weight_slab_duplicate_pointer");
            try {
                slab.free_blocks.erase(best_offset);
                if (best_size > aligned)
                    slab.free_blocks.emplace(best_offset + aligned, best_size - aligned);
                ++slab.live_count;
                if (best_size == slab.largest_free)
                    recompute_largest(slab);
                return ptr;
            } catch (...) {
                allocs_.erase(it);
                throw;
            }
        }

        void* base = alloc_(slab_bytes_);
        if (!base)
            return allocate_direct(bytes);

        const auto id = next_slab_id_++;
        try {
            Slab slab{base, 1, {}, 0};
            if (slab_bytes_ > aligned) {
                slab.free_blocks.emplace(aligned, slab_bytes_ - aligned);
                slab.largest_free = slab_bytes_ - aligned;
            }
            auto [sit, slab_ok] = slabs_.emplace(id, std::move(slab));
            if (!slab_ok)
                throw std::logic_error("weight_slab_duplicate_id");
            try {
                allocs_.emplace(base, Allocation{id, 0, aligned});
            } catch (...) {
                slabs_.erase(sit);
                throw;
            }
            return base;
        } catch (...) {
            free_(base);
            throw;
        }
    }

    void free(void* ptr) {
        if (!ptr)
            return;
        const auto it = allocs_.find(ptr);
        if (it == allocs_.end()) {
            const auto dit = direct_.find(ptr);
            if (dit == direct_.end())
                throw std::invalid_argument("weight_slab_unknown_pointer");
            free_(ptr);
            direct_.erase(dit);
            return;
        }
        const auto alloc = it->second;
        const auto sit = slabs_.find(alloc.slab_id);
        auto& slab = sit->second;
        if (slab.live_count == 1) {
            // Release physical slab before erasing bookkeeping so a failed
            // cudaFree preserves accurate ownership state.
            free_(slab.base);
            allocs_.erase(it);
            slabs_.erase(sit);
            return;
        }

        // Coalesce with immediate neighbours in the offset-ordered free map.
        Bytes start = alloc.offset;
        Bytes size = alloc.size;
        auto next = slab.free_blocks.lower_bound(start);
        bool merge_prev = false;
        auto prev = next;
        if (next != slab.free_blocks.begin()) {
            --prev;
            if (prev->first + prev->second == start) {
                start = prev->first;
                size += prev->second;
                merge_prev = true;
            }
        }
        bool merge_next = false;
        if (next != slab.free_blocks.end() && start + size == next->first) {
            size += next->second;
            merge_next = true;
        }
        if (merge_prev) {
            prev->second = size;
            if (merge_next)
                slab.free_blocks.erase(next);
        } else if (merge_next) {
            slab.free_blocks.erase(next);
            slab.free_blocks.emplace(start, size);
        } else {
            slab.free_blocks.emplace(start, size);
        }
        allocs_.erase(it);
        --slab.live_count;
        if (size > slab.largest_free)
            slab.largest_free = size;
    }

    Bytes largest_free_block() const {
        Bytes best = 0;
        for (const auto& [id, slab] : slabs_)
            best = std::max(best, slab.largest_free);
        return best;
    }

    std::size_t slab_count() const {
        return slabs_.size();
    }

    std::size_t direct_count() const {
        return direct_.size();
    }

private:
    struct Slab {
        void* base = nullptr;
        std::size_t live_count = 0;
        std::map<Bytes, Bytes> free_blocks;
        Bytes largest_free = 0;
    };

    struct Allocation {
        std::size_t slab_id = 0;
        Bytes offset = 0;
        Bytes size = 0;
    };

    static Bytes align_up(Bytes n) {
        if (n > std::numeric_limits<Bytes>::max() - (alignment - 1))
            return 0;
        return (n + alignment - 1) / alignment * alignment;
    }

    static void recompute_largest(Slab& slab) {
        Bytes best = 0;
        for (const auto& [offset, size] : slab.free_blocks)
            best = std::max(best, size);
        slab.largest_free = best;
    }

    void* allocate_direct(Bytes bytes) {
        void* ptr = alloc_(bytes);
        if (!ptr)
            return nullptr;
        try {
            direct_.insert(ptr);
        } catch (...) {
            free_(ptr);
            throw;
        }
        return ptr;
    }

    RawAlloc alloc_;
    RawFree free_;
    Bytes slab_bytes_;
    Bytes max_pooled_bytes_;
    std::size_t next_slab_id_ = 1;
    std::unordered_map<std::size_t, Slab> slabs_;
    std::unordered_map<void*, Allocation> allocs_;
    std::set<void*> direct_;
};
}  // namespace kadan
