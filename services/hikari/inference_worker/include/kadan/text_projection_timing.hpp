#pragma once
#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <tuple>

namespace kadan::text {
// Diagnostic aggregates only. No tensor contents, layer IDs or expert IDs.
struct ProjectionTiming {
    using Key = std::
        tuple<std::string, unsigned, std::size_t, std::size_t, unsigned, std::string>;

    struct Sample {
        std::size_t calls = 0, rows = 0;
        double source_wall_seconds = 0;
    };

    std::map<Key, Sample> groups;
    std::size_t dropped = 0, batch_calls = 0, batch_unsupported = 0;
    double batch_wall_seconds = 0;

    static unsigned row_bucket(std::size_t rows) {
        return rows == 1 ? 1 : rows <= 8 ? 8 : rows <= 32 ? 32 : rows <= 64 ? 64 : 65;
    }

    void
    add(std::string_view name,
        unsigned encoding,
        std::size_t in,
        std::size_t out,
        std::size_t rows,
        const char* dispatch,
        double source_wall_seconds = 0) {
        // Model-owned names: collapse repeated layers, retain projection family.
        if (name.starts_with("blk.")) {
            const auto end = name.find('.', 4);
            if (end != std::string_view::npos)
                name.remove_prefix(end + 1);
        }
        Key key{std::string(name), encoding, in, out, row_bucket(rows), dispatch};
        auto found = groups.find(key);
        if (found == groups.end()) {
            // Bound diagnostic memory and the size of each phase's log record.
            if (groups.size() == 1024) {
                ++dropped;
                return;
            }
            found = groups.emplace(std::move(key), Sample{}).first;
        }
        auto& sample = found->second;
        ++sample.calls;
        sample.rows += rows;
        sample.source_wall_seconds += source_wall_seconds;
    }
};
}  // namespace kadan::text
