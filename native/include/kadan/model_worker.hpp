#pragma once
#include <cstddef>

#include "kadan/weight_backing.hpp"

namespace kadan::serving {
constexpr std::size_t max_resident_capacity = 1048576;

struct Info {
    std::size_t vocabulary, capacity, arena_bytes, host_bytes;
};

struct Token {
    unsigned id;
    bool eos;
    std::size_t committed;
};

class Engine {
public:
    virtual ~Engine() = default;
    virtual Info info() const = 0;
    virtual void reset() = 0;
    virtual Token step(unsigned, bool) = 0;
    // Must verify physical cleanup and zero reservations before returning.
    virtual void close() = 0;
};

class ResidentEngine : public Engine {
public:
    virtual void begin_request() = 0;
    virtual void end_request() = 0;
    virtual void park() = 0;

    virtual WeightBacking::Stats cache_stats() const {
        return {};
    }
};

}  // namespace kadan::serving
