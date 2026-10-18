#include "kadan/image_pool_memory.hpp"

#include <atomic>
#include <barrier>
#include <limits>
#include <source_location>
#include <thread>
#include <vector>

namespace {
void check(bool value, std::source_location at = std::source_location::current()) {
    if (!value)
        throw std::runtime_error("image_pool_memory_test:" + std::to_string(at.line()));
}
}  // namespace

int main() {
    using kadan::image::ImagePoolMemory;
    // Simultaneous segment requests cannot each consume the same headroom.
    ImagePoolMemory pool(10);
    std::barrier start(17), admitted(17), release(17);
    std::atomic_int accepted = 0;
    std::vector<std::jthread> threads;
    threads.reserve(16);
    for (int i = 0; i < 16; ++i)
        threads.emplace_back([&] {
            start.arrive_and_wait();
            const bool owns = pool.acquire(2);
            accepted += owns;
            admitted.arrive_and_wait();
            release.arrive_and_wait();
            if (owns)
                pool.release(2);
        });
    start.arrive_and_wait();
    admitted.arrive_and_wait();
    check(accepted == 5 && pool.current() == 10 && pool.peak() == 10);
    release.arrive_and_wait();
    threads.clear();
    check(pool.current() == 0 && pool.peak() == 10);

    // Physical allocation failure rolls back its claim. Unconfirmed frees retain
    // theirs, preventing another allocation from borrowing possibly-live memory.
    check(pool.acquire(7));
    pool.release(7);
    check(pool.acquire(9));
    check(!pool.acquire(2));
    check(pool.current() == 9);
    pool.release(9);
    bool rejected = false;
    try {
        pool.release(1);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    check(rejected && pool.current() == 0);

    // Finishing scratch cannot release an independently resident bank's charge,
    // including another model on the same device.
    ImagePoolMemory weights(100), other_weights(100), scratch(30);
    check(weights.acquire(80) && other_weights.acquire(90) && scratch.acquire(30));
    scratch.release(30);
    check(weights.current() == 80 && other_weights.current() == 90);
    check(!weights.acquire(21) && !other_weights.acquire(11));
    weights.release(80);
    check(other_weights.current() == 90);

    ImagePoolMemory maximum(std::numeric_limits<std::uint64_t>::max());
    check(maximum.acquire(std::numeric_limits<std::uint64_t>::max()));
    check(!maximum.acquire(1));
    maximum.release(std::numeric_limits<std::uint64_t>::max());
    check(maximum.current() == 0);
    ImagePoolMemory zero(0);
    check(!zero.acquire(1));
}
