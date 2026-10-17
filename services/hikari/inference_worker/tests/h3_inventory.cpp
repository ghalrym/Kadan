// Synthetic checkpoint headers only: no weights decoded, model load or
// execution.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

#include "kadan/h3_generation.hpp"

void check(bool value) {
    if (!value)
        throw std::runtime_error("test_failed");
}

class InventoryCompute final : public kadan::video::H3Compute {
public:
    bool fits = false;
    kadan::Bytes requested = 0;

    bool prepare_weights(kadan::Bytes bytes) override {
        requested = bytes;
        return fits;
    }

    void dense(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
        throw std::runtime_error("unexpected_inference");
    }

    void attention(
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::size_t,
        std::size_t,
        bool,
        std::span<float>,
        const std::atomic_bool&,
        bool
    ) override {
        throw std::runtime_error("unexpected_inference");
    }

    void convrot(
        std::span<const std::uint8_t>,
        std::span<const float>,
        std::span<const float>,
        std::span<const float>,
        std::size_t,
        std::size_t,
        std::size_t,
        std::span<float>,
        const std::atomic_bool&
    ) override {
        throw std::runtime_error("unexpected_inference");
    }
};

int main() {
    char folder[] = "/tmp/kadan-h3-header-XXXXXX";
    check(mkdtemp(folder));

    struct Cleanup {
        const char* p;

        ~Cleanup() {
            std::filesystem::remove_all(p);
        }
    } cleanup{folder};

    auto write = [&](std::size_t bytes) {
        std::string header =
            "{\"__metadata__\":{\"config\":\"" + std::string(bytes, 'x')
            + "\"},\"w\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
        std::ofstream f(std::string(folder) + "/weights.safetensors", std::ios::binary);
        for (unsigned i = 0; i < 8; ++i)
            f.put(char(std::uint64_t(header.size()) >> (i * 8)));
        f << header;
        f.put(0);
        f.put(0);
    };
    auto resources =
        std::make_shared<kadan::Resources>(kadan::Footprint{64 * 1024 * 1024});
    std::atomic_bool cancel = false;
    kadan::video::H3GenerationPaths paths;
    paths.text = std::string(folder) + "/weights.safetensors";
    for (auto length : {521, 1654, 2013, 8192}) {
        write(length);
        auto cache =
            kadan::video::h3_weight_cache(resources, paths, 1024 * 1024, cancel);
        check(bool(cache.host));
        check(!cache.retain_host_stages && !cache.retain_device_stages);
        cache = {};
        check(resources->snapshot().used[0] == 0);
    }
    {
        // The single BF16 scalar needs 2 raw + 4 decoded bytes. The generous
        // budget is capped to the actual inventory plus bounded cache metadata.
        auto cache =
            kadan::video::h3_weight_cache(resources, paths, 48 * 1024 * 1024, cancel);
        check(cache.retain_host_stages);
        check(cache.host->reserved_bytes() == 8 * 1024 * 1024 + 6);
    }
    check(resources->snapshot().used[0] == 0);
    auto compute = std::make_shared<InventoryCompute>();
    for (bool fits : {false, true}) {
        compute->fits = fits;
        auto cache =
            kadan::video::h3_weight_cache(resources, paths, 0, cancel, compute);
        check(compute->requested == 4);
        check(cache.retain_device_stages == fits && !cache.host);
    }
    write(8193);
    bool rejected = false;
    try {
        auto cache =
            kadan::video::h3_weight_cache(resources, paths, 1024 * 1024, cancel);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected && resources->snapshot().used[0] == 0);
    std::cout << "Bounded H3 inventory metadata parsing passed (no inference)\n";
}
