#pragma once
#include "kadan/image_generation.hpp"

namespace kadan::image {
// C++ operator backend owned by the existing serial image Generator.
class AtenGenerator {
public:
    AtenGenerator(std::shared_ptr<Resources>, std::vector<int>);
    ~AtenGenerator();
    void load(const std::string&, const std::atomic_bool&, const Generator::Hook&);
    void prepare(const ImageRequest&, const std::atomic_bool&);
    void generate(
        const ImageRequest&,
        std::span<float>,
        const std::atomic_bool&,
        const Generator::Hook&
    );
    void prepare_batch(std::span<const ImageRequest>, const std::atomic_bool&);
    void generate_batch(
        std::span<const ImageRequest>,
        std::span<const std::span<float>>,
        const std::atomic_bool&,
        const Generator::BatchHook&
    );
    std::vector<Footprint> admissible_batch_footprints(
        const ImageRequest&, std::size_t, const std::atomic_bool&
    );

    const BatchMeasurement& last_batch_measurement() const {
        return measurement_;
    }

    ImageDeviceCacheMemory device_cache_memory() const;
    void unload();

private:
    Footprint admission_footprint(
        const ImageRequest&, std::size_t images, const std::atomic_bool&
    );
    BatchMeasurement measurement_;
    struct Impl;
    std::unique_ptr<Impl> model_;
    std::shared_ptr<Resources> resources_;
    std::vector<int> devices_;
};
}  // namespace kadan::image
