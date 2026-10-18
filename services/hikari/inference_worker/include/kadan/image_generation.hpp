#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "kadan/resources.hpp"

namespace kadan::image {
class AtenGenerator;

struct BatchMeasurement {
    std::size_t images = 0;
    // Index zero is host admission; subsequent entries are independent CUDA devices.
    Footprint peak_allocated, peak_reserved;
    Footprint baseline_allocated, baseline_reserved;
    // Conservative extra denoiser workspace for one more compatible image.
    // Text, source and VAE stages remain sequential and share their workspace.
    Footprint additional_image;
};

struct ImageRequest {
    std::string prompt;
    std::size_t width = 2048, height = 2048, steps = 50;
    std::uint64_t seed = 0;
    std::span<const float> source, mask;
    float strength = .65f;
    bool reference = false;
    std::size_t source_width = 0, source_height = 0;
};

// Checkpoint-specific generation, source-latent edits and reference conditioning.
// Source/mask buffers remain caller-owned and admitted throughout the request.
class Generator {
public:
    using Hook = std::function<void(const char*, std::size_t)>;
    using BatchHook = std::function<void(std::size_t, const char*, std::size_t)>;
    explicit Generator(std::shared_ptr<Resources>, std::vector<int> devices);
    ~Generator();
    void load(const std::string& root, const std::atomic_bool&, const Hook& = {});
    void prepare(const ImageRequest&, const std::atomic_bool&);
    void generate(
        const ImageRequest&,
        std::span<float> rgba_chw,
        const std::atomic_bool&,
        const Hook& = {}
    );
    void prepare_batch(std::span<const ImageRequest>, const std::atomic_bool&);
    void generate_batch(
        std::span<const ImageRequest>,
        std::span<const std::span<float>>,
        const std::atomic_bool&,
        const BatchHook& = {}
    );
    std::vector<Footprint> admissible_batch_footprints(
        const ImageRequest&, std::size_t limit, const std::atomic_bool&
    );
    const BatchMeasurement& last_batch_measurement() const;
    static bool compatible(const ImageRequest&, const ImageRequest&);
    void unload();
    static void validate(const ImageRequest&);

private:
    std::unique_ptr<AtenGenerator> aten_;
    bool busy_ = false;
};

// Writes an owned PNG with exclusive creation. Caller admits publication memory.
void publish_png(
    const std::string&,
    std::span<const float> rgba_chw,
    std::size_t height,
    std::size_t width,
    const std::atomic_bool&
);
}  // namespace kadan::image
