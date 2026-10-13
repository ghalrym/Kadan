#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <string>

#include "kadan/dense_compute.hpp"
#include "kadan/image_timing.hpp"
#include "kadan/media_session.hpp"

namespace kadan::image {
class AtenGenerator;

struct ImageRequest {
    std::string prompt;
    std::size_t width = 2048, height = 2048, steps = 50;
    std::uint64_t seed = 0;
};

// Complete checkpoint-specific text-to-image pipeline, one image at a time. Editing
// and CFG are not supported by this boundary. Executors own separate pipelines.
class Generator {
public:
    using Hook = std::function<void(const char*, std::size_t)>;
    explicit Generator(
        std::shared_ptr<Resources>,
        std::shared_ptr<DenseCompute>,
        std::shared_ptr<MediaSessionActivity>,
        std::vector<int> devices
    );
    ~Generator();
    void load(const std::string& root, const std::atomic_bool&, const Hook& = {});
    void prepare(const ImageRequest&, const std::atomic_bool&);
    void generate(
        const ImageRequest&,
        std::span<float> rgba_chw,
        const std::atomic_bool&,
        const Hook& = {},
        const DenoisingTimings::Sink& = {}
    );
    void unload();
    static void validate(const ImageRequest&);

private:
    struct Impl;
    std::shared_ptr<Resources> resources_;
    std::shared_ptr<DenseCompute> compute_;
    std::shared_ptr<MediaSessionActivity> media_activity_;
    std::vector<int> devices_;
    std::unique_ptr<Impl> model_;
#ifdef KADAN_ATEN_IMAGE
    std::unique_ptr<AtenGenerator> aten_;
#endif
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
