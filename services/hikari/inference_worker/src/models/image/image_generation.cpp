#include "kadan/image_generation.hpp"

#include <algorithm>
#include <cmath>

#include "kadan/image_aten.hpp"

namespace kadan::image {
namespace {
void need(bool b, const char* error) {
    if (!b)
        throw std::runtime_error(error);
}

void stop(const std::atomic_bool& cancel) {
    need(!cancel.load(), "image_cancelled");
}

struct Busy {
    bool& busy;

    explicit Busy(bool& value) : busy(value) {
        need(!busy, "busy");
        busy = true;
    }

    ~Busy() {
        busy = false;
    }
};
}  // namespace

Generator::Generator(std::shared_ptr<Resources> resources, std::vector<int> devices)
    : aten_(std::make_unique<AtenGenerator>(std::move(resources), std::move(devices))) {
}

Generator::~Generator() {
    unload();
}

void Generator::load(
    const std::string& root, const std::atomic_bool& cancel, const Hook& hook
) {
    Busy active(busy_);
    stop(cancel);
    aten_->load(root, cancel, hook);
}

ImageDeviceCacheMemory Generator::device_cache_memory() const {
    return aten_->device_cache_memory();
}

void Generator::unload() {
    need(!busy_, "busy");
    aten_->unload();
}

void Generator::validate(const ImageRequest& request) {
    need(
        !request.prompt.empty() && request.prompt.size() <= 8000 && request.width >= 32
            && request.height >= 32 && request.width <= 3072 && request.height <= 3072
            && request.width * request.height <= 4608 * 1024 && request.width % 32 == 0
            && request.height % 32 == 0 && request.steps >= 2 && request.steps <= 100,
        "image_request_bounds"
    );
    const auto pixels = request.width * request.height;
    const auto source_width =
        request.source_width ? request.source_width : request.width;
    const auto source_height =
        request.source_height ? request.source_height : request.height;
    need(
        source_width >= 32 && source_height >= 32 && source_width <= 3072
            && source_height <= 3072 && source_width % 32 == 0
            && source_height % 32 == 0 && source_width * source_height <= 4608 * 1024,
        "image_source_bounds"
    );
    need(
        std::isfinite(request.strength) && request.strength >= 0
            && request.strength <= 1
            && (request.source.empty()
                || request.source.size() == source_width * source_height * 4)
            && (request.mask.empty()
                || (!request.source.empty() && request.mask.size() == pixels)),
        "image_edit_bounds"
    );
    need(
        request.reference
            ? !request.source.empty() && request.mask.empty()
            : source_width == request.width && source_height == request.height,
        "image_reference_bounds"
    );
    need(
        !request.reference || source_width * source_height >= 65536,
        "image_reference_minimum_65536_pixels"
    );
}

void Generator::prepare(const ImageRequest& request, const std::atomic_bool& cancel) {
    Busy active(busy_);
    stop(cancel);
    aten_->prepare(request, cancel);
}

void Generator::generate(
    const ImageRequest& request,
    std::span<float> output,
    const std::atomic_bool& cancel,
    const Hook& hook
) {
    Busy active(busy_);
    stop(cancel);
    aten_->generate(request, output, cancel, hook);
}

bool Generator::compatible(const ImageRequest& a, const ImageRequest& b) {
    return a.width == b.width && a.height == b.height && a.steps == b.steps
           && a.reference == b.reference && a.source_width == b.source_width
           && a.source_height == b.source_height && a.mask.empty() && b.mask.empty()
           && (a.source.empty() || a.reference) && (b.source.empty() || b.reference)
           && a.source.size() == b.source.size()
           && std::equal(a.source.begin(), a.source.end(), b.source.begin());
}

void Generator::prepare_batch(
    std::span<const ImageRequest> requests, const std::atomic_bool& cancel
) {
    Busy active(busy_);
    stop(cancel);
    aten_->prepare_batch(requests, cancel);
}

void Generator::generate_batch(
    std::span<const ImageRequest> requests,
    std::span<const std::span<float>> outputs,
    const std::atomic_bool& cancel,
    const BatchHook& hook
) {
    Busy active(busy_);
    stop(cancel);
    aten_->generate_batch(requests, outputs, cancel, hook);
}

std::vector<ImageBatchPlan> Generator::admissible_batch_plans(
    const ImageRequest& request, std::size_t limit, const std::atomic_bool& cancel
) {
    Busy active(busy_);
    return aten_->admissible_batch_plans(request, limit, cancel);
}

void Generator::record_batch_measurement(bool calibrated) {
    Busy active(busy_);
    aten_->record_batch_measurement(calibrated);
}

const BatchMeasurement& Generator::last_batch_measurement() const {
    return aten_->last_batch_measurement();
}
}  // namespace kadan::image
