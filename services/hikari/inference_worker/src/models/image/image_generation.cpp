#include "kadan/image_generation.hpp"

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
    need(
        std::isfinite(request.strength) && request.strength >= 0
            && request.strength <= 1
            && (request.source.empty() || request.source.size() == pixels * 4)
            && (request.mask.empty()
                || (!request.source.empty() && request.mask.size() == pixels)),
        "image_edit_bounds"
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
}  // namespace kadan::image
