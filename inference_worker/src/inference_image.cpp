#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string_view>

#include "kadan/image_generation.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
class ImageEngine final : public InferenceEngine {
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::unique_ptr<WorkerCompute> compute_;
    std::unique_ptr<image::Generator> model_;
    std::unique_ptr<HostAllocation> request_memory_;
    image::ImageRequest request_;
    std::string output_;
    std::size_t count_ = 1;

public:
    ImageEngine(std::shared_ptr<Resources> r, Progress p)
        : resources_(std::move(r)), progress_(std::move(p)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        if (available[0] < 16ULL * 1024 * 1024 * 1024)
            throw AdmissionWait("image_host_headroom");
        compute_ = std::make_unique<WorkerCompute>(
            media_plan(available), Workload::image, resources_
        );
        model_ = std::make_unique<image::Generator>(
            resources_, compute_->compute, compute_->media_activity, compute_->devices
        );
        model_->load(request.configuration.checkpoint.string(), cancel, progress_);
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        request_memory_.reset();
        if (request.operation == Operation::load)
            return;
        const auto& input = std::get<ImageRequest>(request.input);
        request_ = {input.prompt, input.width, input.height, input.steps, *input.seed};
        image::Generator::validate(request_);
        count_ = input.count;
        require(count_ == 1 || count_ == 2 || count_ == 4, "image_count");
        output_ = request.workspace.string();
        const Bytes bytes = 8 * 1024 * 1024 + 64 * request_.height * request_.width;
        if (bytes > available[0])
            throw AdmissionWait("image_request_headroom");
        request_memory_ =
            std::make_unique<HostAllocation>(resources_, Workload::image, bytes);
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        if (request.operation == Operation::load)
            return Json::object();
        const auto* timing = std::getenv("KADAN_IMAGE_TIMING");
        const bool timing_enabled = timing && std::string_view(timing) == "1";
        std::vector<fs::path> paths;
        for (std::size_t i = 0; i < count_; ++i) {
            check_cancel(cancel);
            progress_("image_started", i + 1);
            std::vector<float> rgba(request_.height * request_.width * 4);
            auto current = request_;
            current.seed += i;
            image::DenoisingTimings::Sink timing_sink;
            if (timing_enabled)
                timing_sink = [&](const image::DenoisingStepTiming& value) {
                    Json summary = value;
                    summary["id"] = request.id;
                    summary["image"] = i + 1;
                    // A fixed-size diagnostic line, separate from task/progress output.
                    const auto line = "KADAN_IMAGE_TIMING " + summary.dump() + '\n';
                    if (line.size() <= 4096) {
                        std::fwrite(line.data(), 1, line.size(), stderr);
                        std::fflush(stderr);
                    }
                };
            model_->generate(current, rgba, cancel, progress_, timing_sink);
            auto path = output_ + "/image-" + std::to_string(i) + ".png";
            image::publish_png(path, rgba, current.height, current.width, cancel);
            paths.emplace_back(path);
        }
        compute_->idle();
        request_memory_.reset();
        return present_image(request, paths);
    }

    void close() override {
        request_memory_.reset();
        if (model_)
            model_->unload();
        model_.reset();
        if (compute_)
            compute_->park();
        compute_.reset();
    }
};

std::unique_ptr<InferenceEngine>
image_engine(std::shared_ptr<Resources> r, Progress p) {
    return std::make_unique<ImageEngine>(std::move(r), std::move(p));
}
}  // namespace kadan::serving
