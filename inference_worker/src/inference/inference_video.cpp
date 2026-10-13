#include "kadan/h3_generation.hpp"
#include "kadan/h3_memory.hpp"
#include "kadan/h3_profile.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"
#include "kadan/video_progress.hpp"

namespace kadan::serving {
namespace {
class VideoEngine final : public InferenceEngine {
public:
    VideoEngine(std::shared_ptr<Resources> resources, Progress progress)
        : resources_(std::move(resources)), progress_updates_(std::move(progress)),
          progress_([this](const char* phase, std::size_t value) {
              progress_updates_.update(phase, value);
          }) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        if (available[0] < 4ULL * 1024 * 1024 * 1024)
            throw AdmissionWait("video_host_headroom");
        const auto& input = std::get<VideoRequest>(request.input);
        Bytes host_peak = 4ULL * 1024 * 1024 * 1024;
        if (request.operation != Operation::load) {
            const auto profile =
                video::h3_api_profile(input.short_edge, input.aspect, input.duration);
            host_peak = video::h3_request_memory(profile, input.prompt.size());
        }
        if (host_peak > available[0])
            throw AdmissionWait("video_request_headroom");
        paths_ = input.paths;
        const auto plan = media_plan(available, 3ULL * 1024 * 1024 * 1024);
        Footprint contexts(available.size());
        for (int device : plan.devices)
            contexts[device + 1] = compute_context_bytes;
        context_ = resources_->reserve(Workload::video, contexts);
        compute_ = video::h3_cuda_compute(resources_, plan.devices);
        // Retention uses only bounded native reservations; disk remains the backing
        // store.
        const Bytes cache_bytes =
            std::min<Bytes>(4ULL * 1024 * 1024 * 1024, (available[0] - host_peak) / 8);
        cache_ = video::h3_weight_cache(
            resources_, paths_, cache_bytes, cancel, compute_, progress_
        );
        model_ = std::make_unique<video::H3Generation>(
            resources_, compute_, cache_, plan.devices, media_activity_
        );
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        if (request.operation == Operation::load)
            return;
        const auto& input = std::get<VideoRequest>(request.input);
        const auto profile =
            video::h3_api_profile(input.short_edge, input.aspect, input.duration);
        request_ = {
            input.prompt,
            (request.workspace / "video.mp4").string(),
            profile.width,
            profile.height,
            profile.frames,
            4u,
            input.seed
        };
        request_.output_frames = profile.output_frames;
        require(
            !request_.prompt.empty() && request_.prompt.size() <= 32000,
            "video_prompt_bounds"
        );
        if (available[0] < video::h3_request_memory(profile, request_.prompt.size()))
            throw AdmissionWait("video_request_headroom");
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        if (request.operation == Operation::load)
            return Json::object();
        progress_updates_.reset();
        model_->execute(paths_, request_, cancel, progress_);
        compute_->release_scratch();
        return present_video(request, request_.output);
    }

    void close() override {
        model_.reset();
        cache_.reset();
        // An unconfirmed resident launcher keeps its context; never reset under it.
        if (media_activity_->owners.load())
            throw std::runtime_error("video_media_session_busy");
        if (compute_)
            compute_->release_devices();
        compute_.reset();
        if (context_) {
            resources_->released(context_);
            context_ = 0;
        }
    }

private:
    std::shared_ptr<Resources> resources_;
    video::VideoProgress progress_updates_;
    Progress progress_;
    Handle context_ = 0;
    std::shared_ptr<video::H3Compute> compute_;
    std::shared_ptr<MediaSessionActivity> media_activity_ =
        std::make_shared<MediaSessionActivity>();
    std::shared_ptr<checkpoint::ReadCache> cache_;
    std::unique_ptr<video::H3Generation> model_;
    video::H3GenerationPaths paths_;
    video::H3GenerationRequest request_;
};
}  // namespace

std::unique_ptr<InferenceEngine>
video_engine(std::shared_ptr<Resources> resources, Progress progress) {
    return std::make_unique<VideoEngine>(std::move(resources), std::move(progress));
}
}  // namespace kadan::serving
