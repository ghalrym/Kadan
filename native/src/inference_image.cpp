#include "kadan/image_generation.hpp"
#include "kadan/inference_engines.hpp"
#include <filesystem>
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
      : resources_(std::move(r)), progress_(std::move(p)) {}
  void load(const InferenceRequest &request, const Footprint &available,
            const std::atomic_bool &cancel) override {
    if (available[0] < 16ULL * 1024 * 1024 * 1024)
      throw AdmissionWait("image_host_headroom");
    auto input = Json::parse(request.payload);
    compute_ = std::make_unique<WorkerCompute>(media_plan(available),
                                               Workload::image, resources_);
    model_ = std::make_unique<image::Generator>(resources_, compute_->compute);
    model_->load(input.at("checkpoint"), cancel, progress_);
  }
  void prepare(const InferenceRequest &request, const Footprint &available,
               const std::atomic_bool &cancel) override {
    check_cancel(cancel);
    request_memory_.reset();
    if (request.operation == Operation::load)
      return;
    auto input = Json::parse(request.payload);
    const auto &body = input.at("request");
    request_ = {body.at("prompt"), body.at("width"), body.at("height"),
                body.at("steps"), body.at("seed")};
    image::Generator::validate(request_);
    count_ = body.value("count", 1u);
    require(count_ == 1 || count_ == 2 || count_ == 4, "image_count");
    output_ = input.at("output");
    const Bytes bytes = 8 * 1024 * 1024 + 64 * request_.height * request_.width;
    if (bytes > available[0])
      throw AdmissionWait("image_request_headroom");
    request_memory_ =
        std::make_unique<HostAllocation>(resources_, Workload::image, bytes);
  }
  std::string execute(const InferenceRequest &request,
                      const std::atomic_bool &cancel) override {
    if (request.operation == Operation::load)
      return "{}";
    Json paths = Json::array();
    for (std::size_t i = 0; i < count_; ++i) {
      check_cancel(cancel);
      std::vector<float> rgba(request_.height * request_.width * 4);
      auto current = request_;
      current.seed += i;
      model_->generate(current, rgba, cancel, progress_);
      auto path = output_ + "/image-" + std::to_string(i) + ".png";
      image::publish_png(path, rgba, current.height, current.width, cancel);
      paths.push_back(path);
    }
    compute_->idle();
    request_memory_.reset();
    return Json{{"images", paths}}.dump();
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
std::unique_ptr<InferenceEngine> image_engine(std::shared_ptr<Resources> r,
                                              Progress p) {
  return std::make_unique<ImageEngine>(std::move(r), std::move(p));
}
} // namespace kadan::serving
