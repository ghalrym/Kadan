#include "kadan/decision_executor.hpp"
#include "kadan/inference_engines.hpp"
namespace kadan::serving {
class DecisionEngine final : public InferenceEngine {
  std::shared_ptr<Resources> resources_;
  decision::Executor executor_;
  Progress progress_;

public:
  DecisionEngine(std::shared_ptr<Resources> r, Progress progress)
      : resources_(std::move(r)), executor_(resources_),
        progress_(std::move(progress)) {}
  void load(const InferenceRequest &request, const Footprint &available,
            const std::atomic_bool &cancel) override {
    if (available[0] < decision::Executor::envelope_bytes)
      throw AdmissionWait("decision_host_headroom");
    auto input = Json::parse(request.payload);
    executor_.load(input.at("checkpoint"), cancel);
  }
  void prepare(const InferenceRequest &, const Footprint &,
               const std::atomic_bool &cancel) override {
    check_cancel(cancel);
  }
  std::string execute(const InferenceRequest &request,
                      const std::atomic_bool &cancel) override {
    if (request.operation == Operation::load)
      return "{}";
    auto input = Json::parse(request.payload);
    return executor_.execute(
        input.at("request").dump(), cancel,
        [&](std::string_view) { progress_("decision_stage", 0); });
  }
  void close() override { executor_.unload(); }
};
std::unique_ptr<InferenceEngine> decision_engine(std::shared_ptr<Resources> r,
                                                 Progress p) {
  return std::make_unique<DecisionEngine>(std::move(r), std::move(p));
}
} // namespace kadan::serving
