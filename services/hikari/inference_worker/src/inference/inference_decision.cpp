#include "kadan/decision_executor.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"

namespace kadan::serving {
class DecisionEngine final : public InferenceEngine {
    std::shared_ptr<Resources> resources_;
    decision::Executor executor_;
    Progress progress_;

public:
    BackendExecution execution_policy() const override {
        return BackendExecution::isolated;
    }

    DecisionEngine(std::shared_ptr<Resources> r, Progress progress)
        : resources_(std::move(r)), executor_(resources_),
          progress_(std::move(progress)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        if (available[0] < decision::Executor::envelope_bytes)
            throw AdmissionWait("decision_host_headroom");
        executor_.load(request.configuration.checkpoint.string(), cancel);
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        if (request.operation == Operation::load)
            return;
        const auto scope = resources_->snapshot();
        if (decision::Executor::request_memory_bytes
            > std::min(available[0], scope.capacity[0] - scope.used[0]))
            throw AdmissionWait("decision_request_headroom");
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        if (request.operation == Operation::load)
            return Json::object();
        const auto& input = std::get<DecisionRequest>(request.input);
        return present_decisions(executor_.execute(
            input, cancel, [&](std::string_view) { progress_("decision_stage", 0); }
        ));
    }

    void close() override {
        executor_.unload();
    }
};

std::unique_ptr<InferenceEngine>
decision_engine(std::shared_ptr<Resources> r, Progress p) {
    return std::make_unique<DecisionEngine>(std::move(r), std::move(p));
}
}  // namespace kadan::serving
