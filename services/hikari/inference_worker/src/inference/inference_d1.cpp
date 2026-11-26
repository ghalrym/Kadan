#include "kadan/inference_engines.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kadan/checkpoint.hpp"
#include "kadan/lfm3.hpp"
#include "kadan/preparation.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::serving {
namespace {

constexpr std::size_t kD1ContextLimit = 65536;
constexpr std::size_t kD1SetSize = 8;
constexpr auto kD1Deadline = std::chrono::seconds(180);

class Lfm3DecisionEngine final : public InferenceEngine {
public:
    Lfm3DecisionEngine(std::shared_ptr<Resources> resources, Progress progress)
        : resources_(std::move(resources)), progress_(std::move(progress)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto config_doc =
            read_document(request.configuration.checkpoint / "config.json");
        config_ = lfm3::BackboneConfig::parse(config_doc);
        const auto cal_path = request.configuration.checkpoint / "calibration.json";
        if (std::filesystem::exists(cal_path)) {
            calibration_ = lfm3::D1Calibration::parse(read_document(cal_path));
        } else {
            calibration_ = lfm3::D1Calibration{};
        }
        auto budget = std::make_shared<checkpoint::MemoryBudget>(64ULL * 1024 * 1024);
        auto shard = std::make_shared<checkpoint::Shard>(
            request.configuration.checkpoint.string().c_str(),
            "model.safetensors",
            std::move(budget)
        );
        // d1 runs strictly CPU-only F32 (compute = nullptr).
        backbone_ = std::make_unique<lfm3::Backbone>(
            config_,
            shard,
            "model.language_model.",
            resources_,
            Workload::lfm3_decision,
            nullptr,
            true,
            true
        );
        shard_ = std::move(shard);
        tokenizer_ =
            std::make_unique<text::Tokenizer>(resources_, Workload::lfm3_decision);
        tokenizer_->load(
            (request.configuration.checkpoint / "tokenizer.json").string(), cancel
        );
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto& input = std::get<DecisionRequest>(request.input);
        preparation_check(
            !input.state.empty() && input.state.size() <= 8000,
            "d1 state must fit 8000 characters.",
            422
        );
        preparation_check(
            !input.questions.empty() && input.questions.size() <= 8,
            "d1 requires one to eight questions.",
            422
        );
        request_memory_ = std::make_unique<HostAllocation>(
            resources_, Workload::lfm3_decision, 64ULL * 1024 * 1024
        );
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        const auto& input = std::get<DecisionRequest>(request.input);
        progress_("loaded", 1);

        // Enforce 180-second execution deadline alongside external cancellation.
        std::atomic_bool combined_cancel{false};
        std::atomic_bool done{false};
        std::atomic_bool timed_out{false};
        std::thread watchdog([&] {
            const auto deadline = std::chrono::steady_clock::now() + kD1Deadline;
            while (!done.load()) {
                if (cancel.load()) {
                    combined_cancel.store(true);
                    return;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    timed_out.store(true);
                    combined_cancel.store(true);
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });

        Json answers;
        std::exception_ptr error;
        try {
            InferenceTrace tracing(request.trace, "evaluate");
            answers = lfm3::evaluate_d1_decisions(
                input.state,
                input.questions,
                *backbone_,
                *tokenizer_,
                calibration_,
                kD1SetSize,
                kD1ContextLimit,
                combined_cancel,
                [this](std::size_t input_tokens, std::size_t) {
                    progress_("evaluating", input_tokens);
                }
            );
        } catch (...) {
            error = std::current_exception();
        }
        done.store(true);
        if (watchdog.joinable())
            watchdog.join();

        request_memory_.reset();
        if (timed_out.load()) {
            throw RequestError("d1 decision evaluation exceeded the 180s deadline.", 504);
        }
        if (error)
            std::rethrow_exception(error);
        return answers;
    }

    void close() override {
        request_memory_.reset();
        tokenizer_.reset();
        backbone_.reset();
        shard_.reset();
    }

private:
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    lfm3::BackboneConfig config_;
    lfm3::D1Calibration calibration_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::unique_ptr<lfm3::Backbone> backbone_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<HostAllocation> request_memory_;
};

}  // namespace

std::unique_ptr<InferenceEngine>
lfm3_decision_engine(std::shared_ptr<Resources> resources, Progress progress) {
    return std::make_unique<Lfm3DecisionEngine>(
        std::move(resources), std::move(progress)
    );
}

}  // namespace kadan::serving
