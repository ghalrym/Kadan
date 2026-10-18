#include "kadan/inference_engines.hpp"
#include "kadan/s1_mini.hpp"
#include "kadan/s1_mini_prompt.hpp"
#include "kadan/text_tokenizer.hpp"
#include "kadan/utf8_stream.hpp"

namespace kadan::serving {
namespace {
class TranscriptionFormattingEngine final : public InferenceEngine {
public:
    BackendExecution execution_policy() const override {
        return BackendExecution::isolated;
    }

    TranscriptionFormattingEngine(
        std::shared_ptr<Resources> resources, Progress progress
    )
        : resources_(std::move(resources)), progress_(std::move(progress)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        model_ = std::make_unique<s1::Model>(resources_);
        model_->load(request.configuration.checkpoint.string(), cancel);
        tokenizer_ =
            std::make_unique<text::Tokenizer>(resources_, Workload::formatting);
        tokenizer_->load(
            (request.configuration.checkpoint / "tokenizer.json").string(), cancel
        );
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        const auto& input = std::get<TranscriptionFormattingRequest>(request.input);
        request_memory_ = std::make_unique<HostAllocation>(
            resources_,
            Workload::formatting,
            8 * 1024 * 1024 + s1::context_tokens * sizeof(std::uint32_t)
        );
        prompt_.resize(s1::context_tokens);
        const auto transcript_tokens = tokenizer_->encode(input.text, prompt_, cancel);
        require(
            transcript_tokens <= s1::max_transcript_tokens,
            "S1-mini accepts at most 1000 transcript tokens; use a shorter recording."
        );
        const auto count = tokenizer_->encode(
            s1::prompt(input.text), prompt_, cancel, s1::max_transcript_bytes + 1024
        );
        prompt_.resize(count);
        require(
            count + s1::max_output_tokens <= s1::context_tokens,
            "S1-mini prompt exceeds context limit"
        );
        model_->begin_request(s1::context_tokens);
    }

    Json execute(const InferenceRequest&, const std::atomic_bool& cancel) override {
        progress_("formatting", 0);
        auto selected = model_->prefill(prompt_, cancel);
        std::string output, pending;
        bool stopped = false;
        for (std::size_t generated = 0; generated < s1::max_output_tokens;
             ++generated) {
            check_cancel(cancel);
            if (model_->is_eos(selected)) {
                stopped = true;
                break;
            }
            pending += tokenizer_->decode(selected);
            output += utf8_chunk(pending);
            progress_("formatting", generated + 1);
            selected = model_->step(selected, cancel);
        }
        stopped = stopped || model_->is_eos(selected);
        require(
            stopped, "S1-mini reached its output limit; raw transcription retained."
        );
        output += utf8_chunk(pending, true);
        model_->reset_request();
        std::vector<std::uint32_t>().swap(prompt_);
        request_memory_.reset();
        return {{"text", output}, {"model", "superwhisper/s1-mini"}};
    }

    void close() override {
        std::vector<std::uint32_t>().swap(prompt_);
        request_memory_.reset();
        tokenizer_.reset();
        if (model_)
            model_->close();
        model_.reset();
    }

private:
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::unique_ptr<s1::Model> model_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<std::uint32_t> prompt_;
};
}  // namespace

std::unique_ptr<InferenceEngine> transcription_formatting_engine(
    std::shared_ptr<Resources> resources, Progress progress
) {
    return std::make_unique<TranscriptionFormattingEngine>(
        std::move(resources), std::move(progress)
    );
}
}  // namespace kadan::serving
