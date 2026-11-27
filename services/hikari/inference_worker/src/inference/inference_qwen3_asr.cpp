#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"
#include "kadan/qwen3_asr.hpp"
#include "kadan/stt.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::serving {
namespace {
class Qwen3AsrEngine final : public InferenceEngine {
public:
    Qwen3AsrEngine(std::shared_ptr<Resources> resources, Progress progress)
        : resources_(std::move(resources)), progress_(std::move(progress)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        if (available[0] < 1024ULL * 1024 * 1024)
            throw AdmissionWait("qwen3_asr_host_headroom");
        const auto& input = std::get<TranscriptionRequest>(request.input);
        const auto root = request.configuration.checkpoint.string();
        compute_ = std::make_unique<WorkerCompute>(
            media_plan(available), Workload::speech, resources_
        );
        workspace_ = std::make_unique<HostAllocation>(
            resources_, Workload::speech, 256 * 1024 * 1024
        );
        model_ = std::make_unique<stt::Qwen3Asr>(resources_, compute_->compute);
        model_->load(root.c_str(), cancel);

        tokenizer_ = std::make_unique<text::Tokenizer>(
            resources_, Workload::speech, text::Tokenizer::Format::byte_level
        );
        tokenizer_->load((input.tokenizer / "tokenizer.json").string(), cancel);

        frontend_ = std::make_unique<stt::LogMel>(resources_);
        const auto bins = model_->config().audio.num_mel_bins;
        const auto filters = stt::slaney_mel_filters(bins);
        frontend_->load(bins, filters, cancel);
    }

    void prepare(
        const InferenceRequest&, const Footprint&, const std::atomic_bool&
    ) override {
    }

    Json execute(
        const InferenceRequest& request, const std::atomic_bool& cancel
    ) override {
        const auto& input = std::get<TranscriptionRequest>(request.input);
        check_cancel(cancel);
        const auto bins = model_->config().audio.num_mel_bins;
        const auto frames = stt::LogMel::frames(input.pcm.size());
        HostAllocation mel_mem(
            resources_, Workload::speech, bins * frames * sizeof(float)
        );
        std::vector<float> mel(bins * frames);
        frontend_->execute(input.pcm, mel, cancel);
        if (progress_)
            progress_("mel", frames);

        std::string forced_language;
        if (input.language && !input.language->empty()) {
            const auto normalized = stt::normalize_language_hint(*input.language);
            if (normalized)
                forced_language = *normalized;
        }
        const auto result = model_->transcribe(
            mel, frames, *tokenizer_, forced_language, cancel, progress_
        );
        std::optional<std::string> lang =
            result.language.empty() ? std::nullopt : std::optional(result.language);
        return present_transcription(request, result.text, lang);
    }

    ~Qwen3AsrEngine() override {
        close();
    }

    void close() override {
        if (frontend_)
            frontend_->unload();
        frontend_.reset();
        if (tokenizer_)
            tokenizer_->unload();
        tokenizer_.reset();
        if (model_)
            model_->unload();
        model_.reset();
        workspace_.reset();
        compute_.reset();
    }

private:
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::unique_ptr<WorkerCompute> compute_;
    std::unique_ptr<HostAllocation> workspace_;
    std::unique_ptr<stt::Qwen3Asr> model_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<stt::LogMel> frontend_;
};
}  // namespace

std::unique_ptr<InferenceEngine>
qwen3_asr_engine(std::shared_ptr<Resources> resources, Progress progress) {
    return std::make_unique<Qwen3AsrEngine>(std::move(resources), std::move(progress));
}
}  // namespace kadan::serving
