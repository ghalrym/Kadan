#include "kadan/inference_engines.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kadan/checkpoint.hpp"
#include "kadan/lfm3.hpp"
#include "kadan/preparation.hpp"
#include "kadan/text_tokenizer.hpp"

namespace kadan::serving {
namespace {

constexpr std::size_t kVisionMaxContext = 128000;
constexpr std::size_t kPrefillChunkTokens = 512;

ComputePlan vision_plan(const Footprint& available) {
#ifdef KADAN_WORKER_CUDA
    if (available.size() > 1)
        return media_plan(available);
#endif
    return ComputePlan{available, {}};
}

class Lfm3VisionEngine final : public InferenceEngine {
public:
    Lfm3VisionEngine(std::shared_ptr<Resources> resources, Progress progress)
        : resources_(std::move(resources)), progress_(std::move(progress)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto plan = vision_plan(available);
        compute_ =
            std::make_unique<WorkerCompute>(plan, Workload::lfm3_vision, resources_);
        const auto config_doc =
            read_document(request.configuration.checkpoint / "config.json");
        vision_config_ = lfm3::VisionConfig::parse(config_doc);
        text_config_ = lfm3::BackboneConfig::parse(config_doc);
        auto budget = std::make_shared<checkpoint::MemoryBudget>(64ULL * 1024 * 1024);
        auto shard = std::make_shared<checkpoint::Shard>(
            request.configuration.checkpoint.string().c_str(),
            "model.safetensors",
            std::move(budget)
        );
        vision_encoder_ = std::make_unique<lfm3::VisionEncoder>(
            vision_config_, shard, resources_, compute_->compute
        );
        backbone_ = std::make_unique<lfm3::Backbone>(
            text_config_,
            shard,
            "model.language_model.",
            resources_,
            Workload::lfm3_vision,
            compute_->compute,
            true,
            true
        );
        shard_ = std::move(shard);
        tokenizer_ =
            std::make_unique<text::Tokenizer>(resources_, Workload::lfm3_vision);
        tokenizer_->load(
            (request.configuration.checkpoint / "tokenizer.json").string(), cancel
        );
        im_end_id_ = tokenizer_->find_token("<|im_end|>");
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto& input = std::get<VisionRequest>(request.input);
        const std::string raw_bytes = base64_decode(
            input.image_base64, 4 * ((2 * 1024 * 1024 + 2) / 3)
        );
        preparation_check(
            !raw_bytes.empty() && raw_bytes.size() <= 2 * 1024 * 1024,
            "Screenshot must contain at most 2 MiB of image bytes.",
            422
        );
        const lfm3::RgbImage rgb =
            lfm3::decode_vision_image(raw_bytes, input.media_type, 4194304);
        patches_ = lfm3::preprocess_vision_image(rgb, vision_config_);
        const std::size_t merge = vision_config_.spatial_merge_size;
        const std::size_t image_tokens =
            (patches_.grid_h / merge) * (patches_.grid_w / merge);

        std::string image_placeholders;
        image_placeholders.reserve(image_tokens * 9 + input.question.size() + 16);
        for (std::size_t i = 0; i < image_tokens; ++i)
            image_placeholders += "<|image|>";
        image_placeholders += input.question;

        std::vector<lfm3::Lfm3ChatMessage> messages;
        messages.push_back({"user", image_placeholders, Json::array(), ""});
        const std::string prompt_text =
            lfm3::render_lfm3_chat_prompt(messages, Json(), false, false);

        prompt_tokens_.assign(kVisionMaxContext + 1, 0);
        const std::size_t count = tokenizer_->encode(
            prompt_text, prompt_tokens_, cancel,
            std::max<std::size_t>(prompt_text.size() + 64, 4 * 1024 * 1024)
        );
        prompt_tokens_.resize(count);
        preparation_check(
            count >= 1 && count < kVisionMaxContext,
            "Screenshot and question leave no output space in the 128000-token context; no input was truncated.",
            422
        );
        const std::size_t budget = input.max_new_tokens
                                       ? *input.max_new_tokens
                                       : (kVisionMaxContext - count);
        preparation_check(
            budget >= 1 && count + budget <= kVisionMaxContext,
            "Screenshot, question and reserved reply exceed the model's 128000-token context; no input was truncated.",
            422
        );
        output_budget_ = budget;
        request_memory_ = std::make_unique<HostAllocation>(
            resources_,
            Workload::lfm3_vision,
            64ULL * 1024 * 1024 + patches_.pixel_values.size() * sizeof(float)
        );
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        check_cancel(cancel);
        const std::size_t h = text_config_.hidden_size;
        const std::size_t v = text_config_.vocab_size;

        progress_("vision_encode", patches_.num_patches);
        std::vector<float> image_embeds;
        {
            InferenceTrace tracing(request.trace, "vision_encode");
            image_embeds = vision_encoder_->encode(patches_, cancel);
        }
        const std::size_t num_img_tokens = image_embeds.size() / h;

        std::vector<float> prompt_embeds(prompt_tokens_.size() * h);
        backbone_->embed_tokens(prompt_tokens_, prompt_embeds, cancel);
        std::size_t img_idx = 0;
        for (std::size_t t = 0; t < prompt_tokens_.size(); ++t) {
            if (prompt_tokens_[t] == vision_config_.image_token_id) {
                require(img_idx < num_img_tokens, "vision_image_token_mismatch");
                std::copy_n(
                    image_embeds.data() + img_idx * h, h, prompt_embeds.data() + t * h
                );
                ++img_idx;
            }
        }
        require(img_idx == num_img_tokens, "vision_image_token_count_mismatch");

        lfm3::BackboneState state;
        state.reset(text_config_);
        std::vector<float> last_hidden(h, 0.0f);
        progress_("prefill", prompt_tokens_.size());
        {
            InferenceTrace tracing(request.trace, "prefill");
            for (std::size_t offset = 0; offset < prompt_tokens_.size();
                 offset += kPrefillChunkTokens) {
                check_cancel(cancel);
                const std::size_t chunk =
                    std::min(kPrefillChunkTokens, prompt_tokens_.size() - offset);
                std::vector<float> out(chunk * h);
                backbone_->forward(
                    std::span<const float>(prompt_embeds.data() + offset * h, chunk * h),
                    chunk,
                    out,
                    &state,
                    nullptr,
                    cancel,
                    kPrefillChunkTokens
                );
                if (offset + chunk == prompt_tokens_.size())
                    std::copy_n(out.data() + (chunk - 1) * h, h, last_hidden.data());
            }
        }

        lfm3::SamplingConfig sampling;
        sampling.do_sample = true;
        sampling.temperature = 0.1f;
        sampling.top_k = 50;
        sampling.top_p = 1.0f;
        sampling.repetition_penalty = 1.1f;
        std::uint64_t rng_state = 0x517cc1b727220a95ULL ^ prompt_tokens_.size();

        std::vector<std::uint32_t> eos_ids;
        if (text_config_.eos_token_id)
            eos_ids.push_back(*text_config_.eos_token_id);
        if (im_end_id_
            && std::find(eos_ids.begin(), eos_ids.end(), *im_end_id_) == eos_ids.end())
            eos_ids.push_back(*im_end_id_);
        auto is_eos = [&](std::uint32_t tok) {
            return std::find(eos_ids.begin(), eos_ids.end(), tok) != eos_ids.end();
        };

        std::vector<std::uint32_t> generated;
        std::vector<std::uint32_t> history = prompt_tokens_;
        std::vector<float> logits(v);
        std::vector<float> step_embed(h);
        std::vector<float> step_out(h);

        {
            InferenceTrace tracing(request.trace, "decode");
            for (std::size_t step = 0; step < output_budget_; ++step) {
                check_cancel(cancel);
                backbone_->lm_head(last_hidden, 1, logits, cancel);
                const std::uint32_t next_tok =
                    lfm3::sample_logits(logits, sampling, history, rng_state);
                generated.push_back(next_tok);
                history.push_back(next_tok);
                progress_("decode", generated.size());
                if (is_eos(next_tok))
                    break;
                if (step + 1 < output_budget_) {
                    std::uint32_t one[1] = {next_tok};
                    backbone_->embed_tokens(one, step_embed, cancel);
                    backbone_->forward(
                        step_embed, 1, step_out, &state, nullptr, cancel, 1
                    );
                    last_hidden.swap(step_out);
                }
            }
        }

        preparation_check(
            !generated.empty() && is_eos(generated.back()),
            "The vision model ended without a terminal EOS; the observation is incomplete.",
            502
        );
        std::string raw_decoded;
        for (std::uint32_t tok : generated)
            raw_decoded += tokenizer_->decode(tok, false);
        const std::string text = lfm3::extract_lfm3_vision_description(raw_decoded);

        prompt_tokens_.clear();
        prompt_tokens_.shrink_to_fit();
        patches_ = {};
        request_memory_.reset();
        if (compute_)
            compute_->idle();
        return {{"model", request.model}, {"text", text}};
    }

    void close() override {
        prompt_tokens_.clear();
        prompt_tokens_.shrink_to_fit();
        patches_ = {};
        request_memory_.reset();
        tokenizer_.reset();
        backbone_.reset();
        vision_encoder_.reset();
        shard_.reset();
        if (compute_) {
            compute_->park();
            compute_.reset();
        }
    }

private:
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::unique_ptr<WorkerCompute> compute_;
    lfm3::VisionConfig vision_config_;
    lfm3::BackboneConfig text_config_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::unique_ptr<lfm3::VisionEncoder> vision_encoder_;
    std::unique_ptr<lfm3::Backbone> backbone_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::optional<std::uint32_t> im_end_id_;
    std::unique_ptr<HostAllocation> request_memory_;
    lfm3::VisionPatches patches_;
    std::vector<std::uint32_t> prompt_tokens_;
    std::size_t output_budget_ = 0;
};

}  // namespace

std::unique_ptr<InferenceEngine>
lfm3_vision_engine(std::shared_ptr<Resources> resources, Progress progress) {
    return std::make_unique<Lfm3VisionEngine>(std::move(resources), std::move(progress));
}

}  // namespace kadan::serving
