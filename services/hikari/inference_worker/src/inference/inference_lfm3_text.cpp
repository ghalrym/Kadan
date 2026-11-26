#include "kadan/inference_engines.hpp"

#include <algorithm>
#include <chrono>
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

constexpr std::size_t kLfm3MaxContext = 262144;
constexpr std::size_t kPrefillChunkTokens = 512;

ComputePlan lfm3_plan(const Footprint& available) {
#ifdef KADAN_WORKER_CUDA
    if (available.size() > 1)
        return media_plan(available);
#endif
    return ComputePlan{available, {}};
}

class Lfm3TextEngine final : public InferenceEngine {
public:
    Lfm3TextEngine(
        std::shared_ptr<Resources> resources,
        Progress progress,
        std::function<void(const std::string&)> content
    )
        : resources_(std::move(resources)), progress_(std::move(progress)),
          content_(std::move(content)) {
    }

    void load(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto plan = lfm3_plan(available);
        compute_ = std::make_unique<WorkerCompute>(plan, Workload::lfm3_llm, resources_);
        const auto config_doc =
            read_document(request.configuration.checkpoint / "config.json");
        config_ = lfm3::BackboneConfig::parse(config_doc);
        auto budget = std::make_shared<checkpoint::MemoryBudget>(64ULL * 1024 * 1024);
        auto shard = std::make_shared<checkpoint::Shard>(
            request.configuration.checkpoint.string().c_str(),
            "model.safetensors",
            std::move(budget)
        );
        backbone_ = std::make_unique<lfm3::Backbone>(
            config_,
            shard,
            "model.",
            resources_,
            Workload::lfm3_llm,
            compute_->compute,
            true,
            true
        );
        shard_ = std::move(shard);
        tokenizer_ = std::make_unique<text::Tokenizer>(resources_, Workload::lfm3_llm);
        tokenizer_->load(
            (request.configuration.checkpoint / "tokenizer.json").string(), cancel
        );
        end_think_id_ = tokenizer_->find_token("</think>");
        im_end_id_ = tokenizer_->find_token("<|im_end|>");
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        const auto& input = std::get<ChatRequest>(request.input);
        const std::size_t context_limit = request.configuration.context_limit > 0
                                              ? request.configuration.context_limit
                                              : kLfm3MaxContext;
        preparation_check(
            context_limit >= 1 && context_limit <= kLfm3MaxContext,
            "LFM3 context must be between 1 and 262144 tokens.",
            422
        );

        std::vector<lfm3::Lfm3ChatMessage> messages = input.lfm3_messages;
        if (messages.empty()) {
            for (const auto& msg : input.messages)
                messages.push_back({msg.role, msg.text, Json::array(), ""});
        }
        const std::string prompt_text =
            lfm3::render_lfm3_chat_prompt(messages, input.lfm3_tools, true, false);
        prompt_tokens_.assign(context_limit + 1, 0);
        const std::size_t count = tokenizer_->encode(
            prompt_text, prompt_tokens_, cancel, std::max<std::size_t>(prompt_text.size() + 64, 4 * 1024 * 1024)
        );
        prompt_tokens_.resize(count);
        preparation_check(
            count < context_limit,
            "LFM3 prompt leaves no reply space in the supported context; history was not truncated.",
            422
        );
        const std::size_t remaining = context_limit - count;
        output_budget_ = input.max_tokens ? std::min(remaining, *input.max_tokens) : remaining;
        preparation_check(output_budget_ >= 1, "LFM3 token limits must be positive integers.", 422);

        const Bytes scratch_bytes =
            32ULL * 1024 * 1024 + count * sizeof(std::uint32_t);
        request_memory_ = std::make_unique<HostAllocation>(
            resources_, Workload::lfm3_llm, scratch_bytes
        );
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        const auto& input = std::get<ChatRequest>(request.input);
        const auto started = std::chrono::steady_clock::now();
        lfm3::BackboneState state;
        state.reset(config_);

        const std::size_t h = config_.hidden_size;
        const std::size_t v = config_.vocab_size;

        progress_("prefill", prompt_tokens_.size());
        std::vector<float> last_hidden(h, 0.0f);
        {
            InferenceTrace tracing(request.trace, "prefill");
            for (std::size_t offset = 0; offset < prompt_tokens_.size();
                 offset += kPrefillChunkTokens) {
                check_cancel(cancel);
                const std::size_t chunk =
                    std::min(kPrefillChunkTokens, prompt_tokens_.size() - offset);
                std::vector<float> embeds(chunk * h);
                backbone_->embed_tokens(
                    std::span<const std::uint32_t>(prompt_tokens_.data() + offset, chunk),
                    embeds,
                    cancel
                );
                std::vector<float> out(chunk * h);
                backbone_->forward(
                    embeds, chunk, out, &state, nullptr, cancel, kPrefillChunkTokens
                );
                if (offset + chunk == prompt_tokens_.size()) {
                    std::copy_n(out.data() + (chunk - 1) * h, h, last_hidden.data());
                }
            }
        }

        lfm3::SamplingConfig sampling;
        sampling.do_sample = true;
        sampling.temperature = 0.1f;
        sampling.top_k = 50;
        sampling.top_p = 1.0f;
        sampling.repetition_penalty = 1.1f;
        std::uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^ prompt_tokens_.size();

        lfm3::ReasoningBudget reasoning;
        if (input.reasoning_tokens && end_think_id_) {
            reasoning.active = true;
            reasoning.max_reasoning_tokens = *input.reasoning_tokens;
            reasoning.end_think_token = *end_think_id_;
        }

        std::vector<std::uint32_t> eos_ids;
        if (config_.eos_token_id)
            eos_ids.push_back(*config_.eos_token_id);
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
                std::optional<std::uint32_t> prev =
                    generated.empty() ? std::nullopt : std::optional(generated.back());
                reasoning.apply(generated.size(), prev, logits);
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

        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
        const std::string finish =
            (!generated.empty() && is_eos(generated.back())) ? "stop" : "length";

        std::string raw_decoded;
        for (std::uint32_t tok : generated)
            raw_decoded += tokenizer_->decode(tok, false);
        const std::string reply = lfm3::extract_lfm3_reply_text(raw_decoded);
        if (input.stream && !reply.empty() && content_)
            content_(reply);

        const std::size_t prefill_count = prompt_tokens_.size();
        prompt_tokens_.clear();
        prompt_tokens_.shrink_to_fit();
        request_memory_.reset();
        if (compute_)
            compute_->idle();

        return {
            {"model", request.model},
            {"text", reply},
            {"finish_reason", finish},
            {"timing",
             {{"output_tokens", generated.size()},
              {"prefill_tokens", prefill_count},
              {"generation_seconds", seconds}}}
        };
    }

    void close() override {
        prompt_tokens_.clear();
        prompt_tokens_.shrink_to_fit();
        request_memory_.reset();
        tokenizer_.reset();
        backbone_.reset();
        shard_.reset();
        if (compute_) {
            compute_->park();
            compute_.reset();
        }
    }

private:
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::function<void(const std::string&)> content_;
    std::unique_ptr<WorkerCompute> compute_;
    lfm3::BackboneConfig config_;
    std::shared_ptr<checkpoint::Shard> shard_;
    std::unique_ptr<lfm3::Backbone> backbone_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::optional<std::uint32_t> end_think_id_;
    std::optional<std::uint32_t> im_end_id_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<std::uint32_t> prompt_tokens_;
    std::size_t output_budget_ = 0;
};

}  // namespace

std::unique_ptr<InferenceEngine> lfm3_text_engine(
    std::shared_ptr<Resources> resources,
    Progress progress,
    std::function<void(const std::string&)> content
) {
    return std::make_unique<Lfm3TextEngine>(
        std::move(resources), std::move(progress), std::move(content)
    );
}

}  // namespace kadan::serving
