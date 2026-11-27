#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "kadan/chat_prompt.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/inkling_variant.hpp"
#include "kadan/preparation.hpp"
#include "kadan/text_decision.hpp"
#include "kadan/text_sampling.hpp"
#include "kadan/text_tokenizer.hpp"
#include "kadan/utf8_stream.hpp"

namespace kadan::serving {
namespace {
TextPhaseMetrics phase_metrics(std::size_t tokens, const text::Engine::Timing& t) {
    return {
        tokens,
        t.projection_seconds,
        t.attention_seconds,
        t.prepare_seconds,
        t.device_seconds,
        t.host_wait_seconds,
        t.recurrent_seconds,
        t.projections,
        t.batches,
        t.streamed_bytes,
        t.host_bytes,
        t.packed_bytes,
        t.device_bytes,
        t.device_slices,
        t.ram_bytes,
        t.ram_slices,
        t.projection_detail.packed_calls,
        t.projection_detail.host_calls,
        t.projection_detail.large_row_calls,
        t.projection_detail.unsupported_calls,
        t.projection_detail.rows,
        t.projection_detail.batch_calls,
        t.projection_detail.batch_rejections,
        t.projection_detail.large_row_seconds,
        t.projection_detail.unsupported_seconds
    };
}

std::uint64_t hash_image(std::size_t offset, std::span<const std::uint8_t> bytes) {
    std::uint64_t h = 14695981039346656037ULL ^ std::uint64_t(offset);
    h *= 1099511628211ULL;
    for (auto b : bytes) {
        h ^= b;
        h *= 1099511628211ULL;
    }
    return h;
}

struct ChatCache final : InferenceCache {
    std::unique_ptr<text::PrefixSnapshot> prefix;
    WeightIdentity tokenizer;

    Bytes bytes() const override {
        return sizeof(*this) + prefix->bytes();
    }
};

class ChatEngine final : public InferenceEngine {
public:
    ChatEngine(
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
        const auto root = request.configuration.checkpoint.string();
        capacity_ = request.configuration.context_limit;
        require(capacity_ > 0 && capacity_ <= 1048576, "chat_context_limit");
        if (available[0] < 2ULL * 1024 * 1024 * 1024)
            throw AdmissionWait("chat_host_headroom");
        const auto plan = media_plan(available);
        auto layout = text::plan(root.c_str(), capacity_, &cancel);
        constexpr Bytes headroom = 4ULL * 1024 * 1024 * 1024;
        if (layout.host + headroom > available[0])
            throw AdmissionWait("text_host_headroom");
        // Remaining host memory holds weight slices that do not fit in VRAM.
        layout.cache =
            std::min<Bytes>(layout.packed, available[0] - layout.host - headroom);
        const auto variant = inkling_model_variant(request.model);
        const std::string initial_variant =
            variant ? std::string(inkling_variant_name(*variant)) : std::string();
        model_ = std::make_unique<text::Engine>(
            root.c_str(), layout, plan, resources_, &cancel, initial_variant
        );
        model_id_ = request.model;
        checkpoint_ = request.configuration.checkpoint;
        last_weight_swap_ms_ = 0.0;
        tokenizer_ = std::make_unique<text::Tokenizer>(resources_, Workload::llm);
        tokenizer_identity_ =
            checkpoint::GgufMetadata::file_identity(root + "/tokenizer.json");
        tokenizer_->load(root + "/tokenizer.json", cancel);

        id_message_model_ = tokenizer_->added_token_id("<|message_model|>");
        id_content_thinking_ = tokenizer_->added_token_id("<|content_thinking|>");
        id_content_text_ = tokenizer_->added_token_id("<|content_text|>");
        id_content_invoke_tool_json_ =
            tokenizer_->added_token_id("<|content_invoke_tool_json|>");
        id_end_message_ = tokenizer_->added_token_id("<|end_message|>");
        id_image_placeholder_ = tokenizer_->added_token_id("<|unused_200054|>");
    }

    bool try_hot_swap(
        const InferenceRequest& request,
        const Footprint&,
        const std::atomic_bool& cancel
    ) override {
        if (!model_ || request.configuration.context_limit != capacity_)
            return false;
        if (request.model == model_id_
            && request.configuration.checkpoint == checkpoint_) {
            last_weight_swap_ms_ = 0.0;
            return true;
        }
        if (!inkling_hot_swap_compatible(model_id_, request.model))
            return false;
        const auto target_variant = inkling_model_variant(request.model);
        if (!target_variant)
            return false;
        const auto target_root = request.configuration.checkpoint;
        const auto old_tok = checkpoint_ / "tokenizer.json";
        const auto new_tok = target_root / "tokenizer.json";
        if (old_tok != new_tok) {
            constexpr std::size_t max_tokenizer_bytes = 64ULL * 1024 * 1024;
            if (read_bytes(old_tok, max_tokenizer_bytes)
                != read_bytes(new_tok, max_tokenizer_bytes))
                return false;
        }
        model_->set_cancel(cancel);
        if (!model_->switch_variant(
                target_root.string().c_str(),
                inkling_variant_name(*target_variant),
                cancel
            ))
            return false;
        model_id_ = request.model;
        checkpoint_ = target_root;
        tokenizer_identity_ = checkpoint::GgufMetadata::file_identity(new_tok.string());
        last_weight_swap_ms_ = model_->last_weight_swap_ms();
        restored_ = false;
        return true;
    }

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        if (request.operation == Operation::load)
            return;
        if (std::holds_alternative<DecisionRequest>(request.input)) {
            prepare_decisions(std::get<DecisionRequest>(request.input), available, cancel);
            return;
        }
        prepared_decisions_.clear();
        const auto& input = std::get<ChatRequest>(request.input);
        const auto byte_limit = capacity_ * 1024ULL;
        const auto rendered = chat_prompt(
            input.messages, byte_limit, input.tools, input.reasoning_effort
        );
        const std::size_t input_bytes = rendered.text.size();
        require(
            input_bytes > 0 && input_bytes <= byte_limit, "chat_input_transport_bound"
        );
        // The prompt and its leading messages' tokens.
        const Bytes scratch =
            8 * 1024 * 1024 + 8 * input_bytes + 2 * capacity_ * sizeof(std::uint32_t);
        if (available[0]
            < scratch + std::max<Bytes>(32 * 1024 * 1024, input_bytes * 1024ULL))
            throw AdmissionWait("chat_request_headroom");
        request_memory_ =
            std::make_unique<HostAllocation>(resources_, Workload::llm, scratch);

        std::vector<std::uint32_t> raw_tokens(capacity_);
        const auto raw_count = tokenizer_->encode(
            rendered.text, raw_tokens, cancel, byte_limit, rendered.literal_ranges
        );
        raw_tokens.resize(raw_count);

        // Expand each <|unused_200054|> placeholder into N_img placeholder tokens
        // matching the exact Inkling 2 vision encoder output count.
        scheduled_images_.clear();
        std::vector<const std::vector<std::uint8_t>*> message_images;
        for (const auto& msg : input.messages)
            for (const auto& png : msg.decoded_images)
                message_images.push_back(&png);

        if (message_images.empty()) {
            prompt_ = std::move(raw_tokens);
        } else {
            require(
                id_image_placeholder_.has_value(), "chat_missing_image_placeholder"
            );
            const auto placeholder = *id_image_placeholder_;
            prompt_.clear();
            prompt_.reserve(raw_count + message_images.size() * 256);
            std::size_t img_index = 0;
            for (const auto id : raw_tokens) {
                if (id == placeholder) {
                    require(
                        img_index < message_images.size(),
                        "chat_image_placeholder_count"
                    );
                    const auto& png = *message_images[img_index++];
                    const auto n_img = text::image_token_count(png);
                    require(
                        prompt_.size() + n_img <= capacity_, "chat_context_exceeded"
                    );
                    scheduled_images_.push_back(
                        {prompt_.size(), n_img, hash_image(prompt_.size(), png), png}
                    );
                    prompt_.insert(prompt_.end(), n_img, placeholder);
                } else {
                    require(prompt_.size() < capacity_, "chat_context_exceeded");
                    prompt_.push_back(id);
                }
            }
            require(img_index == message_images.size(), "chat_image_placeholder_count");
        }

        prompt_tokens_ = prompt_.size();
        if (input.max_output_tokens.has_value()) {
            max_output_ = *input.max_output_tokens;
            require(
                max_output_ > 0 && prompt_tokens_ + max_output_ <= capacity_,
                "chat_context_exceeded"
            );
        } else {
            require(prompt_tokens_ < capacity_, "chat_context_exceeded");
            max_output_ = capacity_ - prompt_tokens_;
        }
        reasoning_tokens_ = input.reasoning_tokens;
        sampling_ = text::SamplingConfig{};
        if (input.temperature.has_value())
            sampling_.temperature = *input.temperature;
        if (input.top_p.has_value())
            sampling_.top_p = *input.top_p;
        if (input.top_k.has_value())
            sampling_.top_k = *input.top_k;
        if (input.min_p.has_value())
            sampling_.min_p = *input.min_p;
        if (input.repetition_penalty.has_value())
            sampling_.repetition_penalty = *input.repetition_penalty;
        if (input.presence_penalty.has_value())
            sampling_.presence_penalty = *input.presence_penalty;
        sampling_.seed = input.seed;

        std::vector<text::ImageTag> prompt_images;
        prompt_images.reserve(scheduled_images_.size());
        for (const auto& img : scheduled_images_)
            prompt_images.push_back({img.offset, img.tokens, img.hash});

        const bool reuse = input.reuse_prefix;
        // Static prefix boundary: tool_declare + leading system messages + effort.
        // Safe to share across fresh conversations with identical leading tokens.
        std::size_t static_boundary = 0;
        std::size_t system_count = 0;
        while (system_count < input.messages.size()
               && input.messages[system_count].role == "system")
            ++system_count;
        if (reuse && system_count > 0 && system_count < input.messages.size()) {
            std::vector<std::uint32_t> tokens(capacity_);
            const auto prefix = chat_messages(
                std::span(input.messages).first(system_count),
                byte_limit,
                input.tools,
                input.reasoning_effort,
                true
            );
            const auto n = tokenizer_->encode(
                prefix.text, tokens, cancel, byte_limit, prefix.literal_ranges
            );
            if (n > 0 && n < prompt_tokens_
                && std::equal(tokens.begin(), tokens.begin() + n, prompt_.begin())
                && text::images_match_prefix({}, prompt_images, n))
                static_boundary = n;
        }

        // Keep the state before the current turn as a reusable history prefix.
        std::size_t leading = 0;
        if (reuse && input.conversation_id && input.messages.size() > 1) {
            const auto earlier =
                std::span(input.messages).first(input.messages.size() - 1);
            const bool earlier_has_images = std::any_of(
                earlier.begin(),
                earlier.end(),
                [](const ChatMessage& m) { return !m.decoded_images.empty(); }
            );
            if (!earlier_has_images) {
                const bool all_system = std::all_of(
                    earlier.begin(),
                    earlier.end(),
                    [](const ChatMessage& m) { return m.role == "system"; }
                );
                std::vector<std::uint32_t> tokens(capacity_);
                const auto prefix = chat_messages(
                    earlier,
                    byte_limit,
                    input.tools,
                    input.reasoning_effort,
                    all_system
                );
                const auto n = tokenizer_->encode(
                    prefix.text, tokens, cancel, byte_limit, prefix.literal_ranges
                );
                if (n > 0 && n < prompt_tokens_ && n != static_boundary
                    && std::equal(tokens.begin(), tokens.begin() + n, prompt_.begin())
                    && text::images_match_prefix(prompt_images, prompt_images, n))
                    leading = n;
            }
        }
        model_->checkpoint_static_at(static_boundary);
        model_->checkpoint_at(leading);
        history_start_ = static_boundary;
        // Admission belongs before execution so fresh pressure can release/replan.
        try {
            model_->set_cancel(cancel);
            reused_tokens_ = model_->begin_request(
                prompt_, input.conversation_id.value_or(""), reuse, prompt_images
            );
        } catch (const std::exception& error) {
            const std::string reason = error.what();
            if (reason == "exhausted" || reason == "model_physical_headroom")
                throw AdmissionWait(reason);
            throw;
        }
    }

    Json
    execute(const InferenceRequest& request, const std::atomic_bool& cancel) override {
        if (request.operation == Operation::load)
            return Json::object();
        if (std::holds_alternative<DecisionRequest>(request.input))
            return execute_decisions(request, cancel);
        const auto milliseconds = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        const auto started = std::chrono::steady_clock::now();
        auto last = started;
        std::optional<std::chrono::steady_clock::time_point> first_token_at;
        std::optional<std::chrono::steady_clock::time_point> first_visible_at;
        std::optional<std::chrono::steady_clock::time_point> thinking_started_at;
        std::optional<std::chrono::steady_clock::time_point> thinking_ended_at;
        check_cancel(cancel);

        // Encode only images that overlap the uncached suffix.
        const auto image_started = std::chrono::steady_clock::now();
        std::vector<text::ImageSpan> image_spans;
        for (const auto& img : scheduled_images_) {
            if (img.offset + img.tokens <= reused_tokens_)
                continue;
            check_cancel(cancel);
            auto span = model_->encode_image(img.png);
            require(span.tokens == img.tokens, "chat_image_token_mismatch");
            span.offset = img.offset;
            image_spans.push_back(std::move(span));
        }
        const auto image_done = std::chrono::steady_clock::now();
        const double image_encode_ms =
            image_spans.empty() ? 0.0 : milliseconds(image_done - image_started);

        std::uint64_t rng_state = 0;
        if (sampling_.seed.has_value()) {
            rng_state = text::splitmix64(*sampling_.seed);
        } else {
            std::random_device rd;
            const auto now_ns = static_cast<std::uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count()
            );
            const auto id_hash =
                static_cast<std::uint64_t>(std::hash<std::string>{}(request.id));
            rng_state = text::splitmix64(
                (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd())
                ^ now_ns ^ id_hash
            );
        }

        std::vector<unsigned> penalty_history;
        auto record_penalty_token = [&](unsigned id) {
            if (!sampling_.has_penalties() || tokenizer_->is_special(id))
                return;
            penalty_history.push_back(id);
            const std::size_t cap = std::max<std::size_t>(sampling_.penalty_window, 256) * 2;
            if (penalty_history.size() > cap) {
                const std::size_t keep = std::max<std::size_t>(sampling_.penalty_window, 256);
                penalty_history.erase(
                    penalty_history.begin(),
                    penalty_history.end() - static_cast<std::ptrdiff_t>(keep)
                );
            }
        };
        if (sampling_.has_penalties()) {
            const std::size_t start =
                history_start_ < prompt_.size() ? history_start_ : 0;
            for (std::size_t i = start; i < prompt_.size(); ++i)
                record_penalty_token(prompt_[i]);
        }

        auto sample_from = [&](std::vector<float>& logits) -> Token {
            const unsigned chosen =
                text::sample_logits(logits, sampling_, penalty_history, rng_state);
            return {chosen, model_->eos(chosen), model_->committed_tokens()};
        };

        const auto suffix = std::span<const unsigned>(prompt_).subspan(reused_tokens_);
        progress_("prefill_begin", suffix.size());
        model_->set_projection_timing(bool(request.trace));
        model_->take_timing();
        const auto prefill_started = std::chrono::steady_clock::now();
        Token selected;
        {
            InferenceTrace tracing(request.trace, "prefill");
            if (sampling_.is_greedy()) {
                selected = model_->prefill(suffix, image_spans, progress_);
            } else {
                auto logits = model_->prefill_logits(suffix, image_spans, progress_);
                selected = sample_from(logits);
            }
            tracing.text_metrics(phase_metrics(suffix.size(), model_->take_timing()));
        }
        const auto prefill_done = std::chrono::steady_clock::now();
        const double prefill_ms = milliseconds(prefill_done - prefill_started);

        InklingStreamParser parser;
        std::string text, pending, reason = "length";
        std::size_t generated = 0, steps = 0, thinking_tokens = 0;
        bool forced_close_thinking = false;

        auto emit_visible = [&](const std::string& piece) {
            if (piece.empty())
                return;
            if (!first_visible_at)
                first_visible_at = last;
            content_(piece);
            text += piece;
        };

        auto flush_pending_utf8 = [&](bool final_chunk) {
            auto chunk = utf8_chunk(pending, final_chunk);
            if (!chunk.empty())
                emit_visible(parser.on_text(chunk));
        };

        auto match_special =
            [&](unsigned id) -> std::optional<InklingStreamParser::Special> {
            if (id_message_model_ && id == *id_message_model_)
                return InklingStreamParser::Special::message_model;
            if (id_content_thinking_ && id == *id_content_thinking_)
                return InklingStreamParser::Special::content_thinking;
            if (id_content_text_ && id == *id_content_text_)
                return InklingStreamParser::Special::content_text;
            if (id_content_invoke_tool_json_ && id == *id_content_invoke_tool_json_)
                return InklingStreamParser::Special::content_invoke_tool_json;
            if (id_end_message_ && id == *id_end_message_)
                return InklingStreamParser::Special::end_message;
            return std::nullopt;
        };

        {
            InferenceTrace tracing(request.trace, "decode");
            for (; generated < max_output_; ++generated) {
                check_cancel(cancel);
                if (selected.eos) {
                    reason = "stop";
                    break;
                }
                progress_("decode_token", generated + 1);
                last = std::chrono::steady_clock::now();
                if (!first_token_at)
                    first_token_at = last;

                if (const auto sp = match_special(selected.id)) {
                    flush_pending_utf8(true);
                    const bool was_thinking = parser.in_thinking();
                    parser.on_special(*sp);
                    if (*sp == InklingStreamParser::Special::content_thinking) {
                        if (!thinking_started_at)
                            thinking_started_at = last;
                        thinking_ended_at = last;
                    } else if (was_thinking && !parser.in_thinking()) {
                        thinking_ended_at = last;
                    }
                    if (*sp == InklingStreamParser::Special::content_invoke_tool_json
                        && !first_visible_at) {
                        first_visible_at = last;
                    }
                } else {
                    require(
                        !tokenizer_->is_special(selected.id),
                        "chat_unexpected_control_token"
                    );
                    if (!parser.in_tool_json())
                        record_penalty_token(selected.id);
                    if (parser.in_thinking()) {
                        ++thinking_tokens;
                        if (!thinking_started_at)
                            thinking_started_at = last;
                        thinking_ended_at = last;
                    }
                    pending += tokenizer_->decode(selected.id);
                    flush_pending_utf8(false);
                }

                auto next_token = [&](unsigned input_token) -> Token {
                    // Use greedy decoding inside tool JSON so sampling/penalties never
                    // corrupt structured tool names, keys, or argument values.
                    if (sampling_.is_greedy() || parser.in_tool_json())
                        return model_->step(input_token, true);
                    auto logits = model_->step_logits(input_token);
                    return sample_from(logits);
                };

                // Close thinking block if caller supplied a reasoning_tokens budget.
                if (!forced_close_thinking && reasoning_tokens_.has_value()
                    && parser.in_thinking() && thinking_tokens >= *reasoning_tokens_
                    && id_end_message_ && id_message_model_ && id_content_text_
                    && generated + 4 < max_output_) {
                    forced_close_thinking = true;
                    thinking_ended_at = last;
                    flush_pending_utf8(true);
                    model_->step(selected.id, false);
                    model_->step(*id_end_message_, false);
                    model_->step(*id_message_model_, false);
                    parser.on_special(InklingStreamParser::Special::end_message);
                    parser.on_special(InklingStreamParser::Special::message_model);
                    parser.on_special(InklingStreamParser::Special::content_text);
                    selected = next_token(*id_content_text_);
                    steps += 4;
                    generated += 3;
                    continue;
                }

                if (generated + 1 < max_output_) {
                    selected = next_token(selected.id);
                    ++steps;
                }
            }
            tracing.text_metrics(phase_metrics(steps, model_->take_timing()));
        }

        flush_pending_utf8(true);
        auto calls = parser.finish(request.id, reason != "length");
        if (!calls.empty())
            reason = "tool_calls";

        model_->end_request();
        scheduled_images_.clear();

        Json cache = nullptr;
        const auto& input = std::get<ChatRequest>(request.input);
        if (input.conversation_id) {
            const auto memory = model_->prefix_memory();
            Json devices = Json::object();
            for (std::size_t i = 1; i < memory.size(); ++i)
                if (memory[i])
                    devices["cuda:" + std::to_string(i - 1)] = memory[i];
            cache = {
                {"hit", reused_tokens_ != 0},
                {"reused_tokens", reused_tokens_},
                {"stored_tokens", model_->stored_prefix_tokens()},
                {"host_bytes", memory[0]},
                {"device_bytes", std::move(devices)},
                {"reason",
                 reused_tokens_ && restored_ ? "ram_hit"
                                             : std::string(model_->prefix_reason())},
                {"retention_reason",
                 input.reuse_prefix ? "resident_engine" : "disabled"}
            };
        }
        const double cache_restore_ms = model_->last_cache_restore_ms();
        const double weight_swap_ms = std::exchange(last_weight_swap_ms_, 0.0);
        restored_ = false;
        std::vector<std::uint32_t>().swap(prompt_);
        request_memory_.reset();

        const double reasoning_ms =
            thinking_started_at && thinking_ended_at
                ? milliseconds(*thinking_ended_at - *thinking_started_at)
                : 0.0;
        const auto ttft_point = first_visible_at ? first_visible_at : first_token_at;
        const Json decode_tps =
            first_token_at && generated > 1 && last > *first_token_at
                ? Json((generated - 1) * 1000.0 / milliseconds(last - *first_token_at))
                : Json(nullptr);
        return present_chat(
            request,
            Json{
                {"text", text},
                {"tool_calls", std::move(calls)},
                {"finish_reason", reason},
                {"cache", std::move(cache)},
                {"timing",
                 {{"generation_ttft_ms",
                   ttft_point ? Json(milliseconds(*ttft_point - started))
                              : Json(nullptr)},
                  {"first_token_ms",
                   first_token_at ? Json(milliseconds(*first_token_at - started))
                                  : Json(nullptr)},
                  {"prefill_ms", Json(prefill_ms)},
                  {"image_encode_ms", Json(image_encode_ms)},
                  {"cache_restore_ms", Json(cache_restore_ms)},
                  {"weight_swap_ms", Json(weight_swap_ms)},
                  {"reasoning_ms", Json(reasoning_ms)},
                  {"thinking_tokens", thinking_tokens},
                  {"output_tokens", generated},
                  {"prefill_tokens", prompt_tokens_ - reused_tokens_},
                  {"decode_tokens_per_second", decode_tps}}}
            }
        );
    }

    std::unique_ptr<InferenceCache> take_cache() override {
        if (!model_)
            return {};
        auto prefix = model_->take_prefix();
        if (!prefix)
            return {};
        auto cache = std::make_unique<ChatCache>();
        cache->prefix = std::move(prefix);
        cache->tokenizer = tokenizer_identity_;
        return cache;
    }

    void restore_cache(std::unique_ptr<InferenceCache> state) override {
        auto* cache = dynamic_cast<ChatCache*>(state.get());
        if (cache && cache->tokenizer == tokenizer_identity_)
            restored_ = model_->restore_prefix(std::move(cache->prefix));
    }

    void discard_cache() override {
        if (model_)
            model_->reset();
        restored_ = false;
    }

    void close() override {
        std::vector<std::uint32_t>().swap(prompt_);
        prepared_decisions_.clear();
        scheduled_images_.clear();
        request_memory_.reset();
        tokenizer_.reset();
        if (model_)
            model_->close();
        model_.reset();
    }

private:
    static constexpr std::size_t kDecisionTokenLimit = 65536;

    void prepare_decisions(
        const DecisionRequest& input,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) {
        preparation_check(
            !input.state.empty() && input.state.size() <= 8000,
            "Decision state must fit 8000 characters.",
            422
        );
        preparation_check(
            !input.questions.empty() && input.questions.size() <= 8,
            "Decisions require one to eight questions.",
            422
        );
        const Bytes scratch =
            16 * 1024 * 1024 + input.questions.size() * 256 * 1024;
        if (available[0] < scratch + 32 * 1024 * 1024)
            throw AdmissionWait("decision_request_headroom");
        request_memory_ =
            std::make_unique<HostAllocation>(resources_, Workload::llm, scratch);
        prepared_decisions_ = text::prepare_inkling_decisions(
            input.state,
            input.questions,
            *tokenizer_,
            capacity_,
            kDecisionTokenLimit,
            cancel
        );
    }

    Json execute_decisions(
        const InferenceRequest& request, const std::atomic_bool& cancel
    ) {
        const auto& input = std::get<DecisionRequest>(request.input);
        require(
            prepared_decisions_.size() == input.questions.size(),
            "decision_unprepared"
        );
        check_cancel(cancel);

        model_->set_cancel(cancel);
        const std::string decision_conv = "__decision__:" + request.id;

        // Strictly preserve the active chat conversation's KV cache across
        // decision evaluation. Restores unconditionally on success, error, or
        // cancellation so decisions never evict or corrupt live chat state.
        struct PrefixGuard {
            text::Engine* engine = nullptr;
            std::unique_ptr<text::PrefixSnapshot> saved;
            std::string transient_conv;
            std::vector<text::PreparedDecisionQuestion>* prepared = nullptr;
            std::unique_ptr<HostAllocation>* memory = nullptr;

            ~PrefixGuard() noexcept {
                try {
                    if (engine) {
                        engine->forget_conversation(transient_conv);
                        if (saved)
                            (void)engine->restore_prefix(std::move(saved));
                        else
                            engine->abort_request();
                    }
                } catch (...) {
                }
                if (prepared)
                    prepared->clear();
                if (memory)
                    memory->reset();
            }
        };

        PrefixGuard guard{
            model_.get(),
            model_->take_prefix(),
            decision_conv,
            &prepared_decisions_,
            &request_memory_
        };

        std::size_t total_tokens = 0;
        for (const auto& q : prepared_decisions_)
            total_tokens += q.tokens.size();
        progress_("evaluating", total_tokens);

        const std::size_t shared_prefix =
            text::shared_token_prefix_length(prepared_decisions_);
        const bool reuse_state_prefix = shared_prefix > 0;

        Json answers = Json::array();
        InferenceTrace tracing(request.trace, "evaluate");
        for (std::size_t i = 0; i < input.questions.size(); ++i) {
            check_cancel(cancel);
            const auto& prepared = prepared_decisions_[i];
            model_->checkpoint_static_at(0);
            model_->checkpoint_at(i == 0 ? shared_prefix : 0);
            std::size_t reused = 0;
            try {
                reused = model_->begin_request(
                    prepared.tokens, decision_conv, reuse_state_prefix
                );
            } catch (const std::exception& error) {
                const std::string reason = error.what();
                if (reason == "exhausted" || reason == "model_physical_headroom")
                    throw AdmissionWait(reason);
                throw;
            }
            const auto suffix =
                std::span<const unsigned>(prepared.tokens).subspan(reused);
            const auto logits = model_->prefill_logits(suffix, progress_);
            model_->end_request();
            answers.push_back(
                text::score_decision_logits(
                    input.questions[i], logits, prepared.readout_groups
                )
            );
        }
        last_weight_swap_ms_ = 0.0;
        return answers;
    }

    struct ScheduledImage {
        std::size_t offset = 0;
        std::size_t tokens = 0;
        std::uint64_t hash = 0;
        std::vector<std::uint8_t> png;
    };

    bool restored_ = false;
    std::string model_id_;
    fs::path checkpoint_;
    double last_weight_swap_ms_ = 0.0;
    WeightIdentity tokenizer_identity_;
    std::size_t prompt_tokens_ = 0, reused_tokens_ = 0;
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::function<void(const std::string&)> content_;
    std::unique_ptr<text::Engine> model_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<std::uint32_t> prompt_;
    std::vector<text::PreparedDecisionQuestion> prepared_decisions_;
    std::vector<ScheduledImage> scheduled_images_;
    std::size_t capacity_ = 65536, max_output_ = 65536;
    std::size_t history_start_ = 0;
    std::optional<std::size_t> reasoning_tokens_;
    text::SamplingConfig sampling_;

    std::optional<std::uint32_t> id_message_model_;
    std::optional<std::uint32_t> id_content_thinking_;
    std::optional<std::uint32_t> id_content_text_;
    std::optional<std::uint32_t> id_content_invoke_tool_json_;
    std::optional<std::uint32_t> id_end_message_;
    std::optional<std::uint32_t> id_image_placeholder_;
};
}  // namespace

std::unique_ptr<InferenceEngine> chat_engine(
    std::shared_ptr<Resources> resources,
    Progress progress,
    std::function<void(const std::string&)> content
) {
    return std::make_unique<ChatEngine>(
        std::move(resources), std::move(progress), std::move(content)
    );
}
}  // namespace kadan::serving
