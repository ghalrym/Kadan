#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include "kadan/chat_prompt.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"
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
        model_ = std::make_unique<text::Engine>(
            root.c_str(), layout, plan, resources_, &cancel
        );
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

    void prepare(
        const InferenceRequest& request,
        const Footprint& available,
        const std::atomic_bool& cancel
    ) override {
        check_cancel(cancel);
        if (request.operation == Operation::load)
            return;
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

        // Avoid false prefix hits when two different images have identical dimensions.
        bool reuse = input.reuse_prefix;
        if (reuse) {
            const bool images_match =
                cached_image_hashes_.size() <= scheduled_images_.size()
                && std::equal(
                    cached_image_hashes_.begin(),
                    cached_image_hashes_.end(),
                    scheduled_images_.begin(),
                    [](std::uint64_t cached, const ScheduledImage& cur) {
                        return cached == cur.hash;
                    }
                );
            if (!images_match)
                reuse = false;
        }

        // Keep the state before the current turn as a reusable history prefix.
        std::size_t leading = 0;
        if (reuse && input.conversation_id && input.messages.size() > 1
            && scheduled_images_.empty()) {
            std::vector<std::uint32_t> tokens(capacity_);
            const auto earlier =
                std::span(input.messages).first(input.messages.size() - 1);
            const auto prefix =
                chat_messages(earlier, byte_limit, input.tools, input.reasoning_effort);
            const auto n = tokenizer_->encode(
                prefix.text, tokens, cancel, byte_limit, prefix.literal_ranges
            );
            if (n < prompt_tokens_
                && std::equal(tokens.begin(), tokens.begin() + n, prompt_.begin()))
                leading = n;
        }
        model_->checkpoint_at(leading);
        // Admission belongs before execution so fresh pressure can release/replan.
        try {
            model_->set_cancel(cancel);
            reused_tokens_ = model_->begin_request(
                prompt_, input.conversation_id.value_or(""), reuse
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
        const auto started = std::chrono::steady_clock::now();
        auto first = started, last = started;
        check_cancel(cancel);

        // Encode only images that overlap the uncached suffix.
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

        const auto suffix = std::span<const unsigned>(prompt_).subspan(reused_tokens_);
        progress_("prefill_begin", suffix.size());
        model_->set_projection_timing(bool(request.trace));
        model_->take_timing();
        Token selected;
        {
            InferenceTrace tracing(request.trace, "prefill");
            selected = model_->prefill(suffix, image_spans, progress_);
            tracing.text_metrics(phase_metrics(suffix.size(), model_->take_timing()));
        }

        InklingStreamParser parser;
        std::string text, pending, reason = "length";
        std::size_t generated = 0, steps = 0, thinking_tokens = 0;
        bool forced_close_thinking = false;

        auto emit_visible = [&](const std::string& piece) {
            if (piece.empty())
                return;
            if (text.empty())
                first = last;
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
                if (!generated)
                    first = last;

                if (const auto sp = match_special(selected.id)) {
                    flush_pending_utf8(true);
                    parser.on_special(*sp);
                } else {
                    require(
                        !tokenizer_->is_special(selected.id),
                        "chat_unexpected_control_token"
                    );
                    if (parser.in_thinking())
                        ++thinking_tokens;
                    pending += tokenizer_->decode(selected.id);
                    flush_pending_utf8(false);
                }

                // Close thinking block if caller supplied a reasoning_tokens budget.
                if (!forced_close_thinking && reasoning_tokens_.has_value()
                    && parser.in_thinking() && thinking_tokens >= *reasoning_tokens_
                    && id_end_message_ && id_message_model_ && id_content_text_
                    && generated + 4 < max_output_) {
                    forced_close_thinking = true;
                    flush_pending_utf8(true);
                    model_->step(selected.id, false);
                    model_->step(*id_end_message_, false);
                    model_->step(*id_message_model_, false);
                    selected = model_->step(*id_content_text_, true);
                    steps += 4;
                    generated += 3;
                    parser.on_special(InklingStreamParser::Special::end_message);
                    parser.on_special(InklingStreamParser::Special::message_model);
                    parser.on_special(InklingStreamParser::Special::content_text);
                    continue;
                }

                if (generated + 1 < max_output_) {
                    selected = model_->step(selected.id, true);
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

        cached_image_hashes_.clear();
        for (const auto& img : scheduled_images_)
            cached_image_hashes_.push_back(img.hash);
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
        restored_ = false;
        std::vector<std::uint32_t>().swap(prompt_);
        request_memory_.reset();
        const auto milliseconds = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        return present_chat(
            request,
            Json{
                {"text", text},
                {"tool_calls", std::move(calls)},
                {"finish_reason", reason},
                {"cache", std::move(cache)},
                {"timing",
                 {{"generation_ttft_ms",
                   generated ? Json(milliseconds(first - started)) : Json(nullptr)},
                  {"prefill_ms",
                   generated ? Json(milliseconds(first - started)) : Json(nullptr)},
                  {"output_tokens", generated},
                  {"prefill_tokens", prompt_tokens_ - reused_tokens_},
                  {"decode_tokens_per_second",
                   generated > 1 && last > first
                       ? Json((generated - 1) * 1000.0 / milliseconds(last - first))
                       : Json(nullptr)}}}
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
        cached_image_hashes_.clear();
        restored_ = false;
    }

    void close() override {
        std::vector<std::uint32_t>().swap(prompt_);
        scheduled_images_.clear();
        cached_image_hashes_.clear();
        request_memory_.reset();
        tokenizer_.reset();
        if (model_)
            model_->close();
        model_.reset();
    }

private:
    struct ScheduledImage {
        std::size_t offset = 0;
        std::size_t tokens = 0;
        std::uint64_t hash = 0;
        std::vector<std::uint8_t> png;
    };

    bool restored_ = false;
    WeightIdentity tokenizer_identity_;
    std::size_t prompt_tokens_ = 0, reused_tokens_ = 0;
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::function<void(const std::string&)> content_;
    std::unique_ptr<text::Engine> model_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<std::uint32_t> prompt_;
    std::vector<ScheduledImage> scheduled_images_;
    std::vector<std::uint64_t> cached_image_hashes_;
    std::size_t capacity_ = 65536, max_output_ = 65536;
    std::optional<std::size_t> reasoning_tokens_;

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
