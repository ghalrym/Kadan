#include <chrono>

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

class ChatEngine final : public InferenceEngine {
public:
    ChatEngine(
        std::shared_ptr<Resources> resources,
        Progress progress,
        std::function<void(const std::string&)> content,
        ModelStore& store
    )
        : resources_(std::move(resources)), progress_(std::move(progress)),
          content_(std::move(content)), store_(store) {
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
        tokenizer_->load(root + "/tokenizer.json", cancel);
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
        std::size_t input_bytes = 0;
        for (const auto& message : input.messages)
            input_bytes += message.role.size() + message.text.size();
        require(
            input_bytes > 0 && input_bytes <= byte_limit, "chat_input_transport_bound"
        );
        const Bytes scratch =
            8 * 1024 * 1024 + 8 * input_bytes + capacity_ * sizeof(std::uint32_t);
        if (available[0]
            < scratch + std::max<Bytes>(32 * 1024 * 1024, input_bytes * 1024ULL))
            throw AdmissionWait("chat_request_headroom");
        request_memory_ =
            std::make_unique<HostAllocation>(resources_, Workload::llm, scratch);
        max_output_ = input.max_output_tokens;
        require(max_output_ > 0 && max_output_ <= 1024, "chat_output_limit");
        prompt_.resize(capacity_);
        const auto count = tokenizer_->encode(
            chat_prompt(input.messages, byte_limit), prompt_, cancel, byte_limit
        );
        prompt_.resize(count);
        prompt_tokens_ = count;
        require(count + max_output_ <= capacity_, "chat_context_exceeded");
        // Admission belongs before execution so fresh pressure can release/replan.
        try {
            model_->set_cancel(cancel);
            reused_tokens_ = model_->begin_request(
                prompt_, input.conversation_id.value_or(""), input.reuse_prefix
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
        if (request.operation == Operation::load) {
            const auto& input = std::get<ChatRequest>(request.input);
            if (input.context_supplied)
                store_.set_context(
                    request.model,
                    input.context_limit ? Json(*input.context_limit) : Json(nullptr)
                );
            return Json::object();
        }
        const auto started = std::chrono::steady_clock::now();
        auto first = started, last = started;
        check_cancel(cancel);
        const auto suffix = std::span<const unsigned>(prompt_).subspan(reused_tokens_);
        progress_("prefill_begin", suffix.size());
        model_->set_projection_timing(bool(request.trace));
        model_->take_timing();
        Token selected;
        {
            InferenceTrace tracing(request.trace, "prefill");
            selected = model_->prefill(suffix, progress_);
            tracing.text_metrics(phase_metrics(suffix.size(), model_->take_timing()));
        }
        std::string text, pending, reason = "length";
        std::size_t generated = 0, steps = 0;
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
                pending += tokenizer_->decode(selected.id);
                auto chunk = utf8_chunk(pending);
                if (!chunk.empty()) {
                    content_(chunk);
                    text += chunk;
                }
                if (generated + 1 < max_output_) {
                    selected = model_->step(selected.id, true);
                    ++steps;
                }
            }
            tracing.text_metrics(phase_metrics(steps, model_->take_timing()));
        }
        auto tail = utf8_chunk(pending, true);
        if (!tail.empty()) {
            content_(tail);
            text += tail;
        }
        model_->end_request();
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
                {"reason", std::string(model_->prefix_reason())},
                {"retention_reason",
                 input.reuse_prefix ? "resident_engine" : "disabled"}
            };
        }
        std::vector<std::uint32_t>().swap(prompt_);
        request_memory_.reset();
        const auto milliseconds = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        return present_chat(
            request,
            Json{
                {"text", text},
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

    void close() override {
        std::vector<std::uint32_t>().swap(prompt_);
        request_memory_.reset();
        tokenizer_.reset();
        if (model_)
            model_->close();
        model_.reset();
    }

private:
    std::size_t prompt_tokens_ = 0, reused_tokens_ = 0;
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::function<void(const std::string&)> content_;
    ModelStore& store_;
    std::unique_ptr<text::Engine> model_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<std::uint32_t> prompt_;
    std::size_t capacity_ = 8192, max_output_ = 256;
};
}  // namespace

std::unique_ptr<InferenceEngine> chat_engine(
    std::shared_ptr<Resources> resources,
    Progress progress,
    std::function<void(const std::string&)> content,
    ModelStore& store
) {
    return std::make_unique<ChatEngine>(
        std::move(resources), std::move(progress), std::move(content), store
    );
}
}  // namespace kadan::serving
