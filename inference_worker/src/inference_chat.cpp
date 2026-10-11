#include <chrono>

#include "kadan/auto_model.hpp"
#include "kadan/chat_prompt.hpp"
#include "kadan/glm.hpp"
#include "kadan/inference_engines.hpp"
#include "kadan/preparation.hpp"
#include "kadan/text_tokenizer.hpp"
#include "kadan/utf8_stream.hpp"

namespace kadan::serving {
namespace {
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
        const auto& input = std::get<ChatRequest>(request.input);
        const auto root = request.configuration.checkpoint.string();
        glm_ = input.architecture == "glm5_next";
        capacity_ = request.configuration.context_limit;
        require(capacity_ > 0 && capacity_ <= 1048576, "chat_context_limit");
        if (available[0] < 2ULL * 1024 * 1024 * 1024)
            throw AdmissionWait("chat_host_headroom");
        auto plan = media_plan(available);
        if (glm_) {
            const auto layout = glm::plan(root.c_str(), capacity_, &cancel);
            if (layout.host + 1024ULL * 1024 * 1024 > available[0])
                throw AdmissionWait("glm_host_headroom");
            glm_model_ = std::make_unique<glm::Engine>(
                root.c_str(), layout, plan, resources_, &cancel
            );
        } else {
            std::map<int, std::size_t> budgets;
            for (int device : plan.devices)
                budgets.emplace(device, available[device + 1]);
            cuda::ModelOptions options;
            options.capacity = capacity_;
            options.split_residency = true;
            options.weight_ram_bytes =
                std::min<Bytes>(8ULL * 1024 * 1024 * 1024, available[0] / 8);
            options.weight_cold_bytes = 0;
            qwen_model_ = std::make_unique<cuda::AutoModel>(
                root.c_str(), options, budgets, resources_, &cancel
            );
            qwen_model_->end_request();
        }
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
            chat_prompt(input.messages, glm_, byte_limit), prompt_, cancel, byte_limit
        );
        prompt_.resize(count);
        prompt_tokens_ = count;
        require(count + max_output_ <= capacity_, "chat_context_exceeded");
        // Admission belongs before execution so fresh pressure can release/replan.
        try {
            if (glm_) {
                glm_model_->set_cancel(cancel);
                glm_model_->begin_request();
            } else {
                qwen_model_->set_cancel(cancel);
                qwen_model_->begin_request();
            }
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
            store_.select(request.model);
            if (input.context_supplied)
                store_.set_context(
                    request.model,
                    input.context_limit ? Json(*input.context_limit) : Json(nullptr)
                );
            return Json::object();
        }
        const auto started = std::chrono::steady_clock::now();
        auto first = started, last = started;
        auto step = [&](unsigned token, bool generate) {
            check_cancel(cancel);
            if (glm_)
                return glm_model_->step(token, generate);
            const auto selected = qwen_model_->step(token, generate);
            return Token{selected.token, selected.eos, qwen_model_->tokens()};
        };
        Token selected{};
        for (std::size_t i = 0; i < prompt_.size(); ++i) {
            selected = step(prompt_[i], false);
            progress_("prefill_token", i + 1);
        }
        std::string text, pending, reason = "length";
        std::size_t generated = 0;
        for (; generated < max_output_; ++generated) {
            check_cancel(cancel);
            if (selected.eos) {
                reason = "stop";
                break;
            }
            last = std::chrono::steady_clock::now();
            if (!generated) {
                first = last;
                if (glm_) {
                    text = "<think>";
                    content_(text);
                }
            }
            pending += tokenizer_->decode(selected.id);
            auto chunk = utf8_chunk(pending);
            if (!chunk.empty()) {
                content_(chunk);
                text += chunk;
            }
            if (generated + 1 < max_output_)
                selected = step(selected.id, true);
        }
        auto tail = utf8_chunk(pending, true);
        if (!tail.empty()) {
            content_(tail);
            text += tail;
        }
        if (glm_)
            glm_model_->end_request();
        else
            qwen_model_->end_request();
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
                {"timing",
                 {{"generation_ttft_ms",
                   generated ? Json(milliseconds(first - started)) : Json(nullptr)},
                  {"prefill_ms",
                   generated ? Json(milliseconds(first - started)) : Json(nullptr)},
                  {"output_tokens", generated},
                  {"prefill_tokens", prompt_tokens_},
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
        if (glm_model_)
            glm_model_->close();
        glm_model_.reset();
        if (qwen_model_)
            qwen_model_->close();
        qwen_model_.reset();
    }

private:
    std::size_t prompt_tokens_ = 0;
    std::shared_ptr<Resources> resources_;
    Progress progress_;
    std::function<void(const std::string&)> content_;
    ModelStore& store_;
    std::unique_ptr<glm::Engine> glm_model_;
    std::unique_ptr<cuda::AutoModel> qwen_model_;
    std::unique_ptr<text::Tokenizer> tokenizer_;
    std::unique_ptr<HostAllocation> request_memory_;
    std::vector<std::uint32_t> prompt_;
    std::size_t capacity_ = 8192, max_output_ = 256;
    bool glm_ = false;
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
