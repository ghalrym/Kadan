#include <algorithm>
#include <array>

#include "../../weights/metadata_json.hpp"
#include "gguf_model.hpp"
#include "kadan/gguf_text.hpp"

namespace kadan::text {
namespace {
// Checkpoint metadata, cached parameters and bounded chunk activations.
constexpr std::size_t fixed_bytes = 768ULL << 20, chunk_rows = 64;

struct Architecture {
    void (*bind)(GgufWeights&);
    std::size_t (*state)(std::size_t);
    std::unique_ptr<Model> (*model)(const GgufWeights&, DenseCompute&, std::size_t);
};

Architecture architecture(const std::string& name) {
    if (name == "qwen35")
        return {bind_qwen35, qwen35_state_bytes, qwen35_model};
    if (name == "qwen4exp")
        return {bind_qwen4exp, qwen4exp_state_bytes, qwen4exp_model};
    if (name == "glm5-next")
        return {bind_glm, glm_state_bytes, glm_model};
    throw std::runtime_error("gguf_text_architecture");
}

std::unique_ptr<GgufWeights> load(const char* root, const std::atomic_bool* cancel) {
    auto weights = std::make_unique<GgufWeights>(root, cancel);
    architecture(weights->gguf().string("general.architecture")).bind(*weights);
    return weights;
}

std::size_t host_bytes(const std::string& name, std::size_t context) {
    need(
        context && context <= serving::max_resident_capacity, "gguf_text_context_bound"
    );
    return fixed_bytes + architecture(name).state(context);
}
}  // namespace

std::string validate_metadata(const checkpoint::GgufMetadata& m) {
    const auto name = m.string("general.architecture");
    if (name == "qwen35") {
        validate_qwen35(m);
        return "qwen35";
    }
    if (name == "qwen4exp") {
        validate_qwen4exp(m);
        return "qwen4exp";
    }
    if (name == "glm5-next") {
        validate_glm(m);
        return "glm5_next";
    }
    throw std::runtime_error("gguf_text_architecture");
}

void validate_config(
    const char* root, std::shared_ptr<checkpoint::MemoryBudget> budget
) {
    const auto config = checkpoint::json::read(root, "config.json", 65536, budget);
    const auto type = std::string(config.at("model_type").string());
    const auto maximum = config.at("max_position_embeddings").integer();
    need(
        config.at("kadan_checkpoint_format").string() == "gguf"
            && ((type == "qwen35" || type == "qwen4exp") ? maximum == 262144
                : type == "glm5_next"                    ? maximum == 1048576
                                                         : false),
        "gguf_text_config"
    );
}

Plan plan(const char* root, std::size_t context, const std::atomic_bool* cancel) {
    const auto weights = load(root, cancel);
    return {
        context,
        host_bytes(weights->gguf().string("general.architecture"), context),
        weights->packed_bytes(),
        weights->gguf().tensor_count(),
        weights->execution_bytes()
    };
}

struct Engine::Impl {
    Plan p;
    WorkerCompute execution;
    Handle memory = 0;
    std::unique_ptr<GgufWeights> weights;
    std::unique_ptr<Model> model;
    std::size_t tokens = 0;
    bool active = false, poisoned = false;

    Impl(
        const char* root,
        Plan selected,
        const ComputePlan& compute_plan,
        std::shared_ptr<Resources> owner,
        const std::atomic_bool* cancel
    )
        : p(selected), execution(compute_plan, Workload::llm, std::move(owner)) {
        need(bool(execution.compute), "gguf_text_cuda_required");
        memory = execution.resources->reserve(
            Workload::llm, host_footprint(*execution.resources, p.host + p.cache)
        );
        try {
            weights = load(root, cancel);
            const auto name = weights->gguf().string("general.architecture");
            need(
                weights->packed_bytes() == p.packed
                    && weights->gguf().tensor_count() == p.tensors
                    && weights->execution_bytes() == p.execution_weights
                    && host_bytes(name, p.context) == p.host,
                "gguf_text_preflight_changed"
            );
            weights->set_ram_budget(p.cache);
            model = architecture(name).model(*weights, *execution.compute, p.context);
            // False selects the bounded disk-source projection path.
            execution.compute->prepare_weights(p.execution_weights);
            execution.resources->loaded(memory);
        } catch (...) {
            model.reset();
            weights.reset();
            execution.resources->released(memory);
            memory = 0;
            throw;
        }
    }

    ~Impl() {
        model.reset();
        weights.reset();
        if (memory) {
            execution.resources->begin_eviction(memory);
            execution.resources->released(memory);
        }
    }

    serving::Token select(const Vec& logits) const {
        need(logits.size() == model->vocabulary(), "gguf_text_logits");
        const auto chosen =
            unsigned(std::max_element(logits.begin(), logits.end()) - logits.begin());
        return {chosen, model->eos(chosen), tokens};
    }
};

Engine::Engine(
    const char* root,
    Plan p,
    const ComputePlan& compute_plan,
    std::shared_ptr<Resources> owner,
    const std::atomic_bool* cancel
)
    : impl_(std::make_unique<Impl>(root, p, compute_plan, std::move(owner), cancel)) {
}

Engine::~Engine() = default;

void Engine::set_cancel(const std::atomic_bool& cancel) {
    impl_->weights->set_cancel(cancel);
}

std::shared_ptr<Resources> Engine::resources() const {
    return impl_->execution.resources;
}

serving::Info Engine::info() const {
    return {
        impl_->model->vocabulary(),
        impl_->p.context,
        impl_->p.execution_weights,
        impl_->p.host + impl_->p.cache
    };
}

// Every recurrent row is cleared and KV positions are rewritten before use.
void Engine::reset() {
    impl_->model->reset();
    impl_->tokens = 0;
    impl_->poisoned = false;
}

void Engine::begin_request() {
    need(!impl_->active, "gguf_text_request_state");
    impl_->weights->check_unchanged();
    impl_->execution.resume();
    reset();
    impl_->active = true;
}

void Engine::end_request() {
    impl_->execution.idle();
    impl_->active = false;
}

void Engine::park() {
    need(!impl_->active, "gguf_text_active_park");
    impl_->execution.park();
}

serving::Token Engine::step(unsigned token, bool logits) {
    auto& m = *impl_;
    need(m.active && !m.poisoned && m.tokens < m.p.context, "gguf_text_step_state");
    try {
        const std::array<unsigned, 1> row{token};
        const auto values = m.model->forward(row, m.tokens, logits, {});
        m.weights->advance(1);
        ++m.tokens;
        if (!logits)
            return {0, false, m.tokens};
        return m.select(values);
    } catch (...) {
        m.poisoned = true;
        throw;
    }
}

serving::Token Engine::prefill(
    std::span<const unsigned> prompt,
    const std::function<void(const char*, std::size_t)>& progress
) {
    auto& m = *impl_;
    need(
        m.active && !m.poisoned && !prompt.empty() && m.tokens <= m.p.context
            && prompt.size() <= m.p.context - m.tokens,
        "gguf_text_prefill_state"
    );
    try {
        std::size_t completed_layers = 0;
        const Progress layer_progress = [&](const char*, std::size_t) {
            if (progress)
                progress("prefill_layer", ++completed_layers);
        };
        Vec logits;
        for (std::size_t done = 0; done < prompt.size();) {
            const auto count = std::min(chunk_rows, prompt.size() - done);
            const bool last = done + count == prompt.size();
            logits = m.model->forward(
                prompt.subspan(done, count), m.tokens, last, layer_progress
            );
            m.weights->advance(count);
            m.tokens += count;
            done += count;
            if (progress) {
                progress("prefill_chunk", done);
                for (std::size_t row = done - count; row < done; ++row)
                    progress("prefill_token", row + 1);
            }
        }
        return m.select(logits);
    } catch (...) {
        m.poisoned = true;
        throw;
    }
}

void Engine::close() {
    auto owner = resources();
    impl_->active = false;
    impl_->execution.park();
    impl_.reset();
    for (auto bytes : owner->snapshot().used)
        need(bytes == 0, "gguf_text_cleanup_unconfirmed");
}
}  // namespace kadan::text
