#include <algorithm>
#include <array>

#include "../../weights/metadata_json.hpp"
#include "gguf_model.hpp"
#include "kadan/gguf_text.hpp"
#include "kadan/text_prefix.hpp"

namespace kadan::text {
namespace {
// Checkpoint metadata, cached parameters and bounded chunk activations.
// Prefill reads nearly every routed expert once per chunk, so long chunks
// spread that pass over more tokens, as llama.cpp's 512-token micro-batches do.
constexpr std::size_t fixed_bytes = 1024ULL << 20, chunk_rows = 512;

struct Architecture {
    void (*bind)(GgufWeights&);
    std::size_t (*state)(std::size_t);
    std::unique_ptr<Model> (*model)(const GgufWeights&, DenseCompute&, std::size_t);
};

Architecture architecture(const std::string& name) {
    if (name == "qwen35moe")
        return {bind_qwen35moe, qwen35moe_state_bytes, qwen35moe_model};
    if (name == "qwen4exp")
        return {bind_qwen4exp, qwen4exp_state_bytes, qwen4exp_model};
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
    return fixed_bytes + architecture(name).state(context) + context * sizeof(unsigned);
}
}  // namespace

std::string validate_metadata(const checkpoint::GgufMetadata& m) {
    const auto name = m.string("general.architecture");
    if (name == "qwen35moe") {
        validate_qwen35moe(m);
        return "qwen35moe";
    }
    if (name == "qwen4exp") {
        validate_qwen4exp(m);
        return "qwen4exp";
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
            && (type == "qwen35moe" || type == "qwen4exp") && maximum == 262144,
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
    Prefix prefix;
    Footprint retained_state;
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
            prefix = Prefix(p.context);
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
            // KV mirrors stay out of the weight bank's reservation.
            execution.compute->reserve_device_state(model->device_state_bytes());
            // False selects the bounded disk-source projection path.
            execution.compute->prepare_weights(p.execution_weights);
            execution.resources->loaded(memory);
        } catch (...) {
            model.reset();
            weights.reset();
            prefix = Prefix{};
            execution.resources->released(memory);
            memory = 0;
            throw;
        }
    }

    ~Impl() {
        model.reset();
        weights.reset();
        prefix = Prefix{};
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
    impl_->execution.idle();
    impl_->model->reset();
    impl_->tokens = 0;
    impl_->poisoned = false;
    impl_->prefix.clear();
    impl_->retained_state.clear();
}

std::size_t Engine::begin_request(
    std::span<const unsigned> prompt, std::string_view conversation, bool reuse_prefix
) {
    auto& m = *impl_;
    need(!m.active, "gguf_text_request_state");
    m.weights->check_unchanged();
    m.execution.resume();
    if (m.poisoned)
        m.prefix.clear();
    const auto reused = m.prefix.begin(prompt, conversation, reuse_prefix);
    need(!reused || reused == m.tokens, "gguf_text_prefix_position");
    if (!reused) {
        m.execution.idle();
        m.model->reset();
        m.tokens = 0;
        m.poisoned = false;
        m.retained_state.clear();
    }
    m.active = true;
    return reused;
}

void Engine::end_request() {
    auto& m = *impl_;
    need(m.active && !m.poisoned, "gguf_text_request_state");
    m.retained_state = m.execution.idle({}, m.prefix.retaining());
    m.prefix.complete();
    m.active = false;
}

Bytes PrefixSnapshot::bytes() const {
    Bytes bytes = sizeof(*this) + prefix.memory_bytes()
                  + weights.capacity() * sizeof(WeightIdentity)
                  + state.layers.capacity() * sizeof(decltype(state.layers)::value_type)
                  + state.ple.capacity() * sizeof(float)
                  + state.recent.capacity() * sizeof(unsigned);
    const auto attention_bytes = [](const AttentionState& s) {
        return (s.keys.capacity() + s.values.capacity()) * sizeof(float);
    };
    for (const auto& layer : state.layers)
        std::visit(
            [&](const auto& s) {
                using T = std::decay_t<decltype(s)>;
                if constexpr (std::is_same_v<T, AttentionState>)
                    bytes += attention_bytes(s);
                else if constexpr (std::is_same_v<T, RecurrentState>)
                    bytes += (s.convolution.capacity() + s.recurrent.capacity())
                             * sizeof(float);
                else
                    bytes +=
                        attention_bytes(s.attention)
                        + (s.raw_keys.capacity() + s.pooled.capacity()) * sizeof(float);
            },
            layer
        );
    return bytes;
}

std::unique_ptr<PrefixSnapshot> Engine::take_prefix() {
    auto& m = *impl_;
    if (m.active || m.poisoned || !m.prefix.stored())
        return {};
    m.weights->check_unchanged();
    auto saved = std::make_unique<PrefixSnapshot>();
    const auto& gguf = m.weights->gguf();
    saved->weights.reserve(gguf.tensor_count());
    for (std::size_t i = 0; i < gguf.tensor_count(); ++i)
        saved->weights.push_back(gguf.identity(gguf.tensor_at(i).name));
    saved->context = m.p.context;
    saved->state = m.model->take_state();
    saved->prefix = std::move(m.prefix);
    m.prefix = Prefix{};
    return saved;
}

bool Engine::restore_prefix(std::unique_ptr<PrefixSnapshot> saved) {
    auto& m = *impl_;
    const auto& gguf = m.weights->gguf();
    if (!saved || saved->context != m.p.context
        || saved->weights.size() != gguf.tensor_count())
        return false;
    m.weights->check_unchanged();
    for (std::size_t i = 0; i < gguf.tensor_count(); ++i)
        if (saved->weights[i] != gguf.identity(gguf.tensor_at(i).name))
            return false;
    m.model->restore_state(std::move(saved->state));
    m.prefix = std::move(saved->prefix);
    m.tokens = m.prefix.stored();
    return true;
}

Footprint Engine::prefix_memory() const {
    const auto& m = *impl_;
    if (!m.prefix.stored())
        return Footprint(m.execution.resources->snapshot().used.size());
    auto bytes = m.retained_state;
    bytes[0] = architecture(m.weights->gguf().string("general.architecture"))
                   .state(m.p.context)
               + m.p.context * sizeof(unsigned);
    return bytes;
}

std::size_t Engine::stored_prefix_tokens() const {
    return impl_->prefix.stored();
}

std::string_view Engine::prefix_reason() const {
    return impl_->prefix.reason();
}

void Engine::park() {
    need(!impl_->active, "gguf_text_active_park");
    impl_->execution.park();
    impl_->prefix.clear();
    impl_->retained_state.clear();
}

serving::Token Engine::step(unsigned token, bool logits) {
    auto& m = *impl_;
    need(m.active && !m.poisoned && m.tokens < m.p.context, "gguf_text_step_state");
    try {
        const std::array<unsigned, 1> row{token};
        const auto values = m.model->forward(row, m.tokens, logits, {});
        m.weights->advance(1);
        ++m.tokens;
        m.prefix.commit(row);
        if (!logits)
            return {0, false, m.tokens};
        return m.select(values);
    } catch (...) {
        m.poisoned = true;
        m.prefix.clear();
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
            m.prefix.commit(prompt.subspan(done, count));
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
        m.prefix.clear();
        throw;
    }
}

void Engine::set_projection_timing(bool enabled) {
    impl_->weights->set_projection_timing(enabled);
}

Engine::Timing Engine::take_timing() {
    const auto& w = *impl_->weights;
    const auto timing = w.take_timing();
    const auto placement = w.placement();
    return {
        timing.projection,
        timing.attention,
        timing.prepare,
        timing.device,
        timing.host_wait,
        timing.recurrent,
        timing.projections,
        timing.batches,
        timing.streamed,
        timing.host,
        w.packed_bytes(),
        placement.device,
        placement.ram,
        placement.device_slices,
        placement.ram_slices,
        timing.projection_detail
    };
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
