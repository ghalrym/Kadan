#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <type_traits>
#include <variant>

#include "../../weights/metadata_json.hpp"
#include "gguf_model.hpp"
#include "inkling2_vision.hpp"
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
    std::size_t (*device_state)(std::size_t);
    std::unique_ptr<Model> (*model)(const GgufWeights&, DenseCompute&, std::size_t);
};

Architecture architecture(const std::string& name) {
    if (name == "inkling2_loop")
        return {
            bind_inkling2_loop,
            inkling2_loop_state_bytes,
            inkling2_loop_device_state_bytes,
            inkling2_loop_model
        };
    throw std::runtime_error("gguf_text_architecture");
}

std::unique_ptr<GgufWeights> load(const char* root, const std::atomic_bool* cancel) {
    auto weights = std::make_unique<GgufWeights>(root, cancel);
    architecture(weights->gguf().string("general.architecture")).bind(*weights);
    return weights;
}

// The live state and the compacted checkpoint pool budget, each bounded by one
// full-context state plus its token buffer.
std::size_t host_bytes(const std::string& name, std::size_t context) {
    need(
        context && context <= serving::max_resident_capacity, "gguf_text_context_bound"
    );
    return fixed_bytes
           + 2 * (architecture(name).state(context) + context * sizeof(unsigned));
}
}  // namespace

std::string validate_metadata(const checkpoint::GgufMetadata& m) {
    const auto name = m.string("general.architecture");
    if (name == "inkling2_loop") {
        validate_inkling2_loop(m);
        return "inkling2_loop";
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
            && type == "inkling2_loop" && maximum == 1048576,
        "gguf_text_config"
    );
}

Plan plan(const char* root, std::size_t context, const std::atomic_bool* cancel) {
    const auto weights = load(root, cancel);
    const auto name = weights->gguf().string("general.architecture");
    return {
        context,
        host_bytes(name, context),
        weights->packed_bytes(),
        weights->gguf().tensor_count(),
        weights->execution_bytes(),
        0,
        architecture(name).device_state(context)
    };
}

struct Engine::Impl {
    std::string root_dir;
    Plan p;
    WorkerCompute execution;
    Handle memory = 0;
    std::unique_ptr<GgufWeights> weights;
    std::unique_ptr<Model> model;
    std::unique_ptr<InklingVision> vision;
    const std::atomic_bool* cancel = nullptr;
    Prefix prefix;
    CheckpointPool pool;
    std::vector<ImageTag> live_images, pending_images;
    std::vector<unsigned> static_tokens;
    Footprint retained_state;
    std::size_t tokens = 0, boundary = 0, static_boundary = 0;
    double cache_restore_ms = 0.0;
    bool active = false, poisoned = false, save_static = false;

    void save_pool_entry(
        std::string_view conversation,
        bool is_static,
        std::span<const unsigned> entry_tokens,
        std::span<const ImageTag> entry_images
    ) {
        if (!tokens || entry_tokens.size() != tokens || !pool.budget_bytes())
            return;
        auto compacted = model->snapshot_state(tokens, p.context);
        pool.save(
            conversation, is_static, entry_tokens, entry_images, std::move(compacted)
        );
    }

    void restore_pool_entry(const CheckpointPool::Entry& entry) {
        const auto started = std::chrono::steady_clock::now();
        auto state = model->take_state();
        try {
            restore_compact_state(entry.state, state, entry.tokens.size(), p.context);
        } catch (...) {
            model->restore_state(std::move(state));
            throw;
        }
        model->restore_state(std::move(state));
        tokens = entry.tokens.size();
        live_images = entry.images;
        prefix.resume(
            entry.tokens, entry.is_static ? "static_prefix_hit" : "checkpoint_hit"
        );
        cache_restore_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started
                           )
                               .count();
    }

    Impl(
        const char* root,
        Plan selected,
        const ComputePlan& compute_plan,
        std::shared_ptr<Resources> owner,
        const std::atomic_bool* cancel_flag
    )
        : root_dir(root ? root : ""),
          p(selected),
          execution(compute_plan, Workload::llm, std::move(owner)),
          cancel(cancel_flag) {
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
            pool = CheckpointPool(
                architecture(name).state(p.context) + p.context * sizeof(unsigned)
            );
            weights->set_ram_budget(p.cache);
            model = architecture(name).model(*weights, *execution.compute, p.context);
            // KV mirrors stay out of the weight bank's reservation.
            execution.compute->reserve_device_state(model->device_state_bytes());
            // GGUF weight banks retain packed tensor bytes rather than decoded
            // F32 tensors; +12.5% slack keeps Placement's 90% device residency
            // threshold above the full packed footprint without locking away
            // spare VRAM needed by KV mirrors.
            const Bytes slack = p.packed / 8;
            const Bytes packed_target =
                p.packed > std::numeric_limits<Bytes>::max() - slack
                    ? std::numeric_limits<Bytes>::max()
                    : p.packed + slack;
            execution.compute->prepare_weights(packed_target);
            execution.resources->loaded(memory);
        } catch (...) {
            vision.reset();
            model.reset();
            weights.reset();
            pool.clear();
            prefix = Prefix{};
            execution.resources->released(memory);
            memory = 0;
            throw;
        }
    }

    ~Impl() {
        vision.reset();
        model.reset();
        weights.reset();
        pool.clear();
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
    impl_->cancel = &cancel;
    impl_->weights->set_cancel(cancel);
    if (impl_->vision)
        impl_->vision->set_cancel(cancel);
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
    impl_->pool.clear();
    impl_->live_images.clear();
    impl_->pending_images.clear();
    impl_->static_tokens.clear();
    impl_->tokens = 0;
    impl_->boundary = 0;
    impl_->static_boundary = 0;
    impl_->cache_restore_ms = 0.0;
    impl_->poisoned = false;
    impl_->save_static = false;
    impl_->prefix.clear();
    impl_->retained_state.clear();
}

std::size_t Engine::begin_request(
    std::span<const unsigned> prompt, std::string_view conversation, bool reuse_prefix
) {
    return begin_request(prompt, conversation, reuse_prefix, {});
}

std::size_t Engine::begin_request(
    std::span<const unsigned> prompt,
    std::string_view conversation,
    bool reuse_prefix,
    std::span<const ImageTag> prompt_images
) {
    auto& m = *impl_;
    need(!m.active, "gguf_text_request_state");
    m.weights->check_unchanged();
    m.execution.resume();
    m.cache_restore_ms = 0.0;
    if (m.poisoned) {
        m.prefix.clear();
        m.live_images.clear();
    }

    const auto live_stored = m.prefix.stored();
    const bool live_images_ok =
        !live_stored || images_match_prefix(m.live_images, prompt_images, live_stored);
    const bool will_live_hit =
        live_stored > 0 && reuse_prefix && !conversation.empty()
        && conversation == m.prefix.conversation() && live_stored < prompt.size()
        && live_images_ok
        && std::equal(
            m.prefix.committed().begin(), m.prefix.committed().end(), prompt.begin()
        );

    // Preserve a valid outgoing live conversation in the checkpoint pool before
    // resetting, so interleaving conversations or disposable non-reusable calls
    // (such as vision inspections) never clobber cached conversation state.
    if (!will_live_hit && !m.poisoned && live_stored > 0
        && !m.prefix.conversation().empty() && live_stored == m.tokens) {
        m.save_pool_entry(
            m.prefix.conversation(), false, m.prefix.committed(), m.live_images
        );
    }
    if (!will_live_hit && live_stored > 0 && !live_images_ok) {
        m.prefix.clear();
        m.live_images.clear();
    }

    const auto reused = m.prefix.begin(prompt, conversation, reuse_prefix);
    need(!reused || reused == m.tokens, "gguf_text_prefix_position");
    m.pending_images.assign(prompt_images.begin(), prompt_images.end());
    m.save_static = reuse_prefix;
    m.static_tokens.clear();
    if (reuse_prefix && m.static_boundary > 0 && m.static_boundary < prompt.size()) {
        m.static_tokens.assign(prompt.begin(), prompt.begin() + m.static_boundary);
    }
    if (!reused) {
        m.execution.idle();
        m.model->reset();
        m.tokens = 0;
        m.poisoned = false;
        m.retained_state.clear();
        m.live_images.clear();
        // Released mirrors re-upload the restored keys on first attention.
        if (reuse_prefix) {
            if (const auto* entry =
                    m.pool.find_best(prompt, conversation, prompt_images)) {
                try {
                    m.restore_pool_entry(*entry);
                } catch (...) {
                    m.poisoned = true;
                    m.prefix.clear();
                    m.live_images.clear();
                    throw;
                }
            }
        }
    }
    m.active = true;
    return m.tokens;
}

void Engine::checkpoint_at(std::size_t tokens) {
    impl_->boundary = tokens;
}

void Engine::checkpoint_static_at(std::size_t tokens) {
    impl_->static_boundary = tokens;
}

double Engine::last_cache_restore_ms() const {
    return impl_->cache_restore_ms;
}

void Engine::end_request() {
    auto& m = *impl_;
    need(m.active && !m.poisoned, "gguf_text_request_state");
    m.retained_state = m.execution.idle({}, m.prefix.retaining());
    m.prefix.complete();
    m.boundary = 0;
    m.static_boundary = 0;
    m.save_static = false;
    m.pending_images.clear();
    m.static_tokens.clear();
    m.active = false;
}

Bytes PrefixSnapshot::bytes() const {
    return sizeof(*this) + prefix.memory_bytes()
           + images.capacity() * sizeof(ImageTag)
           + weights.capacity() * sizeof(WeightIdentity)
           + state_memory_bytes(state);
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
    saved->images = std::move(m.live_images);
    m.prefix = Prefix{};
    m.live_images.clear();
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
    m.live_images = std::move(saved->images);
    m.tokens = m.prefix.stored();
    return true;
}

Footprint Engine::prefix_memory() const {
    const auto& m = *impl_;
    const auto pool_bytes = m.pool.used_bytes();
    if (!m.prefix.stored() && !pool_bytes)
        return Footprint(m.execution.resources->snapshot().used.size());
    Footprint bytes = m.prefix.stored() && !m.retained_state.empty()
                          ? m.retained_state
                          : Footprint(m.execution.resources->snapshot().used.size());
    const Bytes live_bytes =
        m.prefix.stored()
            ? architecture(m.weights->gguf().string("general.architecture"))
                      .state(m.p.context)
                  + m.p.context * sizeof(unsigned)
            : 0;
    bytes[0] = live_bytes + pool_bytes;
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
    impl_->vision.reset();
    impl_->execution.park();
    impl_->pool.clear();
    impl_->live_images.clear();
    impl_->pending_images.clear();
    impl_->static_tokens.clear();
    impl_->prefix.clear();
    impl_->retained_state.clear();
}

ImageSpan Engine::encode_image(std::span<const std::uint8_t> png_bytes) {
    auto& m = *impl_;
    m.execution.resume();
    if (!m.vision) {
        const auto mmproj = InklingVision::find_mmproj(m.root_dir.c_str());
        need(!mmproj.empty(), "inkling2v_missing_mmproj");
        m.vision = std::make_unique<InklingVision>(mmproj, m.cancel);
    }
    return m.vision->encode(png_bytes, *m.execution.compute);
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
        m.live_images.clear();
        throw;
    }
}

serving::Token Engine::prefill(
    std::span<const unsigned> prompt,
    const std::function<void(const char*, std::size_t)>& progress
) {
    return prefill(prompt, {}, progress);
}

serving::Token Engine::prefill(
    std::span<const unsigned> prompt,
    std::span<const ImageSpan> images,
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
            auto count = std::min(chunk_rows, prompt.size() - done);
            // A chunk ends at a requested checkpoint boundary so the state is
            // saved at that exact token offset.
            if (m.static_boundary > m.tokens && m.static_boundary < m.tokens + count)
                count = m.static_boundary - m.tokens;
            if (m.boundary > m.tokens && m.boundary < m.tokens + count)
                count = m.boundary - m.tokens;
            const bool last = done + count == prompt.size();
            logits = m.model->forward(
                prompt.subspan(done, count), m.tokens, last, layer_progress, images
            );
            m.weights->advance(count);
            m.tokens += count;
            m.prefix.commit(prompt.subspan(done, count));
            m.live_images = images_within_prefix(m.pending_images, m.tokens);
            done += count;
            if (!last && m.static_boundary && m.tokens == m.static_boundary
                && m.save_static && m.live_images.empty()
                && m.static_tokens.size() == m.tokens) {
                m.save_pool_entry("", true, m.static_tokens, {});
            }
            if (!last && m.boundary && m.tokens == m.boundary && m.prefix.retaining()) {
                m.save_pool_entry(
                    m.prefix.conversation(), false, m.prefix.committed(), m.live_images
                );
            }
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
        m.live_images.clear();
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
    impl_->vision.reset();
    impl_->execution.park();
    impl_.reset();
    for (auto bytes : owner->snapshot().used)
        need(bytes == 0, "gguf_text_cleanup_unconfirmed");
}
}  // namespace kadan::text
