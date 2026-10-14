#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "GptOssModel requires legacy default stream"
#endif
#include "kadan/gpt_oss_model.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "gpt_oss_kernel.cuh"
#include "kadan/cuda_error.hpp"
#include "kadan/gpt_oss_manifest.hpp"
#include "kadan/gpt_oss_plan.hpp"
#include "kadan/harmony.hpp"
#include "stack_kernel.cuh"

namespace kadan::cuda {
namespace {
namespace kernel = detail::gpt_oss;
using gpt_oss::Config;
constexpr std::size_t metadata_bytes = 256ULL << 20, staging_bytes = 64ULL << 20,
                      control_bytes = 16ULL << 20;

void require(bool value, const char* error) {
    if (!value)
        throw std::invalid_argument(error);
}

void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

// Cancellation leaves queued work consistent; the next request resets state.
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("model_cancelled") {
    }
};

// Bump allocation in the same aligned units as gpt_oss_plan's arithmetic.
struct Carve {
    std::uint8_t* base;
    std::size_t size, used = 0;

    template <class T> T* take(std::size_t bytes) {
        bytes = gpt_oss::aligned(bytes);
        require(bytes <= size - used, "gpt_oss_carve");
        auto* result = base + used;
        used += bytes;
        return reinterpret_cast<T*>(result);
    }
};

std::string layer_name(std::size_t layer, std::string_view suffix) {
    return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}
}  // namespace

struct GptOssModel::Impl {
    struct Scratch {
        float *x, *normed, *qkv, *attention, *router, *weights, *activation, *partial,
            *frequency;
        unsigned *indices, *status;
    };

    struct Device {
        void* weights = nullptr;
        void* state = nullptr;
        bool touched = false;
        Scratch scratch{};
    };

    struct Layer {
        int device;
        std::size_t resident, slots;
        std::uint16_t *input_norm, *post_norm, *qkv, *qkv_bias, *output, *output_bias,
            *sinks, *router, *router_bias, *gate_up_bias, *down_bias;
        const std::uint8_t** experts;
        std::uint8_t *local, *host;
        std::uint16_t *keys, *values;
    };

    std::shared_ptr<Resources> resources;
    std::shared_ptr<checkpoint::MemoryBudget> metadata =
        std::make_shared<checkpoint::MemoryBudget>(metadata_bytes);
    std::optional<gpt_oss::Manifest> manifest;
    gpt_oss::Placement plan;
    gpt_oss::Rope rope = gpt_oss::rope();
    std::size_t capacity, position = 0;
    std::vector<Layer> layers;
    std::map<int, Device> gpu;
    std::uint16_t *head_norm = nullptr, *head = nullptr;
    float* logits = nullptr;
    unsigned* selected = nullptr;
    // Mapped host memory: the embedding table, then each layer's host experts.
    std::uint8_t *host = nullptr, *mapped = nullptr, *staging = nullptr;
    std::thread::id thread = std::this_thread::get_id();
    Handle host_handle = 0, weights = 0, state = 0;
    bool host_loaded = false, weights_loaded = false, state_loaded = false,
         poisoned = false;
    const std::atomic_bool* cancel = nullptr;

    Impl(std::shared_ptr<Resources> r, std::size_t context)
        : resources(std::move(r)), capacity(context) {
    }

    const Config& config() const {
        return manifest->config();
    }

    void check_cancel() const {
        if (cancel && cancel->load())
            throw Cancelled();
    }

    void current(int device) {
        require(thread == std::this_thread::get_id(), "model_thread_changed");
        check(cudaSetDevice(device));
    }

    Footprint footprint() const {
        auto value = resources->snapshot().capacity;
        std::fill(value.begin(), value.end(), 0);
        return value;
    }

    void initialize(const char* root, const std::map<int, std::size_t>& budgets) {
        check_cancel();
        auto h = footprint();
        manifest.emplace(root, metadata);
        check_cancel();
        plan =
            gpt_oss::place(config(), capacity, budgets, GptOssModel::device_headroom);
        h[0] = metadata_bytes + staging_bytes + control_bytes + plan.host_bytes;
        host_handle = resources->reserve(Workload::llm, h);
        const auto used = resources->snapshot().used;
        for (auto [device, bytes] : plan.allocations) {
            (void)bytes;
            require(used.at(device + 1) == 0, "split_requires_exclusive_context");
            h.at(device + 1) = GptOssModel::device_headroom;
            gpu.emplace(device, Device{});
        }
        resources->resize_loading(host_handle, h);
        for (auto& [device, d] : gpu) {
            d.touched = true;
            current(device);
            int major = 0, minor = 0;
            check(cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, device
            ));
            check(cudaDeviceGetAttribute(
                &minor, cudaDevAttrComputeCapabilityMinor, device
            ));
            require(major == 8 && minor >= 6, "requires_sm86");
        }
        current(plan.head);
        check(cudaHostAlloc(
            reinterpret_cast<void**>(&staging), staging_bytes, cudaHostAllocPortable
        ));
        check(cudaHostAlloc(
            reinterpret_cast<void**>(&host),
            plan.host_bytes,
            cudaHostAllocMapped | cudaHostAllocPortable
        ));
        check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), host, 0));
        load_host();
        begin_request();
        resources->loaded(host_handle);
        host_loaded = true;
    }

    // Reads checkpoint bytes in bounded chunks with cancellation between them.
    void read(
        const std::string& name,
        std::size_t offset,
        std::uint8_t* destination,
        std::size_t bytes
    ) {
        for (std::size_t at = 0; at < bytes; at += staging_bytes) {
            check_cancel();
            const auto count = std::min(staging_bytes, bytes - at);
            manifest->read_tensor(name, offset + at, {destination + at, count});
        }
    }

    static void finite_bf16(const std::uint8_t* data, std::size_t bytes) {
        for (std::size_t i = 1; i < bytes; i += 2)
            require(
                (data[i] & 0x7f) != 0x7f || !(data[i - 1] & 0x80), "gpt_oss_nonfinite"
            );
    }

    static void valid_scales(const std::uint8_t* data, std::size_t bytes) {
        require(std::find(data, data + bytes, 255) == data + bytes, "gpt_oss_scale");
    }

    // BF16 tensor bytes into device memory through pinned staging.
    void upload(const std::string& name, void* destination, std::size_t bytes) {
        require(manifest->tensor(name).bytes == bytes, "gpt_oss_tensor_bytes");
        for (std::size_t at = 0; at < bytes; at += staging_bytes) {
            const auto count = std::min(staging_bytes, bytes - at);
            read(name, at, staging, count);
            finite_bf16(staging, count);
            check(cudaMemcpy(
                static_cast<std::uint8_t*>(destination) + at,
                staging,
                count,
                cudaMemcpyHostToDevice
            ));
        }
    }

    // One expert record: gate_up blocks/scales, then down blocks/scales.
    void read_expert(std::size_t layer, std::size_t expert, std::uint8_t* out) {
        using gpt_oss::gate_up_rows, gpt_oss::mxfp4_groups;
        for (auto [part, rows] :
             {std::pair{"gate_up_proj", gate_up_rows},
              std::pair{"down_proj", Config::hidden}}) {
            const auto blocks = rows * mxfp4_groups * 16, scales = rows * mxfp4_groups;
            const auto prefix = layer_name(layer, "mlp.experts.") + part;
            read(prefix + "_blocks", expert * blocks, out, blocks);
            read(prefix + "_scales", expert * scales, out + blocks, scales);
            valid_scales(out + blocks, scales);
            out += blocks + scales;
        }
    }

    void load_host() {
        const auto embedding = gpt_oss::embedding_bytes();
        require(
            manifest->tensor("model.embed_tokens.weight").bytes == embedding,
            "gpt_oss_tensor_bytes"
        );
        read("model.embed_tokens.weight", 0, host, embedding);
        finite_bf16(host, embedding);
        auto offset = embedding;
        for (std::size_t i = 0; i < config().layers; ++i) {
            Layer layer{};
            layer.device = plan.devices[i];
            layer.resident = plan.resident[i];
            layer.slots = gpt_oss::layer_slots(i, capacity);
            layer.host = mapped + offset;
            for (auto e = layer.resident; e < config().experts; ++e) {
                read_expert(i, e, host + offset);
                offset += gpt_oss::expert_bytes;
            }
            layers.push_back(layer);
        }
        require(offset == plan.host_bytes, "gpt_oss_host_layout");
    }

    void bind() {
        const auto& c = config();
        using gpt_oss::attention_width, gpt_oss::qkv_rows;
        for (auto& [device, d] : gpu) {
            const auto& a = plan.allocations.at(device);
            Carve w{static_cast<std::uint8_t*>(d.weights), a.weights};
            Carve s{static_cast<std::uint8_t*>(d.state), a.state};
            if (device == plan.head) {
                head_norm = w.take<std::uint16_t>(Config::hidden * 2);
                head = w.take<std::uint16_t>(Config::vocab * Config::hidden * 2);
                logits = s.take<float>(Config::vocab * 4);
                selected = s.take<unsigned>(4);
            }
            auto& x = d.scratch;
            x.x = s.take<float>(Config::hidden * 4);
            x.normed = s.take<float>(Config::hidden * 4);
            x.qkv = s.take<float>(qkv_rows * 4);
            x.attention = s.take<float>(attention_width * 4);
            x.router = s.take<float>(c.experts * 4);
            x.indices = s.take<unsigned>(2 * Config::experts_per_token * 4);
            x.weights = reinterpret_cast<float*>(x.indices + Config::experts_per_token);
            x.activation =
                s.take<float>(Config::experts_per_token * Config::intermediate * 4);
            x.partial = s.take<float>(
                Config::heads
                * ((capacity + gpt_oss::attention_chunk - 1) / gpt_oss::attention_chunk)
                * (Config::head_dim + 2) * 4
            );
            x.frequency = s.take<float>(Config::head_dim / 2 * 4);
            x.status = s.take<unsigned>(4);
            for (std::size_t i = 0; i < layers.size(); ++i) {
                auto& l = layers[i];
                if (l.device != device)
                    continue;
                l.input_norm = w.take<std::uint16_t>(Config::hidden * 2);
                l.post_norm = w.take<std::uint16_t>(Config::hidden * 2);
                l.qkv = w.take<std::uint16_t>(qkv_rows * Config::hidden * 2);
                l.qkv_bias = w.take<std::uint16_t>(qkv_rows * 2);
                l.output = w.take<std::uint16_t>(Config::hidden * attention_width * 2);
                l.output_bias = w.take<std::uint16_t>(Config::hidden * 2);
                l.sinks = w.take<std::uint16_t>(Config::heads * 2);
                l.router = w.take<std::uint16_t>(c.experts * Config::hidden * 2);
                l.router_bias = w.take<std::uint16_t>(c.experts * 2);
                l.gate_up_bias =
                    w.take<std::uint16_t>(c.experts * gpt_oss::gate_up_rows * 2);
                l.down_bias = w.take<std::uint16_t>(c.experts * Config::hidden * 2);
                l.experts = w.take<const std::uint8_t*>(c.experts * 4 * sizeof(void*));
                l.local = w.take<std::uint8_t>(l.resident * gpt_oss::expert_bytes);
                l.keys = s.take<std::uint16_t>(l.slots * gpt_oss::kv_width * 2);
                l.values = s.take<std::uint16_t>(l.slots * gpt_oss::kv_width * 2);
            }
        }
    }

    void load_weights() {
        const auto& c = config();
        using gpt_oss::attention_width, gpt_oss::kv_width;
        current(plan.head);
        upload("model.norm.weight", head_norm, Config::hidden * 2);
        upload("lm_head.weight", head, Config::vocab * Config::hidden * 2);
        constexpr auto q_rows = attention_width;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            auto& l = layers[i];
            current(l.device);
            const auto name = [&](std::string_view suffix) {
                return layer_name(i, suffix);
            };
            upload(name("input_layernorm.weight"), l.input_norm, Config::hidden * 2);
            upload(
                name("post_attention_layernorm.weight"), l.post_norm, Config::hidden * 2
            );
            upload(name("self_attn.q_proj.weight"), l.qkv, q_rows * Config::hidden * 2);
            upload(
                name("self_attn.k_proj.weight"),
                l.qkv + q_rows * Config::hidden,
                kv_width * Config::hidden * 2
            );
            upload(
                name("self_attn.v_proj.weight"),
                l.qkv + (q_rows + kv_width) * Config::hidden,
                kv_width * Config::hidden * 2
            );
            upload(name("self_attn.q_proj.bias"), l.qkv_bias, q_rows * 2);
            upload(name("self_attn.k_proj.bias"), l.qkv_bias + q_rows, kv_width * 2);
            upload(
                name("self_attn.v_proj.bias"),
                l.qkv_bias + q_rows + kv_width,
                kv_width * 2
            );
            upload(
                name("self_attn.o_proj.weight"),
                l.output,
                Config::hidden * attention_width * 2
            );
            upload(name("self_attn.o_proj.bias"), l.output_bias, Config::hidden * 2);
            upload(name("self_attn.sinks"), l.sinks, Config::heads * 2);
            upload(name("mlp.router.weight"), l.router, c.experts * Config::hidden * 2);
            upload(name("mlp.router.bias"), l.router_bias, c.experts * 2);
            upload(
                name("mlp.experts.gate_up_proj_bias"),
                l.gate_up_bias,
                c.experts * gpt_oss::gate_up_rows * 2
            );
            upload(
                name("mlp.experts.down_proj_bias"),
                l.down_bias,
                c.experts * Config::hidden * 2
            );
            std::vector<const std::uint8_t*> table(c.experts);
            for (std::size_t e = 0; e < c.experts; ++e) {
                if (e < l.resident) {
                    read_expert(i, e, staging);
                    check(cudaMemcpy(
                        l.local + e * gpt_oss::expert_bytes,
                        staging,
                        gpt_oss::expert_bytes,
                        cudaMemcpyHostToDevice
                    ));
                    table[e] = l.local + e * gpt_oss::expert_bytes;
                } else
                    table[e] = l.host + (e - l.resident) * gpt_oss::expert_bytes;
            }
            check(cudaMemcpy(
                l.experts,
                table.data(),
                table.size() * sizeof(void*),
                cudaMemcpyHostToDevice
            ));
        }
        manifest->check_unchanged();
    }

    void free_device(bool release_weights) {
        for (auto& [device, d] : gpu) {
            if (!d.touched)
                continue;
            current(device);
            check(cudaDeviceSynchronize());
            if (d.state) {
                check(cudaFree(d.state));
                d.state = nullptr;
            }
            if (release_weights && d.weights) {
                check(cudaFree(d.weights));
                d.weights = nullptr;
            }
        }
    }

    void release(Handle& handle, bool& loaded) {
        if (!handle)
            return;
        if (loaded)
            resources->begin_eviction(handle);
        resources->released(handle);
        handle = 0;
        loaded = false;
    }

    void begin_request() {
        check_cancel();
        require(!poisoned && !state, "model_request_state");
        const bool reload = !weights;
        auto w = footprint(), s = footprint();
        for (auto [device, a] : plan.allocations) {
            w.at(device + 1) = a.weights;
            s.at(device + 1) = a.state;
        }
        try {
            if (reload)
                weights = resources->reserve(Workload::llm, w);
            state = resources->reserve(Workload::llm, s);
            for (auto [device, a] : plan.allocations) {
                current(device);
                std::size_t free = 0, total = 0;
                check(cudaMemGetInfo(&free, &total));
                require(
                    free >= a.state + (reload ? a.weights : 0),
                    "model_physical_headroom"
                );
                auto& d = gpu.at(device);
                if (reload)
                    check(cudaMalloc(&d.weights, a.weights));
                check(cudaMalloc(&d.state, a.state));
            }
            bind();
            if (reload)
                load_weights();
            reset_state();
        } catch (...) {
            // Admission and cancellation failures leave nothing allocated.
            try {
                free_device(reload);
                release(state, state_loaded);
                if (reload)
                    release(weights, weights_loaded);
            } catch (...) {
                poisoned = true;
            }
            throw;
        }
        if (reload) {
            resources->loaded(weights);
            weights_loaded = true;
        }
        resources->loaded(state);
        state_loaded = true;
    }

    void reset_state() {
        for (auto& [device, d] : gpu) {
            current(device);
            check(cudaMemset(d.scratch.status, 0, sizeof(unsigned)));
            check(cudaMemcpy(
                d.scratch.frequency,
                rope.inverse_frequency.data(),
                sizeof(rope.inverse_frequency),
                cudaMemcpyHostToDevice
            ));
        }
        position = 0;
        current(plan.head);
    }

    void end_request() {
        try {
            free_device(false);
            release(state, state_loaded);
            if (gpu.contains(plan.head))
                current(plan.head);
        } catch (...) {
            poisoned = true;
            throw;
        }
    }

    void park() {
        end_request();
        try {
            free_device(true);
            release(weights, weights_loaded);
            if (gpu.contains(plan.head))
                current(plan.head);
        } catch (...) {
            poisoned = true;
            throw;
        }
    }

    void layer(std::size_t index, Scratch& x) {
        const auto& l = layers[index];
        const auto& c = config();
        check(kernel::norm(x.x, l.input_norm, x.normed));
        check(
            kernel::linear(
                l.qkv,
                l.qkv_bias,
                x.normed,
                x.qkv,
                gpt_oss::qkv_rows,
                Config::hidden,
                false
            )
        );
        check(kernel::rotary(x.qkv, x.frequency, rope.scaling, position));
        check(kernel::store(x.qkv, l.keys, l.values, position % l.slots));
        check(
            kernel::attention(
                x.qkv,
                l.keys,
                l.values,
                l.sinks,
                std::min(position + 1, l.slots),
                x.partial,
                x.attention
            )
        );
        check(
            kernel::linear(
                l.output,
                l.output_bias,
                x.attention,
                x.x,
                Config::hidden,
                gpt_oss::attention_width,
                true
            )
        );
        check(kernel::norm(x.x, l.post_norm, x.normed));
        check(
            kernel::linear(
                l.router,
                l.router_bias,
                x.normed,
                x.router,
                c.experts,
                Config::hidden,
                false
            )
        );
        check(kernel::route(x.router, c.experts, x.indices, x.weights, x.status));
        check(
            kernel::experts(
                l.experts,
                l.gate_up_bias,
                l.down_bias,
                x.indices,
                x.weights,
                x.normed,
                x.activation,
                x.x
            )
        );
    }

    serving::Token step(unsigned token, bool want_logits) {
        require(!poisoned && state && weights, "model_unavailable");
        require(token < Config::vocab && position < capacity, "gpt_oss_step_state");
        try {
            int previous = layers.front().device;
            current(previous);
            check(
                kernel::embed(
                    reinterpret_cast<const std::uint16_t*>(mapped),
                    token,
                    gpu.at(previous).scratch.x
                )
            );
            for (std::size_t i = 0; i < layers.size(); ++i) {
                check_cancel();
                const int device = layers[i].device;
                auto& scratch = gpu.at(device).scratch;
                if (device != previous) {
                    check(cudaMemcpyPeer(
                        scratch.x,
                        device,
                        gpu.at(previous).scratch.x,
                        previous,
                        Config::hidden * sizeof(float)
                    ));
                    current(device);
                    previous = device;
                }
                layer(i, scratch);
            }
            ++position;
            if (!want_logits)
                return {0, false, position};
            require(previous == plan.head, "gpt_oss_head_device");
            auto& x = gpu.at(plan.head).scratch;
            check(kernel::norm(x.x, head_norm, x.normed));
            check(
                kernel::linear(
                    head,
                    nullptr,
                    x.normed,
                    logits,
                    Config::vocab,
                    Config::hidden,
                    false
                )
            );
            check(detail::stack_select(Config::vocab, logits, selected, x.status));
            unsigned id = 0;
            check(cudaMemcpy(&id, selected, sizeof(id), cudaMemcpyDeviceToHost));
            for (auto& [device, d] : gpu) {
                current(device);
                unsigned flags = 0;
                check(cudaMemcpy(
                    &flags, d.scratch.status, sizeof(flags), cudaMemcpyDeviceToHost
                ));
                if (flags)
                    throw std::overflow_error("model_numeric_failure");
            }
            current(plan.head);
            require(id < Config::vocab, "model_device_selection");
            return {id, serving::harmony::is_stop(id), position};
        } catch (const Cancelled&) {
            throw;
        } catch (...) {
            poisoned = true;
            throw;
        }
    }

    bool cleanup() noexcept {
        if (!host_handle)
            return true;
        try {
            park();
            if (host || staging)
                current(plan.head);
            if (host)
                check(cudaFreeHost(host));
            host = mapped = nullptr;
            if (staging)
                check(cudaFreeHost(staging));
            staging = nullptr;
            layers.clear();
            manifest.reset();
            for (auto& [device, d] : gpu) {
                if (!d.touched)
                    continue;
                require(
                    resources->snapshot().used.at(device + 1)
                        == GptOssModel::device_headroom,
                    "context_has_other_device_owner"
                );
                current(device);
#ifndef KADAN_ATEN_RUNTIME
                check(cudaDeviceReset());
#endif
            }
            release(host_handle, host_loaded);
            return true;
        } catch (...) {
            poisoned = true;
            return false;
        }
    }
};

GptOssModel::GptOssModel(
    const char* root,
    std::size_t capacity,
    const std::map<int, std::size_t>& budgets,
    std::shared_ptr<Resources> resources,
    const std::atomic_bool* cancel
)
    : impl_(std::make_unique<Impl>(std::move(resources), capacity)) {
    impl_->cancel = cancel;
    try {
        impl_->initialize(root, budgets);
    } catch (...) {
        if (!impl_->cleanup())
            throw DeviceBufferQuarantine("gpt_oss_cleanup_unconfirmed");
        throw;
    }
}

GptOssModel::~GptOssModel() {
    if (impl_)
        impl_->cleanup();
}

std::size_t GptOssModel::host_bytes(
    const char* root, std::size_t capacity, const std::map<int, std::size_t>& budgets
) {
    const auto config =
        gpt_oss::read_config(root, std::make_shared<checkpoint::MemoryBudget>(1 << 20));
    return metadata_bytes + staging_bytes + control_bytes
           + gpt_oss::place(config, capacity, budgets, device_headroom).host_bytes;
}

void GptOssModel::set_cancel(const std::atomic_bool& cancel) {
    impl_->cancel = &cancel;
}

void GptOssModel::begin_request() {
    impl_->begin_request();
}

void GptOssModel::end_request() {
    impl_->end_request();
}

void GptOssModel::park() {
    impl_->park();
}

void GptOssModel::close() {
    if (!impl_)
        return;
    if (!impl_->cleanup())
        throw std::runtime_error("gpt_oss_cleanup_unconfirmed");
    impl_.reset();
}

serving::Token GptOssModel::step(unsigned token, bool logits) {
    return impl_->step(token, logits);
}

std::size_t GptOssModel::tokens() const {
    return impl_->position;
}
}  // namespace kadan::cuda
