#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

namespace kadan::serving {
// Protocol order is fixed. Never use addresses, tensor names, or shape labels.
enum class CudaKernel : std::size_t {
    add_bias,
    bf16_input,
    cached_attention_combine,
    cached_attention_kernel,
    check_image_key_value,
    delta_conv_kernel,
    delta_heads_kernel,
    delta_l2_kernel,
    divide_attention,
    finish_int8,
    gguf_decode_kernel,
    gguf_matvec_batch_kernel,
    gguf_matvec_kernel,
    h3_add_kernel,
    h3_attention_divide_kernel,
    h3_attention_init_kernel,
    h3_finish_int8_kernel,
    h3_finite_kernel,
    h3_gated_residual_kernel,
    h3_lora_add_kernel,
    h3_narrow_kernel,
    h3_normalize_kernel,
    h3_online_softmax_kernel,
    h3_qk_rotary_kernel,
    h3_silu_gate_kernel,
    h3_widen_kernel,
    h3_widen_real_kernel,
    h3_zero_bias_kernel,
    image_frequency_kernel,
    image_gates_kernel,
    image_norm_kernel,
    image_projection_status,
    image_residual_kernel,
    image_rotary_kernel,
    image_scheduler_kernel,
    image_silu_pair_kernel,
    image_text_gelu_kernel,
    image_text_norm_kernel,
    image_timestep_silu_kernel,
    image_vae_gather_kernel,
    image_vae_norm_kernel,
    image_vae_scatter_kernel,
    initialize_attention,
    narrow_bias,
    online_softmax,
    pack_image_attention,
    quantize_kernel,
    rotate_kernel,
    scatter_image_attention,
    widen,
    count
};
inline constexpr std::size_t cuda_kernel_count = std::size_t(CudaKernel::count);

struct CudaKernelMetrics {
    std::uint64_t calls = 0;
    double host_seconds = 0;
    std::uint64_t measured = 0;
    double gpu_seconds = 0;
};

struct CudaCallMetrics {
    std::uint64_t calls = 0;
    double host_seconds = 0;
};

struct CudaPhaseMetrics {
    std::array<CudaKernelMetrics, cuda_kernel_count> kernels{};
    // Requested cudaMemcpyKind, including Default; not an inferred direction.
    std::array<CudaCallMetrics, 5> copies{};
    CudaCallMetrics sync;
    std::uint64_t dropped = 0;
    std::uint64_t unavailable = 0;
    std::uint64_t shared_batch_size = 1;
    bool observed = false;
};

struct CudaPhase {
    std::mutex mutex;
    CudaPhaseMetrics metrics;
};

inline thread_local std::shared_ptr<CudaPhase> cuda_phase;
// CUDA code installs a nonblocking finalizer; CPU-only workers need no CUDA link.
inline thread_local void (*cuda_finalize)() noexcept = nullptr;

class CudaPhaseScope {
public:
    explicit CudaPhaseScope(bool enabled, std::size_t batch_size = 1) noexcept
        : previous_(cuda_phase) {
        if (enabled) {
            try {
                context_ = std::make_shared<CudaPhase>();
                context_->metrics.shared_batch_size = batch_size;
            } catch (...) {
                // Telemetry allocation failure cannot fail inference.
            }
        }
        if (cuda_finalize)
            cuda_finalize();
        cuda_phase = context_;
    }

    explicit CudaPhaseScope(std::shared_ptr<CudaPhase> context) noexcept
        : context_(std::move(context)), previous_(cuda_phase) {
        if (cuda_finalize)
            cuda_finalize();
        cuda_phase = context_;
    }

    ~CudaPhaseScope() noexcept {
        if (cuda_finalize)
            cuda_finalize();
        cuda_phase = previous_;
    }

    const std::shared_ptr<CudaPhase>& context() const noexcept {
        return context_;
    }

    CudaPhaseMetrics snapshot() const noexcept {
        if (cuda_finalize)
            cuda_finalize();
        if (context_) {
            try {
                std::lock_guard lock(context_->mutex);
                return context_->metrics;
            } catch (...) {
            }
        }
        return {};
    }

    CudaPhaseScope(const CudaPhaseScope&) = delete;
    CudaPhaseScope& operator=(const CudaPhaseScope&) = delete;

private:
    std::shared_ptr<CudaPhase> context_, previous_;
};
}  // namespace kadan::serving
