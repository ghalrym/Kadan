#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "kadan/resources.hpp"

namespace kadan::image::gpu {

enum class Dtype {
    fp32,
    bf16,
};

inline std::size_t dtype_bytes(Dtype dtype) {
    return dtype == Dtype::fp32 ? 4u : 2u;
}

struct DeviceStats {
    Bytes allocated = 0;
    Bytes peak = 0;
};

// Tracked per-device CUDA allocator and cuBLAS handle with pedantic FP32 math
// (no TF32, no reduced-precision BF16 reduction).
void init_device(int device);
DeviceStats device_stats(int device);
void reset_peak_stats(int device);
void synchronize_device(int device);
void clear_device_workspaces(int device);
void destroy_device_handles(int device) noexcept;
void query_device_memory(int device, Bytes& free_bytes, Bytes& total_bytes);

class DeviceGuard {
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = -1;
    bool active_ = false;
};

class Tensor {
public:
    Tensor() = default;
    Tensor(int device, Dtype dtype, std::vector<std::int64_t> shape);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    bool defined() const {
        return data_ != nullptr;
    }

    int device() const {
        return device_;
    }

    Dtype dtype() const {
        return dtype_;
    }

    const std::vector<std::int64_t>& shape() const {
        return shape_;
    }

    std::size_t dim() const {
        return shape_.size();
    }

    std::int64_t size(std::int64_t axis) const {
        const auto d = std::int64_t(shape_.size());
        const auto idx = axis < 0 ? d + axis : axis;
        return shape_.at(std::size_t(idx));
    }

    std::size_t numel() const {
        return numel_;
    }

    Bytes nbytes() const {
        return bytes_;
    }

    void* data() {
        return data_;
    }

    const void* data() const {
        return data_;
    }

    void reset() noexcept;

    // Reshape metadata in place (element count must match).
    Tensor& reshape_inplace(std::vector<std::int64_t> new_shape);

    // Copy tensor to another (or same) CUDA device, optionally casting dtype.
    Tensor to(int target_device, Dtype target_dtype) const;
    Tensor to_dtype(Dtype target_dtype) const;
    Tensor clone() const;

    // Host <-> Device transfers.
    static Tensor upload_raw(
        int device,
        Dtype stored_dtype,
        Dtype target_dtype,
        std::vector<std::int64_t> shape,
        const void* host_bytes,
        Bytes host_nbytes
    );
    static Tensor from_cpu_float(
        int device,
        std::vector<std::int64_t> shape,
        std::span<const float> values,
        Dtype target_dtype = Dtype::fp32
    );
    static Tensor from_cpu_int64(
        int device, std::vector<std::int64_t> shape, std::span<const std::int64_t> values
    );
    std::vector<float> to_cpu_float() const;

private:
    void* data_ = nullptr;
    int device_ = -1;
    Dtype dtype_ = Dtype::fp32;
    std::vector<std::int64_t> shape_;
    std::size_t numel_ = 0;
    Bytes bytes_ = 0;
};

struct HostFloatTensor {
    std::vector<std::int64_t> shape;
    std::vector<float> data;

    bool defined() const {
        return !data.empty();
    }

    std::int64_t size(std::int64_t axis) const {
        const auto d = std::int64_t(shape.size());
        const auto idx = axis < 0 ? d + axis : axis;
        return shape.at(std::size_t(idx));
    }

    Tensor to_device(int device, Dtype dtype = Dtype::fp32) const {
        return Tensor::from_cpu_float(device, shape, data, dtype);
    }
};

// Download GPU tensor to host FP32 and verify every element is finite.
HostFloatTensor cpu_float(const Tensor& x);

// Linear layers: Y[..., N] = X[..., K] @ W[N, K]^T (+ bias[N]).
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr);

// Qwen3 text encoder primitives (FP32 activations, BF16 or FP32 weights).
Tensor embedding_bf16_to_fp32(const Tensor& weight, const Tensor& indices_i64);
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps = 1e-6f);
Tensor transpose_to_heads(const Tensor& x, std::int64_t heads, std::int64_t head_dim);
Tensor unheads(const Tensor& x);
Tensor text_rope_fp32(
    const Tensor& x, const Tensor& cos_table, const Tensor& sin_table
);
Tensor sdpa_qwen3_fp32(
    const Tensor& q, const Tensor& k, const Tensor& v, std::int64_t active_count
);
Tensor silu_mul_fp32(const Tensor& gate, const Tensor& up);
Tensor add_tensors(const Tensor& a, const Tensor& b);
void add_inplace_fp32(Tensor& dst, const Tensor& src);
Tensor concat_last_3_fp32(const Tensor& a, const Tensor& b, const Tensor& c);

// FLUX.2 denoiser primitives.
Tensor flux2_timestep_proj_bf16(int device, float sigma);
Tensor silu_bf16(const Tensor& x);

struct ModulationTriple {
    Tensor shift;  // FP32 [D]
    Tensor scale;  // FP32 [D] (already 1 + raw_scale)
    Tensor gate;   // BF16 [D]
};

struct DoubleModulation {
    ModulationTriple attention;
    ModulationTriple mlp;
};

DoubleModulation split_double_modulation(const Tensor& projected_bf16);
ModulationTriple split_single_modulation(const Tensor& projected_bf16);
std::pair<Tensor, Tensor> split_out_modulation(const Tensor& projected_bf16);

Tensor flux2_modulated_norm_bf16(
    const Tensor& x_bf16, const Tensor& shift_fp32, const Tensor& scale_fp32, float eps = 1e-6f
);
void flux2_gated_residual_bf16(
    Tensor& x_bf16, const Tensor& delta_bf16, const Tensor& gate_bf16
);
Tensor flux2_rope_bf16(
    const Tensor& x_bf16, const Tensor& cos_fp32, const Tensor& sin_fp32
);
Tensor sdpa_flux2_bf16(const Tensor& q, const Tensor& k, const Tensor& v);
Tensor flux2_swiglu_bf16(const Tensor& x_bf16);

// Axis manipulation helpers for denoiser (BF16).
Tensor concat_axis1_bf16(const Tensor& a, const Tensor& b);
Tensor concat_axis2_bf16(const Tensor& a, const Tensor& b);
Tensor slice_axis1_bf16(const Tensor& x, std::int64_t start, std::int64_t len);
Tensor slice_last_bf16(const Tensor& x, std::int64_t start, std::int64_t len);
Tensor concat_last_bf16(const Tensor& a, const Tensor& b);
Tensor stack_batch_bf16(const std::vector<Tensor>& items);
Tensor broadcast_batch_bf16(const Tensor& single_2d, std::int64_t batch);

// Euler & edit-mask helpers (FP32).
void euler_step_fp32(Tensor& latent, const Tensor& velocity, float dt);
void mix_source_noise_fp32(
    Tensor& latent, const Tensor& source, const Tensor& noise, float sigma
);
void apply_latent_mask_fp32(
    Tensor& latent,
    const Tensor& source,
    const Tensor& noise,
    const Tensor& mask,
    float next_sigma
);

// VAE primitives (FP32).
Tensor group_norm_32_fp32(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    bool apply_silu,
    float eps = 1e-6f
);
Tensor conv2d_fp32(
    const Tensor& x,
    const Tensor& weight,
    const Tensor& bias,
    bool downsample = false
);
Tensor upsample_nearest2d_fp32(const Tensor& x);
Tensor vae_sdpa_fp32(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const std::atomic_bool& cancel
);
Tensor nchw_to_seq_fp32(const Tensor& x);
Tensor seq_to_nchw_fp32(const Tensor& x, std::int64_t c, std::int64_t h, std::int64_t w);
Tensor vae_patchify_bn_norm_fp32(
    const Tensor& x_32,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    float bn_eps
);
Tensor vae_bn_denorm_unpatchify_fp32(
    const Tensor& latent_hw_128,
    std::int64_t h,
    std::int64_t w,
    const Tensor& bn_mean,
    const Tensor& bn_var,
    float bn_eps
);
Tensor vae_clamp_rgba_fp32(const Tensor& rgb);
Tensor slice_channel_fp32(const Tensor& x_nchw, std::int64_t c_start, std::int64_t c_len);
Tensor crop_2d_fp32(
    const Tensor& x_nchw,
    std::int64_t y_start,
    std::int64_t y_len,
    std::int64_t x_start,
    std::int64_t x_len
);
void paste_2d_fp32(
    Tensor& dst_nchw,
    const Tensor& src_nchw,
    std::int64_t dst_y,
    std::int64_t dst_x,
    std::int64_t src_y,
    std::int64_t src_x,
    std::int64_t h_len,
    std::int64_t w_len
);

}  // namespace kadan::image::gpu
