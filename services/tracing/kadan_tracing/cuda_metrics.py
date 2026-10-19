"""Fixed native CUDA kernel protocol; no runtime labels or user data."""

CUDA_KERNELS = (
    "add_bias",
    "bf16_input",
    "cached_attention_combine",
    "cached_attention_kernel",
    "check_image_key_value",
    "delta_conv_kernel",
    "delta_heads_kernel",
    "delta_l2_kernel",
    "divide_attention",
    "finish_int8",
    "gguf_decode_kernel",
    "gguf_matvec_batch_kernel",
    "gguf_matvec_kernel",
    "h3_add_kernel",
    "h3_attention_divide_kernel",
    "h3_attention_init_kernel",
    "h3_finish_int8_kernel",
    "h3_finite_kernel",
    "h3_gated_residual_kernel",
    "h3_lora_add_kernel",
    "h3_narrow_kernel",
    "h3_normalize_kernel",
    "h3_online_softmax_kernel",
    "h3_qk_rotary_kernel",
    "h3_silu_gate_kernel",
    "h3_widen_kernel",
    "h3_widen_real_kernel",
    "h3_zero_bias_kernel",
    "image_frequency_kernel",
    "image_gates_kernel",
    "image_norm_kernel",
    "image_projection_status",
    "image_residual_kernel",
    "image_rotary_kernel",
    "image_scheduler_kernel",
    "image_silu_pair_kernel",
    "image_text_gelu_kernel",
    "image_text_norm_kernel",
    "image_timestep_silu_kernel",
    "image_vae_gather_kernel",
    "image_vae_norm_kernel",
    "image_vae_scatter_kernel",
    "initialize_attention",
    "narrow_bias",
    "online_softmax",
    "pack_image_attention",
    "quantize_kernel",
    "rotate_kernel",
    "scatter_image_attention",
    "widen",
)
CUDA_ATTRIBUTES = frozenset(
    {
        f"kadan.cuda.{name}.{field}"
        for name in CUDA_KERNELS
        for field in ("calls", "host_seconds", "measured", "gpu_seconds")
    }
    | {
        f"kadan.cuda.copy{kind}.{field}"
        for kind in range(5)
        for field in ("calls", "host_seconds")
    }
    | {
        "kadan.cuda.sync.calls",
        "kadan.cuda.sync.host_seconds",
        "kadan.cuda.dropped",
        "kadan.cuda.unavailable",
        "kadan.cuda.shared_batch_size",
    }
)
