# Quantized Klein transformer loading

This shared engine adds no selectable checkpoints. A reviewed checkpoint registration must supply an immutable primary revision, matching pinned auxiliary bundle, and `ImageRecipe` with `quantization` (`fp8` or `nvfp4`) and `weight_filename` (the original relative path in that primary source). The existing `pipeline` field selects the matching installed Diffusers pipeline, including a separately verified KV pipeline when applicable. `ModelManager.get_checkpoint` validates the complete combined bundle before generation. Auxiliary sources provide transformer configuration, tokenizer, scheduler, text encoder and VAE; they must exclude duplicate BF16 transformer weights.

This is **dense fallback inference**: transformer weights are decoded to BF16 on CUDA or FP32 on CPU. It does not provide FP4/FP8 optimized kernels, activation quantization, or compressed inference residency. Download size is not the runtime memory requirement. Settings shows “Checking size” for unknown sizes and uses the manifest total once known.

## Supported serialized formats

FP8 matrix weights use E4M3FN or E5M2, with an optional scalar or output-row `weight_scale`. The stored dtype must agree with any per-layer `comfy_quant` JSON descriptor. Plain floating tensors remain dense. A `pre_quant_scale` vector multiplies input columns of the reconstructed matrix. Activation `input_scale` metadata is consumed without quantizing activations; numerical equivalence to an upstream quantized kernel is not claimed.

NVFP4 requires a per-layer `nvfp4` descriptor, two E2M1 values per byte in **high-nibble-first** order, E4M3 block scales for every 16 values, and scalar `weight_scale_2`. Block scales use padded cuBLAS SWIZZLE_32_4_4 storage. For output row `r` and scale column `c`, with `C` the number of scale columns padded to a multiple of four, the serialized byte address is:

```
((r // 128) * (C // 4) + c // 4) * 512
    + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
```

Scale storage pads rows to 128. Decoded values multiply block scale, global scale, and optional input-column scale. This independent address calculation and nibble conversion adapt to Kadan's existing canonical decoder; no ComfyUI or external quantized-kernel runtime is imported. Format facts were checked against [NVIDIA's block scale layout](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout) and the serialized-layout reference at Comfy-Org/comfy-kitchen commit `be003b7c23c5b01328657955b8bc5d3f073d868e`; no reference implementation was copied.

Unknown descriptors, incompatible shapes/dtypes, nonfinite decoded values, orphaned scale metadata, and missing local assets fail explicitly. A uniform `model.diffusion_model.` or `diffusion_model.` prefix is removed before `Flux2Transformer2DModel.from_single_file` converts original-format keys with the local `transformer/config.json`. The pipeline receives this transformer override and loads all remaining assets locally.

## Allocation and verification

A safetensors header read estimates dense storage before tensor loading. The shared resource manager admits the encoded bundle, two dense transformer copies, 64 MiB decoder scratch, and the existing image workspace allowance before model construction. CUDA placement/offload uses dense residency. Pipeline and transformer references, including failure traceback frames, are dropped before reservation release. Decoding checks cancellation between row tiles; pipeline callbacks retain the existing cancellation behavior.

Tests cover scaled FP8 linear algebra, high-first NVFP4 decoding and global/input scales, independent swizzle addresses across row and column tiles, malformed formats, admission before loading, and release after construction failure. Synthetic safetensors also exercise actual multi-source manifests, integrity checks, download publication, persisted image selection, the native transformer override and atomic image publication. The full pipeline constructors are mocked and companion weights are synthetic. An additional offline CPU smoke constructs a tiny one-double-block/one-single-block Flux2 transformer with local configuration, loads complete synthetic original-format FP8 and NVFP4 checkpoints through the real Diffusers `from_single_file` converter, and verifies every loaded tensor. This smoke runs with the complete pinned runtime; minimal environments without Diffusers skip it. No published weights were downloaded, no full model inference ran, and no GPU performance or peak-memory measurement was performed. Actual published payload/header compatibility still requires a separately authorized model run. The shared contract helper is available to each independent checkpoint registration's tests.
