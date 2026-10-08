# Qwen Image latency investigation

The real sequential-offload acceptance at source `0e84a7bfdf3fd351de7bb366ef584f46339e305d`
on 2026-10-08 produced a valid PNG, but its 498.965-second image forward is not
practical interactive latency. The successful image is separate from the failed
final text restore. GPU0 remains untouched.

## Existing measurement, unchanged settings

Exact checkpoint: Qwen-Image-2.1 revision `d26bb61231c349cf6b7896fa83353113880e1ba3`.
Pipeline: Diffusers `QwenImage21Pipeline`, pinned commit
`8d3c30bfda9b511c00992f40cff4170a5502814d`, cached runtime `eed6c0b12088`.
Device: physical GPU1 RTX3090 only. GPU path requested BF16, one2048×2048 image,
40 inference steps, seed42, default pipeline guidance, sequential CPU offload,
and existing VAE tiling. No precision/step/resolution/guidance change is proposed.

| Existing trace interval | Seconds | What it includes |
| --- | ---: | --- |
| Pipeline construction | 68.564 | Host loading, outside image forward |
| Forward to first denoising callback | 139.912 | Prompt encoding, preparation, first transformer work and transfers; not separately measured |
| Subsequent39 callback intervals | 336.131 | Denoising plus transfers, median8.618s, range8.587–8.703s |
| Last callback to forward return | 22.922 | VAE decode, postprocess and hook cleanup; not pure VAE timing |
| Forward total | 498.965 | Sum above |
| Forward return to B FIFO completion | 1.505 | Parking, PNG/manifest publication and service return |

The old traces cannot separate prompt encoding, CPU/GPU transfers, VAE decode
and output conversion more precisely. No unexpected CPU matrix execution has
been established. Torch CPU dispatch time alone is not evidence that tensors
were computed on CPU. GPU samples and output show execution but do not attribute
individual operators. Instrumentation is needed before naming a measured transfer
percentage or claiming an optimized time.

## Verified local component inventory

Read-only local checkpoint inspection (no GPU/model execution):

| Component | Safetensors file bytes |
| --- | ---: |
| QwenImage21Transformer2DModel,32 layers | 14,230,284,408 |
| Qwen3VLForConditionalGeneration text encoder,36 text layers | 17,534,339,488 |
| AutoencoderKLQwenImage21 VAE | 1,350,989,512 |
| Total | 33,115,613,408 |

The pinned pipeline declares `model_cpu_offload_seq = "text_encoder->transformer->vae"`.
Installed Diffusers `enable_model_cpu_offload` chains Accelerate hooks so the
previous component is offloaded before the next component moves to GPU; the
transformer can remain resident across denoising calls. This avoids repeatedly
moving its entire weights per step and permits VAE residency across decoder tiles.
The existing VAE256-pixel tiles/192-pixel stride imply121 decoder calls at2048²;
actual calls will be counted, not assumed as measured evidence.

## Explicit comparison mode and honest admission

`NativeImage(..., offload_mode="component")` selects whole-component CPU offload.
The default remains sequential. This constructor option is for the reviewed
acceptance harness; no UI/API request field or production environment default
changes. The pinned component order is checked before installing hooks; missing
component files or a largest component exceeding every single-device envelope
fails before execution. No capacities are pooled.

Component mode reserves the entire logical image-phase device capacity remaining
after active owners and persistent contexts, rather than the old8GiB sequential
estimate. With the proposed22GiB parent and two512MiB contexts, that reservation
is22,548,578,304 bytes (21GiB). It covers the largest simultaneously resident
component, retained prompt embeddings/latents, activations, KV, allocator workspace
and transfer/transition peaks. File sizes are sizing inputs, not measured peaks.

| Phase | Reserved device bytes | Difference after component file bytes |
| --- | ---: | ---: |
| Text encoder | 22,548,578,304 | 5,014,238,816 |
| Transformer | 22,548,578,304 | 8,318,293,896 |
| VAE | 22,548,578,304 | 21,197,588,792 |

These differences are the available phase allowance, **not measured activation
requirements or a claim that the candidate fits**. The first profiled run must
verify phase peaks, transitions, allocator reservation, and physical headroom.
Admission is accounting, not a hard CUDA allocator limit. Current fixed parent
contexts remain charged. Host transition accounting with complete packed text
cache is91.080GiB inside96GiB; see RESIDENT-IMAGE-BRIDGE.md. OOM or uncertain cleanup
uses the existing cleanup/quarantine path and cannot be reported as success.

## Targeted profiling and next acceptance

`api.inference.image.profiling.profile_pipeline` is opt-in acceptance-only. Install
it after offload hooks are attached, around the real pipeline call. It records
synchronized wall times for encode_prompt, transformer.forward, vae.decode and
image_processor.postprocess; counts decoder tile forwards without synchronizing
every tile; and records per-stage Torch peak allocated/reserved bytes. Stage
peaks include the outgoing component during transition. Nested tile timing is
CPU wall/enqueue time and must not be summed as GPU execution time.

An optional trace_path captures **only the first transformer call** with Torch
CPU/CUDA profiler, retaining the Chrome trace of kernels and memcpy timestamps.
The summary separates CPU dispatch from device time; overlapping events must not
be blindly added. Trace generation and boundary synchronizations introduce
profiling overhead. Capture unprofiled subsequent step medians separately.
Methods are restored on success or failure; production does not install this
instrumentation. No Torch compile, precision reduction or alternative kernels are
introduced.

After review, compare sequential and component modes using the identical prompt,
seed42, BF16,2048²,40 steps, guidance and VAE tiling. Measure prompt stage, per-step
calls, VAE/tile count, postprocess, publication, HtoD/DtoH trace time and peak memory.
Check output dimensions/visual quality and compare outputs, with deterministic
limitations stated. Do not claim speedup until measured. Lower resolution, fewer
steps, quantization or different guidance would be separate quality tradeoffs.
Keep the existing45-minute run/30-minute image limits, memory/thermal guards,
read-only model volume and dedicated Redis; never expose GPU0. Verify final text
restore and actual application cache reuse using retained bytes and unchanged
source-read counters, with300-second load/restore and30-second token deadlines.
