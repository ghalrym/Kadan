# LTX-2.5 distilled BF16

This checkpoint PR adds native text-to-video with synchronized audio using the
upstream two-stage `DistilledPipeline`. Kadan owns admission, one job worker,
RAM/VRAM reservations, cancellation, process cleanup and atomic output publication.
No ComfyUI or external inference server is involved.

Dependencies: the native video engine and completed multi-component catalog PRs.
The five required artifacts total approximately 66 GiB, per the official README;
actual download disk admission uses the immutable manifest sizes plus reserve.
Only the distilled BF16 checkpoint is included. Dev, quantized variants and DFR
are separate capabilities. No throughput or 3090 compatibility claim is made.

Sources checked October 5, 2026:
- https://github.com/Lightricks/LTX-2/tree/9ec55f9f22798a3198d9c923856824821bc3317e
- https://huggingface.co/Lightricks/LTX-2.5/tree/da86602791e888542b1c755f0b5df376fdb1a3af
- https://github.com/Lightricks/LTX-2/blob/9ec55f9f22798a3198d9c923856824821bc3317e/LICENSE-2_x

The worker requires a separate Python environment with `api/requirements-ltx.txt`.
An operator sets `KADAN_LTX_PYTHON` to its executable. Nothing is installed on
startup. This separation avoids replacing Kadan's LLM Transformers version.
The catalog download uses an existing `HF_TOKEN` if present; users must have
accepted the gated repository's terms independently. Tokens are never forwarded
on redirects. This PR neither creates credentials nor requests access.

Inference is fully offline, uses CPU weight offload, and reserves twice the
checkpoint size in host accounting and one GPU's full managed budget. These are
conservative admission estimates, not measured peak guarantees. The OS process
exit frees actual allocations before reservations are released. Decoder padding
is center-cropped and temporal alignment frames/audio are trimmed to the requested
size and duration. Distilled sampling uses upstream checkpoint-version defaults;
negative prompts are rejected because that path does not use CFG.

Fixture tests cover cancellation, failures, output publication and memory lease
cleanup. No weights were downloaded; actual GPU/weights inference and worker
installation were not run. Hardware memory/performance validation remains required.
Jobs are process-local and are not restored after restart; completed files remain
on disk. This implementation does not claim durable history.
