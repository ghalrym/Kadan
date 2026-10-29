# Screenshot understanding

The `lfm3_vision` worker runs released BF16 `LiquidAI/LFM3-VL-2B` using the checkpoint's custom Transformers code. The normal Compose deployment starts it by default. It serves the distinct `vision` feature through Hikari's existing queue and memory scheduler. Image and video generation retain their existing workers. The [agent action expectations](../../user-expectations/kadan/agent-actions.md) describe the user-visible limits.

This follows Andrew's later direction to make LFM conversation and screenshot understanding the default while retaining other consumers during migration. It adds no dense model to a hardware tier. The pinned checkpoint is `f83d2b0a2296aaa277f394990a9b8b949ddf9691`; its complete runtime artifact manifest is `services/hikari/experimental_worker/catalog_vision.json`. Downloads require the existing explicit license acknowledgement. Loading uses only the verified local files. The model's pinned Python code executes in the worker, so license and code review belong before download and execution.

The documented runtime is Transformers 5.21.0 or newer, PyTorch, torchvision and Pillow. The release card says vLLM support is in progress, while SGLang and llama.cpp support are under review. This worker does not assume Hikari's native inference engine supports LFM3. See the [official pinned model card](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/README.md).

`POST /v1/vision/describe` accepts a PNG or JPEG as `image_base64`, its `media_type`, and a `question`, with optional `max_new_tokens`. The response is `{model, text}`. No external image URL or worker-local path is accepted. Binary input is limited to 2 MiB, decoded dimensions to 4,194,304 pixels, total context to the checkpoint's actual maximum of 128,000 tokens. Questions and returned descriptions have no additional character ceiling. An omitted or null `max_new_tokens` uses all context remaining after the processed screenshot and question; an explicit positive value can reserve a smaller reply but must fit that same context. Oversized work fails explicitly; the checkpoint's own processor settings govern image preprocessing. The pinned `text_config.max_position_embeddings` is 128,000, rather than 131,072. Prompt tokens include processed image tokens, and the requested reply is reserved within that same maximum. The release card's 16,384-token generation example is not an architectural output cap. A terminal EOS is required even for a short response, and an EOS at the exact requested boundary is accepted. No input is truncated to fit. A provisional 8 GiB host and 16 GiB GPU grant admits one request on one BF16-capable CUDA GPU. The checkpoint has about 3.6 GiB of BF16 weights; its six full-attention layers need about 1.5 GiB of BF16 KV state at maximum context (four KV heads, 128 values per head), in addition to sliding-window state, vision processing, prefill activations and runtime overhead. SDPA attention and last-token-only logits avoid eager attention matrices and full-context vocabulary logits. This estimate informs admission; it is not measured evidence that maximum context fits. Allocation failure reports an error instead of reducing context.

Each request uses an isolated model process. There is no prefix cache, persistent loaded model, video understanding, multi-image input, or direct desktop control in this adapter. Hikari cancels through the worker owner; the process must finish or be killed and joined before resources are released. Maximum-context fit and broader screenshot quality require further manual trials.

Before queue admission, the screenshot endpoint decodes and bounds the payload
and checks that its PNG or JPEG signature matches `media_type`. A mismatch returns
422 without submitting inference. The shared check also runs in the executor
before inference dependencies load. These signatures match Pillow's format
classification; they do not establish that the complete image is valid. Pillow
decoding, format verification and the pixel limit remain in the worker.

The worker uses its own `lfm3_vision_models` volume; it does not mount the native model store. The API and worker require `HIKARI_ENABLE_LFM3_VISION=true`. Kadan's runner additionally requires `KADAN_EXPERIMENTAL_LFM3_VISION=true`. Only then, and only with a configured computer, does it bind `computer_look`. The tool captures the computer's current screenshot itself and sends its bytes to Hikari. The runner's existing command validation, saved transcript, ownership, cancellation and result-size limits remain in charge. In particular, `computer_look` still marks and limits its tool observation to the shared 4,000-character transcript policy; the direct Hikari vision API returns the complete model description. That agent-tool policy is separate from the model's output budget and is not a claim that the checkpoint supports only 4,000 characters. Disabling the flag removes the tool from later model decisions and rejects an older saved vision command before screenshot capture.

The [shared worker setup](../../../services/hikari/experimental_worker/README.md) uses the existing Compose project and usual Hikari and Kadan URLs. The vision worker needs no profile; Compose defaults `HIKARI_ENABLE_LFM3_VISION` and `KADAN_EXPERIMENTAL_LFM3_VISION` to true. Explicit overrides still apply. Unit tests use mocked processors and queues and run no inference.

## Manual trial evidence

An owner-authorized BF16 trial downloaded the pinned checkpoint through Hikari
Settings and executed Kadan's `computer_look` against the live computer screenshot.
The returned description correctly identified the Google page, search bar and
visible shortcuts. Two cold vision executors completed in 7.19 and 6.74 seconds.
Peak RSS was 4.43 GiB, CUDA allocation 3.71 GiB and reservation 3.75 GiB. This
verifies this screenshot path, not arbitrary desktop understanding or maximum
context fit. No computer input or control was needed for this observation.
