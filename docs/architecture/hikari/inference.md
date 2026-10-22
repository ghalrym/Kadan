# Inference architecture

Hikari manages model loading and memory allocations across inference workers so users can switch inference types without manually unloading models. The user-visible contract is described in [Inference operations](../../user-expectations/hikari/inference.md).

## Mixture-of-experts language models

Every hardware tier uses a mixture-of-experts (MoE) language checkpoint. Hikari is designed to run checkpoints larger than GPU memory by keeping routed experts across GPU memory (VRAM), system memory (RAM) and disk. The complete checkpoint does not have to fit in VRAM.

Frequently used routed experts should stay in VRAM where capacity permits. Other routed experts can remain in RAM, with disk holding those that do not fit in either memory tier. Placement adapts to expert use within the memory available to the language model. Only routed experts move between these tiers. Non-expert matrices, including shared experts, remain pinned in VRAM while the language model is loaded, so sufficient VRAM for those weights and execution state is still required.

Experts held in RAM execute on the CPU alongside GPU work. Hikari balances that work against streaming experts to the GPU using measured execution and transfer rates. It should not send every RAM-resident expert across the GPU connection by default.

Offloading preserves the selected checkpoint rather than substituting a smaller or dense language model. This allows models larger than VRAM to run, with performance depending on the memory available and the execution and transfer rates of the hardware.

## Memory accounting

Workers on the same computer share that computer's memory capacity, including when they run in separate containers. Hikari accounts for RAM and each physical GPU separately. Several GPUs may participate where supported, but their VRAM is not one contiguous allocation; each GPU must satisfy its own placement requirements. Workers on different computers use their respective computers' resources.

Concurrent requests require enough RAM and VRAM for each worker. Requests wait when their combined requirements exceed available memory. Placement admission must remain independent of the total number of expert slices.

## Memory priority and reclamation

Priority governs access to memory in this order, highest first:

1. Chat and decisions.
2. Speech synthesis.
3. Transcription and transcription formatting.
4. Image generation.
5. Video generation.

A higher-priority request does not interrupt inference already running. It may wait for an active image or video request to finish if there is insufficient memory to run alongside it.

Hikari first tries to make room by reducing an idle worker's memory allocation before unloading its model entirely. It reclaims memory from lower-priority workers first, and from the least recently used workers within a priority. An idle worker with no queued requests can yield memory even to a lower-priority type. A worker with queued requests yields only to a strictly higher-priority type. Memory becomes available to another worker only after its release is confirmed.

Where possible, Hikari keeps enough room for the selected language tier's minimum requirements so chat can start while another type runs. This reservation is best effort: if preserving it prevents other work from fitting, Hikari can let that work proceed. Chat therefore has no unconditional immediate-start guarantee, and sustained higher-priority demand can delay lower-priority work.

For example, an idle chat model holding more memory than it needs can have its allocation reduced to admit an image request. An active video request continues to completion even if chat is waiting for its memory.

## Allocation changes and model lifecycle

Models load on demand. Moving individual MoE experts between memory tiers is distinct from changing a worker's overall memory allocation. Currently, shrinking or growing a resident worker's allocation unloads its whole engine; its next request reloads the model within the new allocation. Switching workloads under memory pressure can therefore incur a full model reload.

When work is no longer waiting for memory, idle workers can regain available capacity, with higher-priority types favored. Extra memory may allow more experts to remain resident, but does not establish a fixed throughput improvement.

The API hub also expires resident models after `HIKARI_MODEL_IDLE_SECONDS` (default 3600, positive integer). Its monotonic clock resets when an inference or lifecycle request finishes, for every worker type. A reconnect starts a fresh idle period because completion history is unavailable. Request scheduling runs first; expiry waits while any worker on that computer has outstanding inference or lifecycle requests. It uses the same confirmed offload path as memory reclamation, retaining the old allocation charge until cleanup completes. An unloaded worker receives another budget only when a request needs it.

Requests must respect their actual input and memory requirements. Hikari must not silently truncate chat history or reduce requested media settings to force a fit. Cancellation must finish cleanup before the released resources are reused.

Concurrent throughput, model quality and physical memory fit require manual validation on the target machine. Scheduling and offloading policies alone do not establish measured latency or guarantee that every combination of inference requests fits.

## Reusable conversation state

Each resident chat engine keeps one completed conversation prefix, its exact committed token sequence and the corresponding attention and recurrent state. Both supported GGUF architectures are recurrent, so a hit requires the entire retained sequence to match the start of a longer prompt; an arbitrary common prefix cannot safely rewind recurrence. Selected output tokens not yet fed through the engine are excluded. On a hit, only the remaining prompt suffix executes, at the retained absolute position.

Request cleanup releases transient scratch while keeping prefix allocations charged to the worker's existing RAM and per-GPU budgets. Missing conversation IDs and disabled reuse retain no entry. A miss resets state, and cancellation discards incomplete state. Cache response counts describe actual reuse, and prefill timing counts only tokens that executed.

The inference owner can detach one completed prefix before model eviction or budget changes. That RAM snapshot survives destruction of the engine and contains no weights, open checkpoint files or device allocations. Attention keys/values already have complete host copies. Device recurrence is copied back synchronously, including convolution history; host recurrence keeps its execution mode. Qwen4Exp additionally retains QSA index keys/pools, PLE convolution history and recent tokens. On restoration, device recurrence is uploaded lazily on its newly selected device. If it cannot be admitted before the first restored step, the complete host state remains available for the host path. KV device mirrors can be rebuilt from the host copy.

Restoration requires the same model ID, workload, checkpoint path, context, tokenizer file identity and every GGUF tensor's immutable shard identity. The exact committed token/conversation match still decides whether any prefix is reused. An incompatible request replaces the one retained entry. There is no cross-worker affinity or disk-cache layer.

Detached bytes remain charged as a host-only worker budget after confirmed offload. A second scheduler offload can reclaim that RAM before admitting another job, with the same busy/queued-work rules. Physical RAM pressure can also discard optional state; cache admission failure must not prevent model eviction or make a job wait indefinitely. Engine cleanup still independently proves that its device/model allocations are gone.

`HIKARI_KV_CACHE_IDLE_SECONDS` uses the existing typed API settings and budget-control path, defaulting to 3600 positive seconds. The execution thread checks expiry only while its queue is empty; it never interrupts a running request. A successful generation starts the retained state's idle period, and model unloading, budget changes or model-loading requests do not extend it. Expiry releases reusable device state or the detached RAM entry independently of model expiry. Worker restart loses all reusable state; checkpoint files and conversation history are separate.

## Workers on another computer

`compose.yaml` also runs workers on a computer other than the API's. Configure
`.env` as described in the [root README](../../../README.md#docker-compose), then start selected workers with
`--no-deps` so the local API and database stay stopped. Set the remote API URL
and its worker token; source edits rebuild and restart these workers too:

```sh
HIKARI_API_URL=ws://API_HOST:8000/v1/workers HIKARI_WORKER_TOKEN=... \
  docker compose -p kadan-workers up --build -d --no-deps hikari-worker-image hikari-worker-video
```

To reuse a model store created by the former standalone worker configuration,
set `HIKARI_MODEL_VOLUME=kadan-workers_worker_models` in that computer's `.env`.
Otherwise the workers project uses `kadan-workers_model_data`. Keep the existing
project name if it differs from `kadan-workers`.

Compose publishes the API on `0.0.0.0` (all IPv4 interfaces). Remote workers
connect using the server's reachable address; use TLS for `wss://`.
