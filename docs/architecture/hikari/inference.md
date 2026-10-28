# Inference architecture

Hikari routes a fixed model stack through connected workers and manages their shared physical resources. The [inference expectations](../../user-expectations/hikari/inference.md) describe observable waiting, cancellation and failure behavior. Model assignments and observed checkpoint availability are separate: the API must not infer a connected executor or completed download from a configured default.

## Execution and memory admission

LFM conversation, vision and audio run in separate Python worker containers. Each request waits for a hub grant, starts a disposable executor and releases its model and cache when that executor exits. The current BF16 implementations use one selected GPU; they do not combine several GPUs into one language-model allocation or offload experts. See [LFM workers](worker-lfm3.md) for checkpoint and execution details.

Native decision, transcription, image and video workers retain their existing specialist engines. Their allocations can span supported devices, but each physical GPU remains an independent capacity. Workers on one computer share one ledger, identified by kernel boot ID and GPU UUID; workers on other computers use separate resources. A budget grant permits loading and does not establish that an engine is resident or that every request fits.

Worker registration separates `physical_ram_free` from `ram_free`: the former initializes the computer's shared capacity, while the latter bounds that container's host grant. Both probes retain their existing headroom margins. Cgroup limits apply to the reporting worker, not every worker sharing its boot ID. Admission and idle growth respect both the shared ledger and the individual ceiling. Older registrations without the physical field conservatively initialize shared capacity from `ram_free`; later registrations never inflate an existing ledger.

## Scheduling and residency

Active priorities are LFM conversation, screenshot understanding, speech and transcript formatting together with decisions; then native transcription; then images; then video. A higher-priority request waits for active lower-priority work when necessary rather than interrupting it. Lower-priority idle allocations are reclaimed first, with release acknowledged before the ledger credits memory. An idle worker with no queued work can also yield to a lower-priority request.

Where feasible, the scheduler reserves the conversation worker's minimum while other work runs. This is best effort and gives no immediate-start guarantee. Sustained higher-priority demand can delay other work.

Changing a retained native engine's allocation unloads the whole engine; a subsequent request reloads within its new budget. Native resident models also expire after `HIKARI_MODEL_IDLE_SECONDS` (3600 seconds by default), measured from completion. Scheduling takes precedence, and expiry waits for active or queued work on that computer. LFM executors already release after every request and retain no reusable prefix or model between turns.

## Submission and cancellation ownership

A job owns every worker recipient whose send may have begun. Cancellation records an unsuccessful outcome, prevents further fanout or requeue, and waits for terminal acknowledgement or disconnect; a failed send is not proof of nondelivery. Running work is not silently replayed on another worker. Queued work can move only while it remains eligible and has not been cancelled.

For LFM work, cleanup terminates and joins the executor before reporting completion or cancellation and returning its allocation. Native workers retain their confirmed-release path. No path may silently truncate prompts, replace the checkpoint, reduce media settings or count unconfirmed cleanup as free memory. Physical fit, concurrent throughput and quality still require target-machine validation.

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
