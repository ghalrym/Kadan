# LFM3 BF16 workers

The default conversation and screenshot workers use the existing Hikari API,
queue, worker token and memory scheduler. They start without profiles in the normal
Compose project, keeping Hikari and Kadan's usual URLs and saved conversations.
Workers on one computer share that API's memory scheduler.

These workers follow Andrew's later direction to make LFM the default. They do
not select a hardware tier, remove native workers or accept a
model license. No model weights ship in the image. Each request loads the pinned
checkpoint into a separate executor; cancellation joins that process before
releasing memory. Replies are buffered, with no persistent model or prefix cache.

## Update the existing stack

Run from the normal checkout with its existing Compose project and environment.
The following updates both workers and their callers. `compose.yaml` defaults
all four text/vision selection flags below to true; remove older false overrides
when adopting the defaults. Service and model-volume names stay the same.

```sh
export HIKARI_ENABLE_LFM3_TEXT=true
export HIKARI_ENABLE_LFM3_VISION=true
export KADAN_EXPERIMENTAL_LFM3_TEXT=true
export KADAN_EXPERIMENTAL_LFM3_VISION=true

docker compose up -d --build --no-deps \
  hikari-api hikari-frontend kadan-api kadan-runner kadan-frontend \
  hikari-worker-lfm3-text hikari-worker-lfm3-vision
```

This recreates the named services, so apply it between manual trials with the
existing Tsuchi and other dependencies available. Keep the existing volumes,
worker token and native workers. The API and worker flags gate admission and
execution separately; Kadan's flags enable its corresponding agent paths.

In Hikari Settings, find **Experimental models**, review the linked LFM Open
License and custom-code notice, and explicitly accept each intended download.
Wait for **Complete**. Enabling a worker never acknowledges the license or
downloads its checkpoint automatically. Upstream account or license prompts
still require the user's approval.

## Check the selected path

- Hikari Chat shows **LFM3 1.7B BF16** and explicitly requests that checkpoint.
  Retry and **New chat** keep it. Missing workers or downloads fail instead of
  using the native model selected by a tier.
- In the usual Kadan conversation, try a short reply and a harmless read-only
  action. Saved sessions, personas, tool permissions and durable action recovery
  remain authoritative. Compare a screenshot answer with **Watching** without
  taking control of the computer.
- Stop a pending request and check cleanup. Record response time, errors and
  actual memory use. A connected worker, completed download or unit-test pass
  does not establish inference quality, speed or physical fit.

Text has a 262,144-token total context; vision has 128,000. Generation can use the
space remaining after the complete prompt rather than a separate 1,024-token
cap. Input is not truncated to fit. The provisional grants are 16 GiB RAM /
20 GiB VRAM for text and 8 GiB RAM / 16 GiB VRAM for vision, each on one GPU.
Full-window quality and concurrent capacity need target-machine validation.
See [text architecture](../../../docs/architecture/hikari/worker-lfm3.md) and
[vision architecture](../../../docs/architecture/hikari/worker-vision.md) for pins,
contracts and measured evidence.

## Disable

Disable the corresponding Kadan and Hikari flags and reapply the affected API
and runner configuration. Stop only the selected `hikari-worker-lfm3-*` services
using the same Compose project. Keep downloaded checkpoints and native workers.
Workers reconnect after transport loss, but unconfirmed work is not replayed.
