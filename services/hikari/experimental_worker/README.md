# LFM3 BF16 workers

The fixed conversation, screenshot and audio workers use the existing Hikari queue, worker token and physical memory scheduler. They are ordinary services in `compose.yaml`; there are no model tiers, enable flags or extra LFM overlays. Hikari and Kadan keep their usual URLs. Qwen Image Turbo, H3 video, Whisper file transcription and Laya decisions keep their native workers.

No weights ship in the image. Each request loads a pinned local checkpoint in a separate executor and releases it after completion or cancellation. No resident conversation model or prefix cache is retained. Workers reconnect after transport loss, but unconfirmed work is not replayed.

## Apply the stack

Use the existing Compose project, environment and model volumes. Between manual trials, rebuild and recreate the affected services:

```sh
docker compose up -d --build --no-deps \
  hikari-api hikari-frontend kadan-api kadan-runner kadan-frontend \
  hikari-worker-lfm3-text hikari-worker-lfm3-vision hikari-worker-lfm3-audio
```

Retired native chat, TTS and formatting containers are not services in this configuration. After checking there is no active work, stop those three obsolete containers explicitly using the previous deployment configuration. Do not use a broad orphan-removal operation or delete model volumes. Keep image, video, transcription, decisions and the existing storage dependencies available.

In Hikari Settings, review each model's linked license and custom-code notice, explicitly acknowledge the intended downloads, then wait for **Complete**. Starting a container never accepts a license or downloads weights. A completed download establishes acquisition, not successful inference or physical fit.

## Manual verification

- Hikari Chat uses the fixed LFM3 conversation model. Send a short request, then try a longer answer and cancellation. There is no alternate chat model picker.
- In Kadan, test a saved conversation and a harmless read-only tool request. Compare a screenshot answer with **Watching**. Existing sessions, personas, command permissions and durable action recovery remain authoritative.
- Generate and inspect a Qwen Image Turbo image and an edit. Keep video available and test its retained path separately; passing language tests does not verify media generation.
- Test the speech form's locale and gender controls, listen to the completed WAV, and check cancellation. For calls, use HTTPS or localhost with browser microphone permission, then verify playback, interruption, mute and reconnect. The fixed-stack voice replacement still needs these live usability checks.

Record elapsed time, actual memory use and errors. Text has 262,144 total context tokens, screenshot understanding 128,000 and audio 32,768. Input and output share each budget; oversized work fails without truncating history. Maximum context does not guarantee physical fit, useful output or simultaneous capacity.

See [text architecture](../../../docs/architecture/hikari/worker-lfm3.md), [screenshot architecture](../../../docs/architecture/hikari/worker-vision.md) and [audio architecture](../../../docs/architecture/hikari/worker-lfm3-audio.md) for checkpoint pins, resource grants, call contracts and measured limits.
