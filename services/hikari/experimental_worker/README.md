# LFM3 BF16 evaluation

These opt-in Python worker containers provide an experimental exception to the
native language worker's MoE policy. They use Hikari's queue, worker token and
memory grants. They do not select a hardware tier, replace native workers, or
accept a model license. No model weights ship in the image.

The conversation worker loads the pinned released `LiquidAI/LFM3-1.7B` BF16
checkpoint with its custom Transformers code. Each request starts a separate
executor and loads the model again. Cancellation joins that process before
releasing memory. There is no persistent model or prefix cache in this trial.
Replies are buffered before delivery over the existing SSE chat interface.

The trial reserves a provisional 16 GiB host memory and 20 GiB on one GPU for text, bounds total
context to 262,144 tokens, and reserves 1,024 tokens for output. Prompt overflow
fails without truncating history. Physical fit, speed, conversation quality and
tool reliability require a manual target-machine trial. A unit-test pass, build,
connected worker or completed download does not establish inference success.

## Morning conversation trial

1. Review the draft PR and its exact pins. Keep the existing Compose file,
   containers and model store. Choose the experiment checkout explicitly for
   each command; avoid a broad `docker compose up` or `down`.
2. Build only the additive worker (this does not start it):

   ```sh
   docker compose -f compose.yaml -f compose.lfm3-text.yaml --profile lfm3-text build hikari-worker-lfm3-text
   ```

3. After deciding to run the reviewed code, set `HIKARI_ENABLE_LFM3_TEXT=true`.
   The reviewed Hikari API must also run with that flag to admit the worker and
   explicit model requests. Recreating the API is an owner deployment step;
   an already-running API from master does not gain these changes through the
   new worker. Preserve its existing database, volumes and worker token.
4. Start only the new worker against that reviewed API:

   ```sh
   HIKARI_ENABLE_LFM3_TEXT=true docker compose -f compose.yaml -f compose.lfm3-text.yaml --profile lfm3-text up -d --no-deps hikari-worker-lfm3-text
   ```

5. Open Hikari Settings and find **Experimental models**. Read the model's
   linked LFM Open License v1.0 and custom-code notice. After accepting the
   license, check its acceptance box and choose **Accept and download**. Wait
   for this model to show **Complete**; failures remain visible there. The
   existing model download API is also available. No credentials are created
   by this trial. Stop if upstream access requires account or license approval.
6. First compare a short explicit `model: "lfm3-1.7b-bf16"` chat request with
   the unchanged native model. Check identity, sensible content, error handling
   and cancellation. For the Kadan trial, run the reviewed API and runner with
   `KADAN_EXPERIMENTAL_LFM3_TEXT=true`. Existing agent sessions, personas, tool
   permissions and durable action recovery remain the authority. Use a short
   conversation; start with a harmless read-only action. Inspect the real result
   before trusting generated claims. Test stopping a pending reply and retrying.
7. Confirm an explicitly selected native model still works and image/video
   services are still available. Record actual response time, errors and GPU
   usage. A single-GPU trial fitted 261,120 prompt tokens plus the 1,024-token
   reply reservation; repeat admission and quality checks for your hardware.
   Concurrent full-window requests have not been validated.

To leave the trial, unset the Kadan experiment flag and return its API and runner
to the owner's existing configuration. Stop only `hikari-worker-lfm3-text` using
the same Compose files. Leave downloaded checkpoints and all native workers in
place. Disabling API experiment flags rejects future experimental requests.

Vision and audio are separate additive draft changes with their own profiles
and trial instructions. Audio's initial integration returns buffered speech
segments through the existing call interruption and playback-acknowledgement
protocol; it makes no claim of full-duplex model inference.

Workers reconnect with capped backoff after an API disconnect and refresh their
resource identity. Unconfirmed requests are not replayed. Each additive Compose
worker also restarts unless explicitly stopped.
