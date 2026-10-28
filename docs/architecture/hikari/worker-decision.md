# Decisions worker (`decision`)

Answers structured decisions (choices, scores and Noul answers) with the pinned
Laya model. It is small and CPU-only so agents get fast answers without taking
GPU memory from generation.

## Program and container

- Its own program, `kadan-worker-decision`, built from
  `services/hikari/inference_worker/workers/decision/`. It links only this engine;
  the shared worker runtime supplies the API contract, the WebSocket client and
  the model store.
- Its own image, built by `workers/decision/Dockerfile`: no CUDA.
- Its own Compose service, `hikari-worker-decision` (CPU only). It can be scaled, and
  run on another computer using the same services in `compose.yaml`.
- It connects to the API and takes memory only as the API grants it. Workers on
  one computer share its RAM and GPUs in the API's ledger.

## Serves

- `/v1/decisions`.

## Engine

- Laya on the CPU, inside a fixed 3 GiB host envelope.

## Memory

- **Minimum and preferred:** the 3 GiB envelope in RAM. It uses no VRAM.

## Scheduling

- Priority 1, shared with the language model. Lower-priority workers leave its
  RAM minimum free while it is not loaded.
- Its budget never includes a GPU.

## Constraints

- Keep it host-only. Giving it VRAM would compete with the language model for
  no benefit.

## Typed decision contract

`POST /v1/decisions` requires nonblank state of at most 8,000 characters and one to eight questions. The serialized request must fit 16,000 characters and the specialist's token budget. Question keys and Choice option keys are unique and nonblank, and instructions and Score levels must be nonblank. Rejecting ambiguity here keeps returned answers paired with their intended questions.

Choice returns an option key and optional probabilities. Score is the expected zero-based rubric index and can be fractional; the UI also shows the nearest level and available level probabilities. Noul returns a probability from zero to one, with optional true/false guidance. Invalid typed model output is a failure rather than fabricated answers.

The playground starts with empty state and no questions. New Choice questions have one empty option; Score questions have two empty levels. Answers remain associated with the evaluated input, and later edits mark them outdated. Retry uses the current form. Cancellation ignores late responses while native cleanup finishes.

See the user expectations for [decisions](../../user-expectations/hikari/decisions.md).
