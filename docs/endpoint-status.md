# Endpoint implementation status

| Feature | Implemented behavior | Remaining work |
| --- | --- | --- |
| Chat | Browser conversation, native loaded-model call, cancellation/errors | Actual catalog/CUDA validation; server conversation persistence is absent |
| Decisions | Same runtime, strict JSON answer validation, editable questions | Actual model output quality; grammar-constrained decoding is absent |
| Images and edits | Typed forms/history/error handling | Model provider, source ingestion, image artifacts/history |
| Video | Typed parameters, actual-returned-job polling, empty initial jobs | Model provider, execution queue, artifacts/history |
| Speech and cloning | Typed discriminated voice requests and empty history | Model provider, sample ingestion, playable artifacts/history |
| Transcription | Typed audio-reference/formatting request and errors | Model provider, audio ingestion, actual formatting |
| Requests and metrics | Bounded process-local observations, live memory where available | Durable history and measured model token timing are absent |
| Settings | Verified checkpoint downloads and persisted selection/runtime controls | Other modality model catalogs and configuration |
| API access | Live OpenAPI and existing interactive docs | No keys/authentication changes are included |

Media generation endpoints currently return HTTP 503. Their forms are API wiring,
not functional image/video/audio inference. Input references are not fetched and
files cannot be uploaded. The backend never substitutes canned success results.
The README provides the shared-resource goal but no designated media checkpoints
or provider integrations; old model names existed only in fixture data.

Before implementing media execution, choose for each modality:

1. Checkpoint/version and license, output quality target, and supported operations
   (for example text-to-image versus editing, described voices versus cloning).
2. Target hardware budget and acceptable latency, precision and offload strategy.
3. Local input ingestion and validation rules, artifact storage/retention, and
   delivery contracts. Video also needs durable job/cancellation semantics.

Every future model adapter must use Kadan's existing shared ResourceManager:
reserve host and per-device memory, acquire active leases, complete in-flight work
before eviction/freeing, and cooperate with cancellation. A separate external
inference engine or resource accounting bypass is not part of this design.

Current verification uses tiny synthetic CPU architecture references, controlled
HTTP responses and real unavailable/error paths. No full catalog checkpoint,
CUDA inference, actual media generation, measured peak memory or throughput has
been validated. Nothing in these feature branches has been merged or deployed.
