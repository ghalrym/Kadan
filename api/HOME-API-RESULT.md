# Home API MR119 activation: passed

The user explicitly authorized activation on2026-10-08. Reviewed source
`dea7b0fade2b068b8e142a1666bf39e495ad0d99` is serving the existing home API;
MR119 remains draft/unmerged. Exact-commit CI passed: API37836953566 and
native37836953520 (push runs37836947771/37836948039 also passed).

Only kadan-api-1 was recreated. It uses the verified cached runtime
`sha256:eed6c0b1208855f89e27093c2ad6abaada302919260787524e5c901ee56f57aa`, resident
worker SHA256`02309c0817596588b52f86586748fe4120ce81c3b652607fe6a27016e2496e58`,
KADAN_LLM_BACKEND=native-resident and KADAN_IMAGE_OFFLOAD=component. PhysicalGPU1
(UUIDGPU-2a2378dd-08c1-6f69-6317-a253d90e76b3) is the only exposed GPU, mapped to
logical0 for both text and images. GPU accounting is22GiB. Production host
capacity was339,875,476,275bytes(80% of available memory at initialization); this
is not the earlier isolated96GiB acceptance limit. Compilation remains disabled.

## Real HTTP verification

2026-10-08 20:21:52–20:27:39UTC, through the existing localhost8000 API:

| Request | Result | HTTP duration |
| --- | --- | --- |
| A: 2 plus 2 | 200, answer4 | 0.693s |
| B: original apple prompt,2048²/BF16/40steps/seed42 | 200, one PNG | 296.042s |
| C: 3 plus 4, submitted while B ran | 200, answer7 | 345.683s including FIFO wait |

Redis simultaneously showed image jobc69ecd15eaee4a9f8bd1bc0120b9873c running and
text job539c3b8adc334981a1e4f9a106c00dbf queued. C finished49.804s after B's HTTP
response, including restore and generation; that interval is not pure restore time.
The complete check took346.943s. The same native child PID42/session remained
throughout. Its20,825,156,824-byte backing reservation stayed admitted and its
anonymous RSS stayed about20GiB. Child rchar increased only1,423bytes between the
parked image phase and final text completion, consistent with the separately
validated full RAM cache, not a checkpoint reread. The public lifecycle endpoint
does not expose internal cache-hit/source-byte counters; those were verified in
the prior isolated acceptance, not directly read through HTTP here.

The2048×2048 PNG was visually checked and its SHA256 is byte-identical to earlier
accepted eager/sequential/component images:
`6df59d86af4143efd1e3e5b419812754bd2b9ed1b10ccd33c3bb2ecac4dbf42b`.
Published image5ddc2ceb-3ab1-4d9a-abb5-99002066e094 is served at
`/v1/images/5ddc2ceb-3ab1-4d9a-abb5-99002066e094/files/0`.
This is real generated media; no mock data was inserted into Postgres.

Final lifecycle: ready, saved small model/context65,536, no error. Image GPU
reservation was gone; image host residency and native backing remained reusable.
CPU maximum71.5°C, GPU1 maximum82°C, sampled GPU1 peak22,459MiB; no physical guard
or thermal event. /health returned ok directly and via the existing frontend
proxy. GPU0 was released as part of the authorized move to the proven same-card
configuration. The service stays running on GPU1 for review testing.

## Preservation and hot reload

Original tracked checkout remainsb6a6c901ccf6ab046cbb53cffb871490f57fdb86. Only its
local compose.override.yaml was changed for the authorized API configuration.
Other local files/settings, all H3/TTS/model mounts, Redis/Postgres volumes and
frontend configuration were preserved. Frontend6403e0a2c3ac, Redis8966f50c30db and
Postgres47d1968f154e were not recreated. Old API /tmp development artifacts(81MiB)
and exact previous override/container metadata were saved before recreation.

Uvicorn --reload --reload-dir /app/api remains enabled. The editable reviewed
source bind is:
`/home/andrew/Documents/Codex/2026-10-08/task-4/home-api-mr119/source/api/`.
It is an archive of the deployed commit; original checkout api/ edits do not alter
this review service. The source-revision.txt records its starting revision.
New generated media persists in the adjacentmedia/ bind. No whole image was built.

Rollback instructions and exact previous override are in
`/home/andrew/Documents/Codex/2026-10-08/task-4/home-api-mr119/DEPLOYMENT.md` and
`rollback/compose.override.yaml`. Restore that override to the original checkout,
then run `docker compose -p kadan up -d --no-build --no-deps --pull never --wait api`
there. This restarts only API and restores its previous GPU0 setup; preserve volumes.

Raw HTTP responses, PNG, queue/resource/native-IO/physical samples and monitor
script are underhome-api-mr119/evidence/ beside that deployment file. The separate
MR120 compilation experiment was safely stopped before image/compile execution
for this activation. It remains unvalidated and is not mounted into the home API.
