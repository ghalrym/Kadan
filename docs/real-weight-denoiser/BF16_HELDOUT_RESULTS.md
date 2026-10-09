# BF16 held-out step results

Executed independently reviewed commit
`f0b4eefc615a6370fb6bb43a13264a0128e0ea13` with unchanged criteria and resource
bounds. Step 20 passed. Step 39 capture succeeded, but replay was interrupted by
the CPU-temperature guard and remains incomplete. Both windows restored the exact
API container with the native `small` model ready. No production dual-GPU path
was enabled.

| Case | Numerical outcome | Capture / replay seconds | Total API pause |
|---|---|---|---|
| Step 20 | 172 boundaries passed; maximum absolute difference 0, zero violations/nonfinite | 239.710 / 376.526 | 696.828 seconds |
| Step 39 | 60 completed boundaries bit-exact, through teacher block 19; remaining checks unexecuted | 359.683 / 153.720 before stop | 595.077 seconds |

Step 20 covered bit-exact transport ownership/roundtrip checks, all 32 full-shape
teacher blocks, independently advancing 1/4/32-block chains and final
norm/projection. Both ranks wrote matching passed verdicts. The successful context
packets totaled 8,876,858,825 bytes. Their byte-exact manifest and hash-bound cleanup
receipt were preserved before removing only owned temporary contexts. The step-39
admission check verified step-20 pass, restoration and cleanup before starting.

Step 39's launcher raised `CPU temperature unavailable or above guard` during
replay. The guard requires a nonempty CPU temperature list and maximum below 80°C.
The rejected sample is not persisted, so the exact triggering value is unknown;
the highest retained preceding sample was 75.75°C. Do not misrepresent that saved
maximum as the triggering temperature. The supervisor reaped its workers, the
container exited without OOM, and the API was restored. No tolerance, thermal
threshold or deadline was relaxed. No automatic retry occurred. All 35 step-39
context/manifest/ownership files and partial diagnostic reports remain locally.

The current reviewed launcher always captures before replay, so rerunning it
would duplicate contexts and threaten the shared 32-GiB envelope. The next useful
change is a separately reviewed replay-only admission path for these exact retained
contexts, with fresh resource admission and the same criteria/bounds. The incomplete
attempt must remain recorded. Full trajectory/image protocol preparation remains
conditional on a completed step-39 pass.

## Provenance and recovery

- Original immutable weight manifest:
  `15ef4ca04d95404a0467524015097e306f8dab871eca027635d716b843b469ce`.
- Step-20 context manifest:
  `191c779ee58a7477b33d26a8846ee864c787aa4fa80aa7d5c29bcedc93be0551`.
- Step-39 context manifest:
  `9ceff1294a8aecf4a3211ab952d5b8f3beeec151fc8640431d536e2257c0fafa`.
- Exact API container restored:
  `1efcf5729f776dd90bbc0611c63a474bc696cadc0d48b9011fbb320e167b6499`.
  Latest start: `2026-10-09T00:03:16.403255278Z`; healthy, native `small` ready.
- Source, container configuration/mount identity and unrelated services were
  checked by the launcher. GPU0 returned to 2 MiB after recovery.
- Exact executed source CI: API push [37860256948](https://github.com/ghalrym/Kadan/actions/runs/37860256948),
  native push [37860256906](https://github.com/ghalrym/Kadan/actions/runs/37860256906),
  API PR [37860262147](https://github.com/ghalrym/Kadan/actions/runs/37860262147),
  native PR [37860262219](https://github.com/ghalrym/Kadan/actions/runs/37860262219),
  all successful. The same source passed 38 CPU diagnostic tests.

Compact evidence is in `evidence-holdouts-f0b4eef/`: manifests, numerical reports,
rank verdicts, review/CI records, supervisor results, resource summaries and
sanitized restoration facts. No tensors or raw container configuration are committed.
Full local evidence remains under `holdout-reviewed-f0b4eef-step20` and
`holdout-reviewed-f0b4eef-step39` in the task workspace.

The first-slice result and step 20 are passed. Overall BF16 acceptance remains
incomplete; prior FP32 gates remain failed and candidate v1 remains rejected.
Independent full scheduler trajectories, decoded-image acceptance and API review
are still outstanding. These cached-step comparisons make no full-image speed or
production-readiness claim.
