# Cached-only block compilation experiment

Harness-only follow-up to the failed whole-block regional compilation in MR119.
No API compilation setting, production deployment, checkpoint or quality change.

CachedOnlyBlock wraps each original bound forward. Its ordinary Python dispatcher
calls extraction eagerly and calls torch.compile(original_forward) only for cached
mode. Storage audits also remain outside the graph. No clone/contiguous/detach
inside a compiled graph is relied upon for storage isolation.

After each extraction and every cached call, all K/V tensors must own storage of
exactly numel×element_size bytes, with zero offset, independent K/V and layer
pointers, and unchanged pointer/version/shape. The original27-token prompt requires
221,184bytes for each K or V;64tensors total14,155,776bytes(13.5MiB). Extraction and
first/final cached calls additionally hash all K/V bytes to reject content mutation.
Weak references reject reuse of any still-live prior request cache without keeping
its GPU tensors alive. Same-prompt extracted hashes must match; changed-prompt
hashes must differ. Hash/audit time is recorded separately and both eager/compiled
paths have the same instrumentation; exclude audited call1/2/40 and profiled call10
from steady-step speed comparisons.

FIFO: text A; eager original B; text C; compiled cold original D; text E; compiled
warm original F; text G; compiled changed prompt H; text I; eager changed prompt J;
text K. Each image runs40steps at2048²/BF16/seed42. Warm F must not recompile after
park/unpark. Changed prompt compilation is recorded, not presumed absent. Compare
B/D/F and H/J pixels and inspect images before any quality claim.

Uses the same explicit32GiB compiler allowance within128GiB no-swap benchmark
limit,5GiB ext4 disk cache limit,22GiB single-GPU accounting and physical/thermal
supervisor guards as the prior corrected experiment. Live GPU0 is not exposed.
Manifest and exact host-specific launch arguments are adjacent. Six CPU tests
cover dispatch, backing views, request reuse and cached mutation. The GPU run was safely stopped before compilation when the user prioritized activating reviewed MR119 on the home API. Model cleanup and container exit were confirmed; GPU1 returned to baseline before deployment. No compiled result or speedup is available. Do not interpret this candidate as a recommended production option.

## Review requirements before any resumed GPU run

The source review accepted the eager-extract/cached-compile dispatch. Before
resuming, extend per-image deadlines to changed-prompt H/J; assert parked reserved
and physical VRAM against retained accounting, not allocated bytes alone; and
require explicit B/D/F and H/J output comparisons before a correctness success.
The recorded audit_seconds includes GPU waits from hashing and must not be
subtracted as pure instrumentation overhead. The paused manifest is historical;
update it and record new checksums before running again.
