# Runner assignment and recovery

Tsuchi atomically assigns registered agent keys to stable runner identities, up to
`KADAN_RUNNER_CAPACITY` per runner (default five). All assigned agents count,
including agents with no queued work. Agents are logical records, not permanent OS
threads. One runner serves multiple agents; Hikari still schedules inference across
its existing workload workers. Image and pet queue loops remain independent of
LangGraph turns and do not consume additional agent slots.

When every runner is full, excess registered agents and their queued work wait in
durable storage. Idle owners keep their slots; stopping a process does not free them.
Each process registers a fresh session UUID and renews a 60-second infrastructure
ownership lease. After expiry, a replacement with the same stable runner ID can
claim ownership and recover expired work. Ordinary registration by a different
session is rejected while the current lease is live; the stock runner uses this
path. Lease renewal performs no agent inference or Decisions API calls; periodic
agent wakes and the future agent heartbeat are deferred.

For a deliberate live handoff, inspect `GET /api/runners`, then register the
replacement with `PUT /api/runners/{runner_id}`: supply its new `worker_id` and
the current session's `worker_id` as `expected_worker_id`. While the lease is live,
a mismatched expected session is rejected. After expiry, replacement does not
require the expected-session field. Registration retains the agent assignments
and rejects capacity below their count.

Changing the owner atomically expires its assigned running conversation leases
while preserving their transcripts. Storage rejects the old session's subsequent
worker-owned run updates and new message delivery. This does not stop the old process,
cancel actions already in progress or undo their external effects. Recovery is
not seamless continuation: inspect unconfirmed computer actions before retrying.

A different runner does not automatically inherit a stopped runner’s agents.
Inspect `GET /api/runners`, then explicitly move an agent with
`POST /api/agents/{key}/runner`, supplying both `runner_id` and
`expected_runner_id` (the current owner, or `null` for an unassigned agent).
`runner_id: null` releases the assignment; an available runner can subsequently
claim it. Reassignment compares the expected owner atomically, checks destination
capacity and fences earlier ownership. A capacity reduction below the number of
existing assignments is rejected; explicitly move agents before reducing it.
Assignments are durable across process restarts; they are not queue leases.

Keep IDs unique and stable. The root `compose.yaml` provides a second explicit service behind the
`runners` profile with ID `runner-2`; do not scale identical copies of `runner-1`.
When mounting a custom instruction file, mount it in every runner: snapshots are
read at run start by the executing process, and resumed runs retain their snapshot.
Tracing uses the existing durable W3C parent across the API/runner boundary, stays
disabled without an OTLP endpoint and excludes prompt/tool payloads.
