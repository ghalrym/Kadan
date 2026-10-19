# Compose releases

A release is a complete immutable service-image map bound to one reviewed,
CI-passing master commit and the canonical Compose topology. Publication owns
builds and registry pushes; a server only pulls digest pins. The fixed GitHub
publication workflow is the provenance boundary. The host retains its existing
Compose project, environment and persistent volumes. No release can introduce
arbitrary service commands or rewrite host deployment topology.

One host controller owns release order, a last-known-good record and a durable
replacement journal under a process lock. A later release must descend from the
installed source. Compatibility-epoch changes, data migration changes and topology
changes need an explicit manual transition. Recovery restores prior image pins,
never prior database contents, so accepted automatic updates must remain compatible
with both the installed and previous data contracts.

Admission is local to each single-process owner. A host-owned persistent directory
fence is read at admission and observed through container-local Unix status sockets.
No administrative HTTP endpoint is exposed. Draining follows dependency order:
Kadan request/job owners, Hikari inference, then Tsuchi storage. Downstream services
remain available until accepted upstream work finishes. Full response streaming,
background operations and accepted jobs belong to the drain, rather than only the
initial HTTP handler. Readiness and non-mutating smoke checks precede reopening.
Unknown owner state preserves the fence instead of permitting destructive recovery.
Operational logs identify release source, affected service names and recovery phase;
request tracing remains with the existing request spans and does not acquire
parallel deployment performance counters.

The API and its seven native workers are one connection-lifecycle group. A change
to any member replaces the group with previous pins retained for unchanged images.
A strict idle barrier freezes maintenance and control admission, requires every
inference and store lane to have no busy, queued or releasing work, and binds the
proof to stable worker generations and immutable container identities. The proof
precedes every stop. No unfinished-job state is snapshotted or adopted.

A controller cold-start marker is valid only after the recorded old cohort is
conclusively stopped. A fresh API and fresh native process generations must match
that marker and prove idle before readiness can reopen admission. Recovery can
stop only proven old epochs or transaction-bound new epochs which have never
admitted work. Unknown disconnects, hidden processes, restarted epochs and daemon
errors preserve the fence and journal rather than turning uncertainty into a stop.
Native protocol version 5 and manual initial enrollment are prerequisites.

The host-local dispatcher polls only the fixed successful publication workflow,
shares the rollout lock and recovers pending transactions before network discovery.
Commit ancestry decides ordering independently of workflow run numbers. Delivery
receipts prevent duplicate work without hiding late completions. Discovery errors
retry with bounded backoff; deployment failures stop polling and surface a logged
reason/nonzero exit for operator attention. The source is inactive until explicitly
started; host access, activation and notification integration are outside this
architecture change.
