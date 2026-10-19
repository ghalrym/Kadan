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

The current native protocol cannot prove or adopt pre-existing work after a
Hikari API restart. Automatic Hikari API/native replacements are therefore blocked;
a native queue-idle/snapshot contract is a prerequisite. Enrollment also requires a
manually verified native idle baseline. The implemented draft can replace compatible
Kadan, Tsuchi and frontend images while retaining all inference processes and data.
