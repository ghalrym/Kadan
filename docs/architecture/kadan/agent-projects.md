# Agent-owned project plans

Projects reuse Tsuchi Space resources, revisions and parent validation. A plan
has the exact `kadan-project` tag and one `project-status:<status>` tag. The runner
validates the status against `ProjectStatus`; missing, duplicate or unknown status
tags require repair rather than silently treating the plan as active. The goal,
step states, progress, pending questions, decisions and document references remain
ordinary editable Markdown. There is no new resource kind, database or migration.
Only active same-agent Spaces can parent children, as enforced by Tsuchi.

`AgentWorkspace` owns the runner's agent-scoped Space reads and writes. Tools
cannot select another namespace. `create_project` saves an active plan;
`update_project` requires an explicit status and read version. Ordinary Space
updates use `update_workspace_document`, which cannot overwrite a marked project.
Updates preserve title, tags outside the status namespace, links, attachments and
other content metadata. Decks are readable, not writable through these tools;
memories, personas and conversations are excluded.

Context preparation searches Spaces for the project marker and exact-filters
tags, including at most eight recent search results in the saved event context.
The index includes IDs, titles, versions and statuses, not plan bodies. Its
`next_offset` counts the underlying search page before filtering. The list tool
can retrieve further pages, including an empty filtered page with a continuation.
Index failures are recorded in the context so the model must check before
creating another plan. Resumed runs retain their context snapshot; document
reads and version checks still see current storage. The ordinary context budget
applies to the index and tools; insufficient context fails visibly.

The existing run lease is checked before tools and again after an update's read,
immediately before writing. Tsuchi's resource version check arbitrates concurrent
user edits. Resource mutations are not transactional with run cancellation and
have no run receipt or idempotency key: Stop cannot undo an in-flight save.
Unconfirmed writes are never replayed after restart; errors describe uncertainty
and require inspection. Read-only calls can be repeated. There is no automatic
retry of a conflicting or uncertain write.

Project state and execution state are deliberately separate. The existing
24-command response limit, cancellation, instruction snapshots and event-driven
run scheduling remain authoritative. No project update schedules a run.
Instructions require progress reporting, checkpointing and yielding for a user
decision. These semantic choices and status transitions are model-directed, not
enforced approval gates. No external coding harness, recurring execution or
multi-agent collaboration is added.

This draft stacks on #396 for document creation. #395 memory tools and #397
typed browsing routes remain independent; runner registry edits will need to be
combined when #395 lands. #398's document navigation and #391 retries are optional
UI improvements. The storage foundation remains existing space/deck documents
and separate memory records. See [project expectations](../../user-expectations/kadan/agent-projects.md)
and [conversation lifecycle](agent-conversations.md).
