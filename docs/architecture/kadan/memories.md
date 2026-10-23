# Memory tools

The runner supplies the agent namespace to `AgentMemories`; tool arguments cannot change it. List and ordinary read operations expose active memory resources only. `read_memory` accepts `include_trashed` for checking uncertain writes; a trashed result includes identity, title, version and deletion time without Markdown or tags. Writes reject other resource kinds, other namespaces and mismatched resource IDs. See [Forgetting memories](../../user-expectations/kadan/forgetting-memories.md) for user expectations.

`forget_memory` reads the active record and checks the supplied version before calling Tsuchi's existing resource DELETE with `expected_version` in the query. Tsuchi checks the version again in its write transaction and preserves the record and revision history. A successful tool result requires a matching memory with a deletion time and the next version. No storage schema or API changes are required.

Every memory mutation checks run ownership immediately before its storage request, after any preflight read. Cancellation or lease loss after the request starts cannot undo a committed change. Recovery repeats reads but returns uncertainty instead of replaying writes; the agent must inspect the current record before deciding what to do next. An unknown outcome is not a confirmation of success or failure.
