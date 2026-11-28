# Agent namespaces

Tsuchi scopes resources, revisions, search entries, assets, runs, image jobs, pet jobs, runner assignments and activity records to an agent namespace identified by `agents.key`. Global configuration tables (`mcp_servers`, `mcp_oauth_states` and `runners`) are shared across the installation and are not owned by any single agent.

## Permanent deletion

`DELETE /api/agents/{key}` returns `204 No Content` (or `404` when the namespace does not exist) and permanently removes the agent and every record owned by `key` inside a single SQLite write transaction:

1. Verify the agent exists and record all `assets.id` values scoped to `key`.
2. Clear `agents.persona_id` and `agents.selected_pet_id` before removing referenced rows so circular references cannot block deletion.
3. Delete dependent rows in foreign-key-safe order: `image_jobs`, `agent_runs`, `pet_jobs`, `t3_thread_watches` matching `json_extract(data, '$.origin_agent_key') = key`, `runner_assignments`, `resource_search`, `revisions`, `resources`, `assets`, `activity`, and finally `agents`.
4. After the transaction commits, unlink each collected asset blob file from `files/{asset_id}` (ignoring missing files safely) and clear any in-memory run drafts for `key`.

Because blob removal runs only after SQLite commits, a failed or rolled-back transaction leaves files on disk intact. Global `mcp_servers`, `mcp_oauth_states`, `runners` rows and all other agents' records and files remain untouched. Deleting an agent frees its key immediately for fresh creation (including Kadan's default-agent auto-bootstrap).

Kadan shows one agent at a time; Kadan Server's [Your agent](../../user-expectations/kadan-server/your-agent.md) covers what users see of its stored content.
