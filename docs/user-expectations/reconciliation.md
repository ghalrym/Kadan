# Expectations reconciliation

This documentation sweep was checked on 28 October 2026 against public master `ab8681a91eec7ca0bf937d8e63be67e48c0eb0c2`. It restores all 51 feature documents from the earlier public repository snapshot [`8c58afe6`](https://github.com/ghalrym/Kadan/tree/8c58afe6e56d66a21b69fb64fa126929f4bcfe41/docs/user-expectations), preserves the newer finding-chat-text page already on master, and reconciles behavior with current source.

## Source coverage

The historical collection was read in full: 51 original documents, 19 proposal bundles containing 40 variants, one unpublished memory/persona/Workspace reconciliation, and eight navigation pages. Original bodies correspond to the public historical Git documents after normalizing navigation links. Public originals were restored from repository history; private source provenance and unrelated material are excluded. Historical pages, their privacy and contents were left intact.

All original paths remain present. The service overviews link to every feature; the table below records the restored coverage. Later pages add agent memory, renaming, reading position, Activity and agent history access.

| Area | Restored original documents |
| --- | --- |
| kadan (16) | [Kadan user expectations](kadan/overview.md), [Agent actions](kadan/agent-actions.md), [Shared agent instructions](kadan/agent-instructions.md), [Pets](kadan/pets.md), [Viewing and controlling the computer](kadan/computer.md), [Agent conversations](kadan/agent-conversations.md), [Images in agent conversations](kadan/image-generation.md), [Saving chat notes](kadan/saving-chat-notes.md), [Stopping chat work](kadan/stopping-chat-work.md), [Forgetting memories](kadan/forgetting-memories.md), [Personas](kadan/personas.md), [Runner availability](kadan/runner-availability.md), [Service status](kadan/service-status.md), [Voice calls](kadan/voice-calls.md), [Voice input](kadan/voice-input.md), [Workspace browsing](kadan/workspace.md) |
| hikari (15) | [Decisions](hikari/decisions.md), [Hikari user expectations](hikari/overview.md), [API access](hikari/api-access.md), [Chat playground](hikari/chat-playground.md), [Image editing](hikari/image-editing.md), [Image generation](hikari/image-generation.md), [Inference operations](hikari/inference.md), [Language model loading and context](hikari/model-lifecycle.md), [Model settings and downloads](hikari/model-settings.md), [Request monitoring](hikari/request-monitoring.md), [Speech to text and formatting](hikari/transcription.md), [Text to speech](hikari/text-to-speech.md), [Video generation](hikari/video-generation.md), [Live voice calls](hikari/voice-calls.md), [Workers on other computers](hikari/remote-workers.md) |
| tsuchi (17) | [Tsuchi user expectations](tsuchi/overview.md), [Storage overview](tsuchi/storage-overview.md), [Storage activity](tsuchi/activity.md), [Agent namespaces](tsuchi/namespaces.md), [Stored conversations](tsuchi/conversation-records.md), [Slide decks](tsuchi/slide-decks.md), [Export](tsuchi/export.md), [Files](tsuchi/files.md), [Revisions](tsuchi/revisions.md), [Images](tsuchi/images.md), [Memories](tsuchi/memories.md), [Stored personas](tsuchi/personas.md), [Trash](tsuchi/trash.md), [Backup](tsuchi/backup.md), [Restore](tsuchi/restore.md), [Search](tsuchi/search.md), [Spaces](tsuchi/spaces.md) |
| shared (3) | [Performance tracing](tracing.md), [Security and operator responsibility](security.md), [Install and use](install-and-use.md) |

## Material reconciliations

| Current behavior | Public ancestry and checked implementation |
| --- | --- |
| Owned personas apply to later text work and new calls; manual additional selection remains separate. Hidden Kadan editor does not imply absent curation. | [#400](https://github.com/ghalrym/Kadan/pull/400), [#402](https://github.com/ghalrym/Kadan/pull/402); [persona design](../architecture/kadan/personas.md), `runner/tools/persona.py`, `api/routes/calls.py`, `api/services.py`. |
| Agent rename preserves identity; sidebar assignment controls preserve runner capacity and handoff limits. | [#403](https://github.com/ghalrym/Kadan/pull/403), [#405](https://github.com/ghalrym/Kadan/pull/405); `RenameAgent.tsx`, `RunnerAssignment.tsx`, [conversation design](../architecture/kadan/agent-conversations.md), [runners](../architecture/kadan/runners.md). |
| Unsent recovery stays with its conversation, while the agent's ordinary tab draft remains available. | [#401](https://github.com/ghalrym/Kadan/pull/401); `Chat.tsx`, `outbox.ts`. |
| Workspace starts on Documents, separates memory, preserves hierarchy/navigation, refreshes selected records and retries failed reads. | [#391](https://github.com/ghalrym/Kadan/pull/391), [#398](https://github.com/ghalrym/Kadan/pull/398), [#407](https://github.com/ghalrym/Kadan/pull/407); `Workspace.tsx`, `workspaceDocuments.ts`, [Workspace design](../architecture/kadan/workspace.md). |
| Memory tools and Markdown document creation exist; background recall, persistent projects and document updates are not implied. | [#395](https://github.com/ghalrym/Kadan/pull/395), [#396](https://github.com/ghalrym/Kadan/pull/396), [#397](https://github.com/ghalrym/Kadan/pull/397); `runner/tools/memory.py`, `runner/tools/workspace.py`, `runner/agent_runs.py`. |
| Users can find saved text and keep their reading position as replies arrive. | [#419](https://github.com/ghalrym/Kadan/pull/419), [#428](https://github.com/ghalrym/Kadan/pull/428); `Chat.tsx`; existing finding-chat-text page preserved. |
| Activity distinguishes completion from delivery; polling summaries do not expose internal transcripts. | [#458](https://github.com/ghalrym/Kadan/pull/458), [#463](https://github.com/ghalrym/Kadan/pull/463); `Activity.tsx`, `activity.ts`, `api/pydantic_models/conversation_work.py`. |
| Agents can search/read older delivered chat with explicit coverage and version limits. | [#460](https://github.com/ghalrym/Kadan/pull/460); `runner/tools/history.py`, `runner/conversation_history.py`, [conversation design](../architecture/kadan/agent-conversations.md). |
| Hikari can copy observed request details, without request replay or full prompt/output capture. | [#447](https://github.com/ghalrym/Kadan/pull/447); `services/hikari/frontend/src/components/RequestDetails.tsx`. |

Kadan frontend filenames above are under `services/kadan/web/src/`; runner and API filenames are under `services/kadan/kadan/`. The [proposal inventory](proposals.md) records all 19 original PR statuses and the newer open work. #402 and #405 reached master through other merged branches; #406, #408 and #409 did not. Their merged labels alone were not used as evidence.

| Stacked PR | Merge commit | Target branch | Merge commit is an ancestor of checked master |
| --- | --- | --- | --- |
| [#402](https://github.com/ghalrym/Kadan/pull/402) | `ebb467e6` | `codex/agent-persona-curation` | Yes |
| [#403](https://github.com/ghalrym/Kadan/pull/403) | `18c34ac8` | `codex/conversation-recovery-scope` | Yes |
| [#405](https://github.com/ghalrym/Kadan/pull/405) | `2d1ebc40` | `codex/agent-display-name` | Yes |
| [#407](https://github.com/ghalrym/Kadan/pull/407) | `59d7070a` | `ai-reed/workspace-document-navigation` | Yes |
| [#406](https://github.com/ghalrym/Kadan/pull/406) | `d77c4aba` | `ai-reed/agent-owned-projects` | No |
| [#408](https://github.com/ghalrym/Kadan/pull/408) | `046ffd21` | `codex/workspace-file-delivery` | No |
| [#409](https://github.com/ghalrym/Kadan/pull/409) | `89444bb2` | `codex/workspace-file-input` | No |

## Remaining limits

This is a source and ancestry reconciliation, not a deployment or live-model verification. Inference quality, throughput, memory fit and end-user recovery flows still require the owner's manual validation on the target machine. Current feature pages retain those limits rather than claiming new guarantees.

Open proposals remain unresolved product/development work. The old memory/persona reconciliation cannot be accepted verbatim because later master implements several of its formerly missing capabilities. This sweep does not merge or implement those proposals.

Only Markdown is changed. Local checks verified all 51 original source bodies against public history, all original paths, 298 local Markdown targets, privacy exclusions, and unchanged unrelated agent instructions. The existing finding-chat-text page is byte-for-byte unchanged. Whitespace checks pass. None of the changed paths match any of the six existing CI workflow path filters. Application unit tests, inference and automated integration tests were not run for this documentation-only change.
