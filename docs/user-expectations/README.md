# User expectations

These pages are the canonical product expectations for Kadan and its supporting services. Each feature describes observable behavior, limits and recovery. They were reconciled against master `ab8681a9` on 28 October 2026; see the [reconciliation record](reconciliation.md) for coverage and implementation evidence.

| Area | Start here |
| --- | --- |
| Kadan | [Agents, conversations, memory, computer and Workspace](kadan/overview.md) |
| Hikari | [Models, inference, media and voice](hikari/overview.md) |
| Tsuchi | [Storage, editing, files, history and recovery](tsuchi/overview.md) |
| Shared | [Installation](install-and-use.md), [security and operator responsibility](security.md), [performance tracing](tracing.md) |

Current feature pages describe behavior present on master. Model-directed actions remain subject to the model choosing and using its tools; available tools do not guarantee task completion. [Open and stacked proposals](proposals.md) preserve future direction separately and do not promise current functionality.

Change the relevant expectations with any behavior change. Keep implementation details in [architecture](../architecture), retain significant guarantees and limitations, and distinguish intended experience from implemented behavior. Historical source material remains history rather than a second canonical copy.
