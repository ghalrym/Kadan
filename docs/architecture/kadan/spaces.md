# Chats and Spaces

Each agent can have many chats, and chats can be grouped into Spaces. A Space has a name, instructions and its own chats; it holds no files or memories. Kadan owns the presentation and request rules, and Tsuchi stores the records. Runs are already scoped to one conversation (`run.conversation_id`), so multiple chats need no change to run ownership, queueing or delivery.

## Storage

| Kadan concept | Tsuchi record |
| --- | --- |
| Chat | `conversation` resource |
| Space | `project` resource; `title` is the Space name and `markdown` holds its instructions |
| Chat in a Space | `conversation` whose `parent_id` is the project |

Tsuchi's existing `space` kind is a Markdown document with a parent and child hierarchy. Saved chat results and the agent's create-document tool create `space` documents, and the composer's References picker lists them. Reusing that kind for Kadan Spaces would make every saved document appear as a Space and let documents nest inside Spaces. A separate `project` kind keeps the two collections apart without migrating existing documents. Tsuchi's own interface labels projects "Kadan Spaces".

Tsuchi enforces the shape when a resource is created, updated or restored:

- A project has no parent.
- Only a conversation can have a project as its parent, and the project must be active and in the same agent namespace. A project ends the ancestor chain, so projects never appear inside `space` hierarchies.
- The existing `space` rules are unchanged. Conversations may still be parented to a `space` document, but Kadan treats those chats as outside any Space.

Tsuchi refuses to trash a resource with active children. Deleting a Space therefore trashes each of its active chats first, then the project. If a chat is added or the project changes during deletion, the trash request conflicts and Kadan repeats the whole sequence a few times before reporting a conflict. Trashed chats and Spaces stay recoverable in Tsuchi's trash; restoring a chat requires its Space to be active.

Moving or renaming a chat rewrites the whole conversation resource through Tsuchi's versioned update. Runs append messages to the same resource concurrently, so Kadan edits Tsuchi's own JSON (keeping fields it does not model), sends the version it read and, on a version conflict, re-reads and retries up to five times. A conflict can never overwrite a newer message list. Trashing a chat uses the same retry. Trashing a chat while a reply is running leaves that run unable to save further messages; Tsuchi stops offering runs for trashed conversations.

## Display titles

New chats are stored with the title `New chat`. The single conversation that Kadan created per agent before multiple chats was titled `Chain`. When the stored title, stripped, is one of these defaults and the chat has a user message, the summary title is that first nonblank user message with whitespace collapsed, cut to 60 characters including a trailing `…`. Otherwise it is the stored title. The rule is applied when summaries are built, so the stored title is never rewritten and a renamed chat keeps its name. The preview uses the same first user message, cut to 160 characters, or is empty.

## API

All routes are scoped to `/api/agents/{key}` and return plain-language errors as `{detail, code}`.

| Route | Result |
| --- | --- |
| `GET /conversations` | `ConversationSummary[]`: every active chat, in or out of a Space, newest `updated_at` first, then by ID |
| `POST /conversations` | 201 with the full conversation `Resource` |
| `PUT /conversations/{conversation_id}` | `ConversationSummary` |
| `DELETE /conversations/{conversation_id}` | 204; the chat moves to Tsuchi's trash |
| `GET /spaces` | `Space[]`, newest `updated_at` first |
| `POST /spaces` | 201 with the `Space` |
| `GET /spaces/{space_id}` | `Space` |
| `PUT /spaces/{space_id}` | `Space` |
| `DELETE /spaces/{space_id}` | 204; trashes the Space's chats, then the Space |

`ConversationSummary` has `id`, `agent_key`, `title` (the display title above), `space_id`, `message_count`, `preview`, `version`, `created_at` and `updated_at`. `space_id` is the parent ID only when the parent is a project; otherwise it is null. Summaries omit message bodies; the chat view loads the open conversation as a resource and then follows it through the conversation runs snapshot.

`Space` has `id`, `agent_key`, `title`, `instructions`, `version`, `created_at` and `updated_at`.

Requests reject unknown fields:

- `POST /conversations` takes optional `title` (nonblank, stripped, default `New chat`) and optional `space_id`.
- `PUT /conversations/{id}` takes optional `title` and `space_id`. Only fields present in the body change: an omitted `space_id` keeps the chat where it is, and an explicit `null` moves it out of its Space. A `null` title is rejected. An empty body returns the current summary without writing.
- `POST /spaces` takes `title` (nonblank, stripped) and optional `instructions` (at most 48,000 characters, default empty).
- `PUT /spaces/{id}` takes `title`, `instructions` and `expected_version`. Tsuchi checks the version, and its conflict message is returned with 409.

A `space_id` or `{space_id}` that is missing, trashed, from another agent or not a project returns 404, and no chat is created or moved. A conversation ID that is missing, trashed or not a conversation returns 404. Lists page through Tsuchi 200 records at a time and merge pages by ID, keeping the newest version when a concurrent write moves a record between pages.

## Space instructions in runs

When the runner prepares a new run's context, it reads the conversation's parent. If the parent is an active project of the same agent, the system message gains a line naming the Space and, when its instructions are not blank, the instructions themselves after the shared instructions and personas. A missing, trashed or non-project parent adds nothing and does not fail the run; other storage failures fail context preparation as any other read would. As with personas, the instructions are captured in the run's saved context, so editing a Space affects later runs, not one already started.

## Frontend

`App.tsx` loads the agent's summaries, Spaces and personas together, then loads only the open conversation. The selection is remembered with the agent key it belongs to, so a refresh that switches agents never requests the old chat under the new agent. When the agent has no chats, the frontend creates one. Chat snapshots update the matching summary locally (`conversations.ts` mirrors the display-title rule), so the sidebar follows new messages without listing again.

Empty chats are hidden from lists unless open, and **New chat** reuses an empty chat in the same place before creating another, so abandoned new chats do not pile up. A Space page creates its chat first and hands the typed text to the chat view, which sends it once through the normal send path when the chat can reply. That keeps the outbox, retry and recovery rules identical to messages typed in the chat itself.

See the user expectations for [agent conversations](../../user-expectations/kadan/agent-conversations.md), [Spaces](../../user-expectations/kadan/spaces.md) and [navigation](../../user-expectations/kadan/navigation.md).
