# Space

Space is the agent's library of things it and the user make: pages, slide decks, images and files, organized into folders. Kadan owns the library's presentation and request rules; Tsuchi owns every record, version, link rule and the trash. Kadan keeps no second store and no index, so Space always reflects Tsuchi's current records.

## Item mapping

| Space item | Tsuchi record | `parent_id` in Space |
| --- | --- | --- |
| Folder | `folder` resource | the resource's `parent_id` |
| Page | `space` resource (Markdown) | the resource's `parent_id` |
| Slides | `deck` resource | the resource's `parent_id` |
| Image | asset with `is_image` | the asset's `folder_id` |
| File | any other asset | the asset's `folder_id` |

Memories, personas and conversations are not items. Assets tagged `pet` are excluded: Kadan's pet client tags every pet artwork upload with `pet` and the pet ID, and that artwork belongs to the pet, not the library. Generated images (tagged `generated-image`), chat attachments and files the agent saves are included.

Resources and assets live in separate Tsuchi tables, but both use random UUIDs, so one ID names one item. Item routes look the ID up as a resource first and then as an asset. A missing ID, a resource of a non-item kind, pet artwork, or a trashed item for anything other than restore returns 404 with "This item no longer exists."

## Why these storage fields

**Folder kind.** Folders could have been `space` documents with no text, but saved chat replies and the agent's document tool already create `space` records, and the References picker lists them as documents. A separate `folder` kind keeps folders out of document lists and lets Tsuchi enforce what may go in them. A folder has the Markdown shape with empty text and no attachments. Tsuchi's link rules, checked on create, update and restore:

- A folder's parent, if any, must be an active folder in the same namespace.
- A page or deck may sit in an active folder or, as before, an active `space` page.
- Memories, personas and conversations may still sit in a `space` page but never in a folder.
- Every ancestor is checked by the rules for its own kind, and cycles are rejected.

**Asset `folder_id`.** Assets have no `parent_id`, and making them resources would have meant migrating bytes and links. An optional `folder_id` on the asset's JSON record is enough: Tsuchi checks that it names an active folder in the same namespace when it is set at upload or in a metadata update.

**`pinned`.** Pinning is shared metadata on both resources and assets, so it survives in exports and is visible to Tsuchi. Records saved before these fields existed load as unpinned and outside any folder; the fields live in Tsuchi's JSON `data` column and need no migration. Older editors (Tsuchi's own forms, persona and memory corrections) do not send them, so a Tsuchi update that omits `pinned` or `folder_id` keeps the saved value, and restoring an older revision keeps the current pin.

## Trash semantics

Tsuchi refuses to trash a resource with active children, and for a folder also counts active assets whose `folder_id` names it. Deleting a folder or page in Space therefore trashes from the bottom up: Kadan lists active items, walks the subtree and, for each node, trashes its child resources (recursively), then its files, then the node. Consequences:

- A file still attached to an active chat or page cannot be trashed. Deleting it directly reports "This file is attached to a chat or page, so it can't be moved to trash." During a folder delete such a file is moved to the top level instead, so the delete completes and the file stays usable.
- A memory, persona or conversation stored under a deleted page is moved to the top level rather than trashed, since it is not a Space item.
- If anything changes during the walk, the whole delete starts again from a fresh listing, up to five times, before reporting a conflict. Items already trashed stay trashed; nothing is half-deleted silently.

Restore returns one item. A resource restore requires its parent to be active: Kadan checks first and maps Tsuchi's own refusal (for a parent trashed in the meantime) to 409 "Restore the folder it was in first." or "Restore the page it was in first." A page whose attached files are trashed returns 409 "Restore the files this item uses first." An asset whose folder is missing or trashed is restored to the top level: Tsuchi clears its `folder_id` instead of refusing, because a file has no content that depends on its folder. Restoring a folder does not restore what was trashed with it. Restoring an active item returns it unchanged, so retries are harmless.

## Edits and concurrency

Item edits (`title`, `parent_id`, `pinned`) are read-modify-write. For resources Kadan edits Tsuchi's own JSON, as [chats do](agent-conversations.md#multiple-chats), so fields it does not model are kept; for assets it sends the current title, tags and sources plus only the changed pin and folder fields. Both send the version they read and retry up to five times on a version conflict, because the agent may be editing the same record. Page saves are different: the browser sends the version it opened, and Tsuchi's conflict message is returned with 409 so the user's draft is never written over newer text.

Kadan checks a new parent before writing so it can explain the problem: a missing or trashed parent returns 404 "The folder you chose no longer exists.", and a folder or file aimed at a page or deck returns 422 "Folders and files can only go inside folders." Tsuchi still enforces the rules in its write transaction; its cycle rejection is returned as 422 "An item can't go inside itself or something it contains."

## API

All routes are scoped to `/api/agents/{key}/space`, reject unknown fields and return errors as `{detail, code}`.

| Route | Result |
| --- | --- |
| `GET /items` | `SpaceItem[]`: every active item in every folder, newest `updated_at` first, then by ID |
| `GET /trash` | `SpaceItem[]`: every trashed item, same order |
| `POST /folders` | 201 `SpaceItem`; body `{title, parent_id?}` |
| `POST /pages` | 201 `SpaceItem`; body `{title, markdown? = "", parent_id?}` |
| `GET /pages/{id}` | `SpacePage` |
| `PUT /pages/{id}` | `SpacePage`; body `{title, markdown, expected_version}`; 409 with Tsuchi's message on conflict |
| `GET /slides/{id}` | `SpaceSlides` |
| `PUT /items/{id}` | `SpaceItem`; body `{title?, parent_id?, pinned?}` |
| `DELETE /items/{id}` | 204; moves the item, and everything inside a folder or page, to trash |
| `POST /items/{id}/restore` | `SpaceItem` |

`SpaceItem` has `id`, `type` (`folder`, `page`, `slides`, `image` or `file`), `title`, `parent_id`, `pinned`, `filename`, `media_type` and `size_bytes` (null for resources), `version`, `created_at`, `updated_at` and `deleted_at`. `SpacePage` adds `markdown`; `SpaceSlides` adds `slides`, Tsuchi's deck slides with `id`, `layout`, `title`, `body`, `notes` and `image_asset_id`.

Titles are nonblank and stripped. In `PUT /items/{id}` only fields present in the body change: an omitted `parent_id` keeps the item where it is and an explicit `null` moves it to the top level; a `null` title or pin is rejected, and an empty body returns the item without writing. Page Markdown is limited to Tsuchi's 2,000,000 characters.

Uploads use the existing `POST /api/agents/{key}/assets` multipart route with an optional `folder_id` form field; Tsuchi validates it in the same write as the asset record. A folder that no longer exists returns 404 "The folder you chose no longer exists." File bytes are read and downloaded through the existing asset content routes.

Lists page through Tsuchi 200 records at a time for resources (kinds `folder`, `space`, `deck`) and assets, merging pages by ID and keeping the newest version when a concurrent write moves a record between pages. The lists are unfiltered; type filters, search, pinned and recents are applied in the browser over the one list.

## Ownership

`api/space_client.py` owns the mapping, lookups, recursive trash and restore rules; `api/routes/agents/space/` holds the route and request models; `api/pydantic_models/space.py` holds the response models. Tsuchi's repository owns the folder link rules, the asset folder check, the folder-contents trash check and the restore-to-top rule for assets.

See the user expectations for [Space](../../user-expectations/kadan/space.md).
