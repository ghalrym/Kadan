# Folders, asset folders and pins

Tsuchi stores the records behind Kadan's [Space](../kadan/space.md) library. Three additions support it without a schema migration, because resources and assets keep their fields in a JSON `data` column:

- A `folder` resource kind. It has the Markdown shape but must have empty text and no attachments. A folder's parent, if any, is an active folder in the same namespace. `space` and `deck` resources may sit in an active folder or an active `space`; memories, personas and conversations may sit only in a `space`. Every ancestor is validated by the rules for its own kind, so a page in a page in a folder is valid while a memory in a folder is not, and cycles are rejected.
- An optional asset `folder_id`, accepted at upload (multipart `folder_id`) and in `PUT /assets/{id}`. It must name an active folder in the namespace. Restoring an asset whose folder is missing or trashed clears `folder_id` instead of failing, because a file's contents do not depend on its folder.
- A `pinned` flag on resources and assets, defaulting to false for older records.

Trashing a folder is refused (409) while it has active child resources or active assets whose `folder_id` names it; the asset check reads `json_extract(data, '$.folder_id')`. Kadan trashes a folder's contents first.

Clients written before these fields existed do not send them. A resource update that omits `pinned`, or an asset update that omits `pinned` or `folder_id`, keeps the saved value instead of resetting it, and restoring an older revision keeps the current pin. Sending a field explicitly, including `null` for `folder_id`, changes it.

See the expectations for [folders](../../user-expectations/tsuchi/folders.md).
