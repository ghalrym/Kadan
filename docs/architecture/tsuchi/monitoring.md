# Storage monitoring

Tsuchi derives namespace statistics and recent activity from its stored records. These summaries help inspect storage without treating a bounded view as a complete event ledger. Kadan Server's [Home](../../user-expectations/kadan-server/home.md) agent card reads the stats endpoint; nothing in the UI reads the activity endpoint anymore.

The namespace stats endpoint separates active resource and asset counts, image counts, trashed resource and asset counts, retained asset bytes, and database bytes. Images are a subset of assets. Retained bytes include trashed files; database size describes the shared database across namespaces. This distinction prevents a namespace count or trash action from implying reclaimed disk capacity.

The activity endpoint returns newest-first storage-change observations without pagination into older events. They are independent of agent execution and inference telemetry.

Namespace export is a portable snapshot, not a full-volume backup. The existing [export and backup instructions](backup.md) remain the home for backup scope and restore procedures.
