# Storage monitoring

Tsuchi derives namespace statistics and recent activity from its stored records. These summaries help inspect storage without treating a bounded view as a complete event ledger. See the user expectations for [storage overview](../../user-expectations/tsuchi/storage-overview.md) and [activity](../../user-expectations/tsuchi/activity.md).

The namespace stats endpoint separates active resource and asset counts, image counts, trashed resource and asset counts, retained asset bytes, and database bytes. Images are a subset of assets. Retained bytes include trashed files; database size describes the shared database across namespaces. This distinction prevents a namespace count or trash action from implying reclaimed disk capacity.

The UI requests eight activity entries for Overview and 200 for Activity, newest first, without pagination into older events. Both pages refresh every 30 seconds while open and support manual refresh. These are storage-change observations, independent of agent execution and inference telemetry.

Namespace export is a portable snapshot, not a full-volume backup. The existing [export and backup instructions](backup.md) remain the home for backup scope and restore procedures.
