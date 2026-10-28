# Storage overview

Tsuchi's **Overview** summarizes the selected agent's storage. Resource and file counts exclude trash unless labelled otherwise. File storage includes retained bytes in trash, while **Database · all namespaces** reports the shared database size. Trashing content does not reclaim its bytes.

The Files total includes images; Trash counts both resources and files. Collection cards open the corresponding lists. **Export namespace** downloads that agent's [portable archive](export.md), which differs from a complete [backup](backup.md).

Overview also shows a small selection of recent [storage activity](activity.md). It is not an agent's reasoning or an inference queue. Counts and recent changes refresh periodically; **Refresh data** requests an immediate update. Loading failures must remain distinguishable from zero usage.

Switching agents changes the namespace totals and recent activity, while the shared database-size statistic remains global. See [Storage monitoring](../../architecture/tsuchi/monitoring.md) for measurement details.
