# Storage activity

Tsuchi's **Activity** shows recent saved changes for the selected agent, newest first. It includes storage operations such as creating, updating, trashing and restoring records. It does not show the agent's reasoning or every action taken on its computer.

The view is bounded and has no pagination into older events. Older changes can disappear from the displayed list, so it is not a complete audit history. An empty activity list does not prove that the agent has never performed work. [Storage overview](storage-overview.md) shows a smaller selection from the same activity.

Activity refreshes periodically while open. **Refresh data** requests another update, and switching namespaces shows the other agent's changes. Loading errors must remain visible rather than becoming an empty successful result.

Opening or refreshing Activity does not run inference, replay a change or cancel work. See [Storage monitoring](../../architecture/tsuchi/monitoring.md) for the view's scope and limits.
