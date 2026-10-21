# Overview and activity

Overview summarizes the selected namespace. Resource and file counts exclude trash unless explicitly labelled otherwise. File storage includes retained file bytes in trash; Database · all namespaces reports the shared database size rather than only the selected agent's records. Trashing content is not a way to reclaim its stored bytes.

The Files total includes images, with the image count shown separately. Trash counts both resources and files. Collection cards open their corresponding lists. Export namespace downloads the selected namespace's portable ZIP; it is distinct from a complete stopped-service backup. See [Export and backup](../../../services/tsuchi/README.md#export-and-backup) for scope and restore limitations.

Overview shows the latest eight activity entries. Activity shows the latest 200, newest first. These are storage-change events, not an agent's reasoning transcript, inference queue or complete performance trace. The page does not offer pagination into older events, so absence from the list is not proof that an older change never occurred.

Overview and Activity refresh every 30 seconds while open. Refresh data requests an update immediately. Counts and activity are not a live stream, and an error must not be read as an empty namespace. Switching namespaces changes the records shown; the shared database-size statistic remains global.
