# Export and backup

**Export** in the website downloads a ZIP for the selected namespace. It includes a JSON manifest with resources, revisions, metadata, trash, activity, and original files under `files/{asset-id}`. This is a portable export; there is no ZIP import endpoint in v1.

For a complete backup of all namespaces, stop Kadan Server from Settings (so the
SQLite database and files cannot change during the copy) and archive the `tsuchi`
folder inside your Kadan data directory (`<data_dir>/tsuchi`):

```sh
tar -czf tsuchi-backup.tar.gz -C "$KADAN_DATA_DIR/tsuchi" .
```

Check that the archive is readable before relying on it:

```sh
tar -tzf tsuchi-backup.tar.gz
```

Restore your own trusted backup into a **new empty folder**, preserving the original:

```sh
mkdir -p "$RESTORE_DIR" && test -z "$(ls -A "$RESTORE_DIR")" && \
  tar -xzf tsuchi-backup.tar.gz -C "$RESTORE_DIR"
```

Point Kadan Server's data folder at a copy of the restored store (or inspect it separately) before switching over. The original folder stays untouched.
