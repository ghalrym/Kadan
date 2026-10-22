# Export and backup

**Export** in the website downloads a ZIP for the selected namespace. It includes a JSON manifest with resources, revisions, metadata, trash, activity, and original files under `files/{asset-id}`. This is a portable export; there is no ZIP import endpoint in v1.

For a complete backup of all namespaces, stop the service and archive the whole volume. Stop it first so the SQLite database and files cannot change during the copy. The commands below apply to Tsuchi managed by this repository's root Compose project. If you still use the standalone deployment, stop and back up through that original Compose project first; root Compose cannot stop a container belonging to another project. Run these commands from the repository root, writing the backup somewhere with enough free disk space:

```sh
docker compose stop tsuchi-api tsuchi-frontend
docker compose run --rm -T --no-deps --entrypoint tar tsuchi-api \
  -czf - -C /data . > tsuchi-backup.tar.gz
docker compose up -d
```

Check that the archive is readable before relying on it:

```sh
tar -tzf tsuchi-backup.tar.gz
```

Restore your own trusted backup into a **new empty volume**, preserving the original. The following uses a separate Compose project and refuses to extract into a nonempty data directory:

```sh
TSUCHI_DATA_VOLUME=tsuchi-restored-data docker compose -p tsuchi-restored run --rm -T --no-deps --entrypoint sh tsuchi-api \
  -c 'test -z "$(ls -A /data)" && tar -xzf - -C /data --no-same-owner' \
  < tsuchi-backup.tar.gz
TSUCHI_DATA_VOLUME=tsuchi-restored-data TSUCHI_PORT=8621 docker compose -p tsuchi-restored up -d --no-deps tsuchi-api tsuchi-frontend
```

Inspect the restored instance at <http://127.0.0.1:8621> before switching anything over. Coordinate this additional service and port with the server owner. The original volume is untouched.
