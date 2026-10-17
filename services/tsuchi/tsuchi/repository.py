import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

from .config import StorageError
from .models import (
    MAX_CONVERSATION_MESSAGES,
    Activity,
    Agent,
    AgentInput,
    Asset,
    ExportManifest,
    ExportRevision,
    Message,
    Metadata,
    Resource,
    ResourceInput,
    ResourceKind,
    ResourceRecord,
    ResourceVersion,
    Stats,
    resource_adapter,
    resource_input_adapter,
    utc_now,
)


def search_expression(query: str) -> str:
    """Treat punctuation and FTS operators as text, never SQL/FTS syntax."""
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)[:30]
    return " AND ".join(f'"{token}"*' for token in tokens)


def resource_search_text(resource: Resource) -> str:
    parts = [resource.title, resource.markdown, *resource.tags]
    parts.extend(f"{source.label} {source.url}" for source in resource.sources)
    if resource.deck:
        for slide in resource.deck.slides:
            parts.extend([slide.title, slide.body, slide.notes])
    for message in resource.messages:
        parts.append(message.content)
        parts.extend(f"{source.label} {source.url}" for source in message.sources)
    return "\n".join(parts)


def asset_references(resource: ResourceInput) -> set[UUID]:
    refs = set(resource.attachment_ids)
    for message in resource.messages:
        refs.update(message.attachment_ids)
    if resource.deck:
        refs.update(
            slide.image_asset_id
            for slide in resource.deck.slides
            if slide.image_asset_id
        )
    return refs


class Repository:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.database_path = data_dir / "tsuchi.sqlite3"

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as db:
            db.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS agents (
                    key TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY, agent_key TEXT NOT NULL REFERENCES agents(key),
                    kind TEXT NOT NULL, parent_id TEXT, deleted INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS resources_namespace ON resources(agent_key, deleted, kind);
                CREATE INDEX IF NOT EXISTS resources_parent ON resources(agent_key, parent_id, deleted);
                CREATE VIRTUAL TABLE IF NOT EXISTS resource_search USING fts5(
                    id UNINDEXED, agent_key UNINDEXED, text, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    agent_key TEXT NOT NULL REFERENCES agents(key), resource_id TEXT NOT NULL,
                    version INTEGER NOT NULL, created_at TEXT NOT NULL, snapshot TEXT NOT NULL,
                    PRIMARY KEY (agent_key, resource_id, version)
                );
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, agent_key TEXT NOT NULL REFERENCES agents(key),
                    deleted INTEGER NOT NULL, is_image INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assets_namespace ON assets(agent_key, deleted, is_image);
                CREATE TABLE IF NOT EXISTS activity (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_key TEXT NOT NULL REFERENCES agents(key), data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS activity_namespace ON activity(agent_key, sequence);
            """)
            # Serialize the additive migration for existing storage volumes.
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(agents)")}
            if "avatar_asset_id" not in columns:
                db.execute("ALTER TABLE agents ADD COLUMN avatar_asset_id TEXT")
            db.commit()

    @contextmanager
    def connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 30000")
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def health(self) -> None:
        with self.connect() as db:
            db.execute("SELECT count(*) FROM agents").fetchone()

    @staticmethod
    def _agent(db: sqlite3.Connection, key: str) -> Agent:
        row = db.execute("SELECT * FROM agents WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise StorageError("Agent namespace not found", 404)
        return Agent.model_validate(dict(row))

    def get_agent(self, key: str) -> Agent:
        with self.connect() as db:
            return self._agent(db, key)

    def list_agents(self) -> list[Agent]:
        with self.connect() as db:
            return [
                Agent.model_validate(dict(row))
                for row in db.execute("SELECT * FROM agents ORDER BY name, key")
            ]

    def create_agent(self, value: AgentInput) -> Agent:
        agent = Agent(**value.model_dump(), created_at=utc_now())
        with self.connect(write=True) as db:
            if db.execute(
                "SELECT 1 FROM agents WHERE key = ?", (agent.key,)
            ).fetchone():
                raise StorageError("This agent key already exists", 409)
            db.execute(
                "INSERT INTO agents (key, name, created_at) VALUES (?, ?, ?)",
                (agent.key, agent.name, agent.created_at.isoformat()),
            )
            self._activity(db, agent.key, "created", "agent", agent.key, agent.name)
        return agent

    def update_avatar(self, key: str, asset_id: UUID | None) -> Agent:
        with self.connect(write=True) as db:
            agent = self._agent(db, key)
            if asset_id is not None:
                asset = self._asset(db, key, asset_id)
                self._active(asset)
                if not asset.is_image:
                    raise StorageError("Avatars must reference a verified raster image")
            if agent.avatar_asset_id != asset_id:
                db.execute(
                    "UPDATE agents SET avatar_asset_id = ? WHERE key = ?",
                    (str(asset_id) if asset_id else None, key),
                )
                self._activity(db, key, "avatar_updated", "agent", key, agent.name)
            return agent.model_copy(update={"avatar_asset_id": asset_id})

    @staticmethod
    def _activity(
        db: sqlite3.Connection,
        key: str,
        action: str,
        entity_type: str,
        entity_id: str,
        title: str,
    ) -> None:
        activity = Activity(
            id=uuid4(),
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            title=title,
            created_at=utc_now(),
        )
        db.execute(
            "INSERT INTO activity (agent_key, data) VALUES (?, ?)",
            (key, activity.model_dump_json()),
        )

    def activity(self, key: str, limit: int = 50) -> list[Activity]:
        with self.connect() as db:
            self._agent(db, key)
            return [
                Activity.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM activity WHERE agent_key = ? ORDER BY sequence DESC LIMIT ?",
                    (key, limit),
                )
            ]

    @staticmethod
    def _resource(db: sqlite3.Connection, key: str, resource_id: UUID) -> Resource:
        row = db.execute(
            "SELECT data FROM resources WHERE agent_key = ? AND id = ?",
            (key, str(resource_id)),
        ).fetchone()
        if row is None:
            raise StorageError("Resource not found in this namespace", 404)
        return resource_adapter.validate_json(row[0])

    def get_resource(self, key: str, resource_id: UUID) -> Resource:
        with self.connect() as db:
            self._agent(db, key)
            return self._resource(db, key, resource_id)

    def list_resources(
        self,
        key: str,
        kind: ResourceKind | None = None,
        query: str = "",
        trash: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Resource]:
        with self.connect() as db:
            self._agent(db, key)
            clauses, params = ["r.agent_key = ?", "r.deleted = ?"], [key, int(trash)]
            if kind:
                clauses.append("r.kind = ?")
                params.append(kind.value)
            expression = search_expression(query)
            if query.strip() and not expression:
                return []
            if expression:
                clauses.append(
                    "r.id IN (SELECT id FROM resource_search WHERE resource_search MATCH ? AND agent_key = ?)"
                )
                params.extend([expression, key])
            params.extend([limit, offset])
            sql = (
                "SELECT r.data FROM resources r WHERE "
                + " AND ".join(clauses)
                + " ORDER BY r.updated_at DESC, r.id LIMIT ? OFFSET ?"
            )
            return [
                resource_adapter.validate_json(row[0])
                for row in db.execute(sql, params)
            ]

    @staticmethod
    def _check_version(current: Resource | Asset, expected_version: int) -> None:
        if current.version != expected_version:
            raise StorageError("This item changed. Reload it before saving again.", 409)

    @staticmethod
    def _active(current: Resource | Asset) -> None:
        if current.deleted_at:
            raise StorageError("Restore this item from trash before editing it", 409)

    def _validate_links(
        self, db: sqlite3.Connection, key: str, value: ResourceInput, resource_id: UUID
    ) -> None:
        parent_id = value.parent_id
        visited = {resource_id}
        while parent_id:
            if parent_id in visited:
                raise StorageError(
                    "A Space cannot contain itself or one of its ancestors"
                )
            visited.add(parent_id)
            parent = self._resource(db, key, parent_id)
            if parent.kind != ResourceKind.SPACE or parent.deleted_at:
                raise StorageError("Parent must be an active Space in this namespace")
            parent_id = parent.parent_id
        for asset_id in asset_references(value):
            asset = self._asset(db, key, asset_id)
            if asset.deleted_at:
                raise StorageError("Restore attached assets before linking them")
        if value.deck:
            for slide in value.deck.slides:
                if (
                    slide.image_asset_id
                    and not self._asset(db, key, slide.image_asset_id).is_image
                ):
                    raise StorageError(
                        "Slide images must reference a verified raster image"
                    )
        for message in value.messages:
            for target_id in message.resource_ids:
                if target_id != resource_id:
                    self._resource(db, key, target_id)

    @staticmethod
    def _save_resource(db: sqlite3.Connection, resource: Resource) -> None:
        db.execute(
            """INSERT INTO resources VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, parent_id=excluded.parent_id,
            deleted=excluded.deleted, updated_at=excluded.updated_at, data=excluded.data""",
            (
                str(resource.id),
                resource.agent_key,
                resource.kind.value,
                str(resource.parent_id) if resource.parent_id else None,
                int(resource.deleted_at is not None),
                resource.updated_at.isoformat(),
                resource.model_dump_json(),
            ),
        )
        db.execute(
            "DELETE FROM resource_search WHERE id = ? AND agent_key = ?",
            (str(resource.id), resource.agent_key),
        )
        db.execute(
            "INSERT INTO resource_search VALUES (?, ?, ?)",
            (str(resource.id), resource.agent_key, resource_search_text(resource)),
        )
        db.execute(
            "INSERT INTO revisions VALUES (?, ?, ?, ?, ?)",
            (
                resource.agent_key,
                str(resource.id),
                resource.version,
                resource.updated_at.isoformat(),
                resource.model_dump_json(),
            ),
        )

    def create_resource(self, key: str, value: ResourceInput) -> Resource:
        now = utc_now()
        resource = resource_adapter.validate_python(
            {
                **value.model_dump(),
                "id": uuid4(),
                "agent_key": key,
                "version": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
        with self.connect(write=True) as db:
            self._agent(db, key)
            self._validate_links(db, key, value, resource.id)
            self._save_resource(db, resource)
            self._activity(
                db, key, "created", "resource", str(resource.id), resource.title
            )
        return resource

    def update_resource(
        self, key: str, resource_id: UUID, value: ResourceInput, expected_version: int
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            return self._update_resource(
                db, key, current, value, expected_version, "updated"
            )

    def _update_resource(
        self,
        db: sqlite3.Connection,
        key: str,
        current: Resource,
        value: ResourceInput,
        expected_version: int,
        action: str,
    ) -> Resource:
        self._check_version(current, expected_version)
        self._active(current)
        if value.kind != current.kind:
            raise StorageError("Resource kind cannot change")
        self._validate_links(db, key, value, current.id)
        updated = resource_adapter.validate_python(
            {
                **value.model_dump(),
                "id": current.id,
                "agent_key": key,
                "version": current.version + 1,
                "created_at": current.created_at,
                "updated_at": utc_now(),
            }
        )
        self._save_resource(db, updated)
        self._activity(db, key, action, "resource", str(current.id), updated.title)
        return updated

    def trash_resource(
        self, key: str, resource_id: UUID, expected_version: int
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            self._check_version(current, expected_version)
            self._active(current)
            if db.execute(
                "SELECT 1 FROM resources WHERE agent_key = ? AND parent_id = ? AND deleted = 0 LIMIT 1",
                (key, str(resource_id)),
            ).fetchone():
                raise StorageError(
                    "Move or trash this Space's active children first", 409
                )
            now = utc_now()
            updated = current.model_copy(
                update={
                    "deleted_at": now,
                    "updated_at": now,
                    "version": current.version + 1,
                }
            )
            self._save_resource(db, updated)
            self._activity(
                db, key, "trashed", "resource", str(resource_id), current.title
            )
            return updated

    def restore_resource(
        self, key: str, resource_id: UUID, expected_version: int
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            self._check_version(current, expected_version)
            if not current.deleted_at:
                raise StorageError("This resource is not in trash", 409)
            self._validate_links(db, key, current, resource_id)
            updated = current.model_copy(
                update={
                    "deleted_at": None,
                    "updated_at": utc_now(),
                    "version": current.version + 1,
                }
            )
            self._save_resource(db, updated)
            self._activity(
                db, key, "restored", "resource", str(resource_id), current.title
            )
            return updated

    def versions(self, key: str, resource_id: UUID) -> list[ResourceVersion]:
        with self.connect() as db:
            self._resource(db, key, resource_id)
            return [
                ResourceVersion(
                    version=row["version"],
                    created_at=row["created_at"],
                    snapshot=resource_adapter.validate_json(row["snapshot"]),
                )
                for row in db.execute(
                    "SELECT * FROM revisions WHERE agent_key = ? AND resource_id = ? ORDER BY version DESC",
                    (key, str(resource_id)),
                )
            ]

    def restore_version(
        self, key: str, resource_id: UUID, version: int, expected_version: int
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            row = db.execute(
                "SELECT snapshot FROM revisions WHERE agent_key = ? AND resource_id = ? AND version = ?",
                (key, str(resource_id), version),
            ).fetchone()
            if row is None:
                raise StorageError("Version not found", 404)
            snapshot = resource_adapter.validate_json(row[0])
            value = resource_input_adapter.validate_python(
                snapshot.model_dump(exclude=set(ResourceRecord.model_fields))
            )
            return self._update_resource(
                db, key, current, value, expected_version, "version_restored"
            )

    def append_message(
        self, key: str, resource_id: UUID, message: Message, expected_version: int
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            if current.kind != ResourceKind.CONVERSATION:
                raise StorageError("Messages can only be added to conversations")
            if any(existing.id == message.id for existing in current.messages):
                raise StorageError("This message ID already exists", 409)
            if len(current.messages) >= MAX_CONVERSATION_MESSAGES:
                raise StorageError(
                    f"Conversation already contains the maximum of {MAX_CONVERSATION_MESSAGES} messages",
                    409,
                )
            value = resource_input_adapter.validate_python(
                {
                    **current.model_dump(exclude=set(ResourceRecord.model_fields)),
                    "messages": [*current.messages, message],
                }
            )
            return self._update_resource(
                db, key, current, value, expected_version, "message_added"
            )

    @staticmethod
    def _asset(db: sqlite3.Connection, key: str, asset_id: UUID) -> Asset:
        row = db.execute(
            "SELECT data FROM assets WHERE agent_key = ? AND id = ?",
            (key, str(asset_id)),
        ).fetchone()
        if row is None:
            raise StorageError("Asset not found in this namespace", 404)
        return Asset.model_validate_json(row[0])

    def get_asset(self, key: str, asset_id: UUID) -> Asset:
        with self.connect() as db:
            self._agent(db, key)
            return self._asset(db, key, asset_id)

    def list_assets(
        self, key: str, images: bool = False, trash: bool = False, query: str = ""
    ) -> list[Asset]:
        with self.connect() as db:
            self._agent(db, key)
            rows = db.execute(
                "SELECT data FROM assets WHERE agent_key = ? AND deleted = ? AND (? = 0 OR is_image = 1) ORDER BY updated_at DESC, id",
                (key, int(trash), int(images)),
            )
            assets = [Asset.model_validate_json(row[0]) for row in rows]
            tokens = query.casefold().split()
            return [
                asset
                for asset in assets
                if all(
                    token
                    in " ".join(
                        [
                            asset.title,
                            asset.filename,
                            *asset.tags,
                            *(
                                f"{source.label} {source.url}"
                                for source in asset.sources
                            ),
                        ]
                    ).casefold()
                    for token in tokens
                )
            ]

    @staticmethod
    def _save_asset(db: sqlite3.Connection, asset: Asset) -> None:
        db.execute(
            """INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET deleted=excluded.deleted, updated_at=excluded.updated_at, data=excluded.data""",
            (
                str(asset.id),
                asset.agent_key,
                int(asset.deleted_at is not None),
                int(asset.is_image),
                asset.updated_at.isoformat(),
                asset.model_dump_json(),
            ),
        )

    def create_asset(self, asset: Asset) -> Asset:
        with self.connect(write=True) as db:
            self._agent(db, asset.agent_key)
            self._save_asset(db, asset)
            self._activity(
                db, asset.agent_key, "uploaded", "asset", str(asset.id), asset.title
            )
        return asset

    def update_asset(
        self, key: str, asset_id: UUID, value: Metadata, expected_version: int
    ) -> Asset:
        with self.connect(write=True) as db:
            current = self._asset(db, key, asset_id)
            self._check_version(current, expected_version)
            self._active(current)
            updated = Asset.model_validate(
                {
                    **current.model_dump(),
                    **value.model_dump(),
                    "version": current.version + 1,
                    "updated_at": utc_now(),
                }
            )
            self._save_asset(db, updated)
            self._activity(db, key, "updated", "asset", str(asset_id), updated.title)
            return updated

    def trash_asset(self, key: str, asset_id: UUID, expected_version: int) -> Asset:
        with self.connect(write=True) as db:
            current = self._asset(db, key, asset_id)
            self._check_version(current, expected_version)
            self._active(current)
            for row in db.execute(
                "SELECT data FROM resources WHERE agent_key = ? AND deleted = 0", (key,)
            ):
                if asset_id in asset_references(resource_adapter.validate_json(row[0])):
                    raise StorageError(
                        "Remove this asset from active resources before moving it to trash",
                        409,
                    )
            now = utc_now()
            updated = current.model_copy(
                update={
                    "deleted_at": now,
                    "updated_at": now,
                    "version": current.version + 1,
                }
            )
            self._save_asset(db, updated)
            db.execute(
                "UPDATE agents SET avatar_asset_id = NULL WHERE key = ? AND avatar_asset_id = ?",
                (key, str(asset_id)),
            )
            self._activity(db, key, "trashed", "asset", str(asset_id), current.title)
            return updated

    def restore_asset(self, key: str, asset_id: UUID, expected_version: int) -> Asset:
        with self.connect(write=True) as db:
            current = self._asset(db, key, asset_id)
            self._check_version(current, expected_version)
            if not current.deleted_at:
                raise StorageError("This asset is not in trash", 409)
            updated = current.model_copy(
                update={
                    "deleted_at": None,
                    "updated_at": utc_now(),
                    "version": current.version + 1,
                }
            )
            self._save_asset(db, updated)
            self._activity(db, key, "restored", "asset", str(asset_id), current.title)
            return updated

    def stats(self, key: str) -> Stats:
        with self.connect() as db:
            self._agent(db, key)
            counts = {kind: 0 for kind in ResourceKind}
            for row in db.execute(
                "SELECT kind, count(*) AS total FROM resources WHERE agent_key = ? AND deleted = 0 GROUP BY kind",
                (key,),
            ):
                counts[ResourceKind(row["kind"])] = row["total"]
            assets = [
                Asset.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM assets WHERE agent_key = ?", (key,)
                )
            ]
            return Stats(
                resources=counts,
                assets=sum(asset.deleted_at is None for asset in assets),
                images=sum(
                    asset.is_image and asset.deleted_at is None for asset in assets
                ),
                trashed_assets=sum(asset.deleted_at is not None for asset in assets),
                trashed_resources=db.execute(
                    "SELECT count(*) FROM resources WHERE agent_key = ? AND deleted = 1",
                    (key,),
                ).fetchone()[0],
                storage_bytes=sum(asset.size_bytes for asset in assets),
                database_bytes=sum(
                    path.stat().st_size
                    for path in [
                        self.database_path,
                        Path(str(self.database_path) + "-wal"),
                    ]
                    if path.exists()
                ),
                revisions=db.execute(
                    "SELECT count(*) FROM revisions WHERE agent_key = ?", (key,)
                ).fetchone()[0],
            )

    def export_manifest(self, db: sqlite3.Connection, key: str) -> ExportManifest:
        return ExportManifest(
            exported_at=utc_now(),
            agent=self._agent(db, key),
            resources=[
                resource_adapter.validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM resources WHERE agent_key = ? ORDER BY id", (key,)
                )
            ],
            versions=[
                ExportRevision(
                    resource_id=row["resource_id"],
                    version=row["version"],
                    created_at=row["created_at"],
                    snapshot=resource_adapter.validate_json(row["snapshot"]),
                )
                for row in db.execute(
                    "SELECT * FROM revisions WHERE agent_key = ? ORDER BY resource_id, version",
                    (key,),
                )
            ],
            assets=[
                Asset.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM assets WHERE agent_key = ? ORDER BY id", (key,)
                )
            ],
            activity=[
                Activity.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM activity WHERE agent_key = ? ORDER BY sequence DESC",
                    (key,),
                )
            ],
        )
