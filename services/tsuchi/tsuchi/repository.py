import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4, uuid5

from kadan_tracing import carrier, traced

from .config import StorageError
from .models import (
    Activity,
    Agent,
    AgentInput,
    AgentRename,
    AgentRun,
    AgentRunState,
    Asset,
    AssetMetadata,
    ConversationCompactionUpdate,
    ConversationMemoryStateUpdate,
    ConversationTaskPage,
    ConversationTaskStatus,
    DEFAULT_GROUP_CHAT_ROOM_ID,
    ExportManifest,
    ExportRevision,
    GroupChatDeliveryStatus,
    GroupChatEventSubscription,
    GroupChatEventSubscriptionInput,
    GroupChatEventSubscriptionUpdate,
    GroupChatMessage,
    GroupChatMessageInput,
    GroupChatMessagesPage,
    GroupChatMessageWriteResult,
    GroupChatParticipant,
    GroupChatParticipantInput,
    GroupChatParticipantTouch,
    GroupChatParticipantUpdate,
    GroupChatRoom,
    GroupChatRoomUpdate,
    GroupChatSubscriptionStatus,
    GroupChatWebhookDelivery,
    GroupChatWebhookDeliveryUpdate,
    ImageJob,
    ImageJobState,
    MarkdownContent,
    MarkdownResource,
    McpOAuthState,
    McpServer,
    McpServerInput,
    McpServerUpdate,
    Message,
    OwnedPersonaUpdate,
    PetJob,
    PetManifest,
    PetReferenceMode,
    PetRejection,
    PetState,
    ReferenceSnapshot,
    RequestReference,
    Resource,
    ResourceInput,
    ResourceKind,
    ResourceRecord,
    ResourceVersion,
    RunEvent,
    Runner,
    Stats,
    T3ThreadWatch,
    T3ThreadWatchInput,
    T3ThreadWatchUpdate,
    pet_step_order,
    resource_adapter,
    resource_input_adapter,
    utc_now,
)
from .transcript_migration import upgrade_transcripts


def search_expression(query: str) -> str:
    """Treat punctuation and FTS operators as text, never SQL/FTS syntax."""
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if len(tokens) > 30:
        raise StorageError("Resource searches cannot exceed 30 words.", 422)
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


SPACE_PARENTS = frozenset({ResourceKind.SPACE})
FOLDER_PARENTS = frozenset({ResourceKind.FOLDER})
PAGE_PARENTS = frozenset({ResourceKind.FOLDER, ResourceKind.SPACE})
PARENT_ERRORS = {
    SPACE_PARENTS: "Parent must be an active Space in this namespace",
    FOLDER_PARENTS: "Parent must be an active folder in this namespace",
    PAGE_PARENTS: "Parent must be an active folder or Space in this namespace",
}


def parent_kinds(kind: ResourceKind) -> frozenset[ResourceKind]:
    """Folders hold folders, Spaces and decks; anything else may sit in a Space."""
    if kind == ResourceKind.FOLDER:
        return FOLDER_PARENTS
    if kind in {ResourceKind.SPACE, ResourceKind.DECK}:
        return PAGE_PARENTS
    return SPACE_PARENTS


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
                CREATE TABLE IF NOT EXISTS runners (
                    runner_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL UNIQUE,
                    capacity INTEGER NOT NULL CHECK(capacity > 0), lease_expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runner_assignments (
                    agent_key TEXT PRIMARY KEY REFERENCES agents(key),
                    runner_id TEXT NOT NULL REFERENCES runners(runner_id)
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
                CREATE TABLE IF NOT EXISTS pet_jobs (
                    id TEXT PRIMARY KEY, agent_key TEXT NOT NULL REFERENCES agents(key),
                    status TEXT NOT NULL, updated_at TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS pet_jobs_namespace ON pet_jobs(agent_key, updated_at);
                CREATE INDEX IF NOT EXISTS pet_jobs_status ON pet_jobs(status);
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY, agent_key TEXT NOT NULL REFERENCES agents(key),
                    conversation_id TEXT NOT NULL REFERENCES resources(id),
                    status TEXT NOT NULL, updated_at TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS agent_runs_conversation ON agent_runs(agent_key, conversation_id);
                CREATE INDEX IF NOT EXISTS agent_runs_status ON agent_runs(status);
                CREATE TABLE IF NOT EXISTS image_jobs (
                    id TEXT PRIMARY KEY, agent_key TEXT NOT NULL REFERENCES agents(key),
                    run_id TEXT NOT NULL REFERENCES agent_runs(id),
                    conversation_id TEXT NOT NULL REFERENCES resources(id),
                    request_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, updated_at TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS image_jobs_status ON image_jobs(status);
                CREATE TABLE IF NOT EXISTS activity (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_key TEXT NOT NULL REFERENCES agents(key), data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS activity_namespace ON activity(agent_key, sequence);
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mcp_oauth_states (
                    state TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS t3_thread_watches (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS t3_thread_watches_status ON t3_thread_watches(status, updated_at);
                CREATE TABLE IF NOT EXISTS group_chat_rooms (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_chat_participants (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL REFERENCES group_chat_rooms(id),
                    handle TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    token_hash TEXT UNIQUE,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    UNIQUE(room_id, handle)
                );
                CREATE INDEX IF NOT EXISTS group_chat_participants_room
                    ON group_chat_participants(room_id, kind);
                CREATE INDEX IF NOT EXISTS group_chat_participants_token
                    ON group_chat_participants(token_hash);
                CREATE TABLE IF NOT EXISTS group_chat_messages (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL REFERENCES group_chat_rooms(id),
                    seq INTEGER NOT NULL,
                    sender_scope TEXT NOT NULL,
                    client_request_id TEXT,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    UNIQUE(room_id, seq),
                    UNIQUE(room_id, sender_scope, client_request_id)
                );
                CREATE INDEX IF NOT EXISTS group_chat_messages_room_seq
                    ON group_chat_messages(room_id, seq);
                CREATE TABLE IF NOT EXISTS group_chat_event_subscriptions (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL REFERENCES group_chat_rooms(id),
                    participant_id TEXT NOT NULL REFERENCES group_chat_participants(id),
                    event_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS group_chat_subs_participant
                    ON group_chat_event_subscriptions(participant_id, status);
                CREATE INDEX IF NOT EXISTS group_chat_subs_room_event
                    ON group_chat_event_subscriptions(room_id, event_name, status);
                CREATE TABLE IF NOT EXISTS group_chat_webhook_deliveries (
                    id TEXT PRIMARY KEY,
                    subscription_id TEXT NOT NULL REFERENCES group_chat_event_subscriptions(id),
                    room_id TEXT NOT NULL REFERENCES group_chat_rooms(id),
                    participant_id TEXT NOT NULL REFERENCES group_chat_participants(id),
                    seq INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    next_attempt_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    UNIQUE(subscription_id, seq)
                );
                CREATE INDEX IF NOT EXISTS group_chat_deliveries_pending
                    ON group_chat_webhook_deliveries(status, next_attempt_at);
            """)
            # Serialize the additive migration for existing storage volumes.
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(agents)")}
            if "selected_pet_id" not in columns:
                db.execute("ALTER TABLE agents ADD COLUMN selected_pet_id TEXT")
            if "persona_id" not in columns:
                db.execute("ALTER TABLE agents ADD COLUMN persona_id TEXT")
            upgrade_transcripts(db)
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
    def _runner(db: sqlite3.Connection, runner_id: str) -> Runner:
        row = db.execute(
            "SELECT * FROM runners WHERE runner_id = ?", (runner_id,)
        ).fetchone()
        if row is None:
            raise StorageError("Runner not found", 404)
        return Runner(
            **dict(row),
            agent_keys=[
                r[0]
                for r in db.execute(
                    "SELECT agent_key FROM runner_assignments WHERE runner_id = ? ORDER BY agent_key",
                    (runner_id,),
                )
            ],
        )

    def list_runners(self) -> list[Runner]:
        with self.connect() as db:
            return [
                self._runner(db, row[0])
                for row in db.execute(
                    "SELECT runner_id FROM runners ORDER BY runner_id"
                )
            ]

    def _expire_agent_runs(self, db: sqlite3.Connection, key: str) -> None:
        for row in db.execute(
            "SELECT data FROM agent_runs WHERE agent_key = ? AND status = 'running'",
            (key,),
        ).fetchall():
            run = AgentRun.model_validate_json(row[0])
            self._save_run(
                db,
                run.model_copy(
                    update={
                        "lease_expires_at": utc_now(),
                        "version": run.version + 1,
                        "updated_at": utc_now(),
                    }
                ),
            )

    def register_runner(
        self,
        runner_id: str,
        worker_id: UUID,
        capacity: int = 5,
        expected_worker_id: UUID | None = None,
    ) -> Runner:
        Runner(
            runner_id=runner_id,
            worker_id=worker_id,
            capacity=capacity,
            lease_expires_at=utc_now() + timedelta(seconds=60),
        )
        with self.connect(write=True) as db:
            duplicate = db.execute(
                "SELECT runner_id FROM runners WHERE worker_id = ? AND runner_id != ?",
                (str(worker_id), runner_id),
            ).fetchone()
            if duplicate:
                raise StorageError("Worker already owns another runner identity", 409)
            row = db.execute(
                "SELECT worker_id FROM runners WHERE runner_id = ?", (runner_id,)
            ).fetchone()
            if row is None:
                if expected_worker_id is not None:
                    raise StorageError("Runner ownership changed", 409)
                db.execute(
                    "INSERT INTO runners VALUES (?, ?, ?, ?)",
                    (
                        runner_id,
                        str(worker_id),
                        capacity,
                        (utc_now() + timedelta(seconds=60)).isoformat(),
                    ),
                )
            else:
                current = self._runner(db, runner_id)
                if (
                    current.worker_id != worker_id
                    and current.worker_id != expected_worker_id
                    and current.lease_expires_at > utc_now()
                ):
                    raise StorageError(
                        "Runner replacement requires its current worker ID", 409
                    )
                if capacity < len(current.agent_keys):
                    raise StorageError("Capacity is below assigned agent count", 409)
                if current.worker_id != worker_id:
                    for key in current.agent_keys:
                        self._expire_agent_runs(db, key)
                db.execute(
                    "UPDATE runners SET worker_id = ?, capacity = ?, lease_expires_at = ? WHERE runner_id = ?",
                    (
                        str(worker_id),
                        capacity,
                        (utc_now() + timedelta(seconds=60)).isoformat(),
                        runner_id,
                    ),
                )
            return self._runner(db, runner_id)

    def assign_runner(self, runner_id: str, worker_id: UUID) -> Runner:
        with self.connect(write=True) as db:
            runner = self._runner(db, runner_id)
            if runner.worker_id != worker_id or runner.lease_expires_at <= utc_now():
                raise StorageError("Runner ownership changed", 409)
            rows = db.execute(
                "SELECT key FROM agents WHERE key NOT IN (SELECT agent_key FROM runner_assignments) "
                "ORDER BY created_at, key LIMIT ?",
                (runner.capacity - len(runner.agent_keys),),
            ).fetchall()
            db.executemany(
                "INSERT INTO runner_assignments VALUES (?, ?)",
                [(row[0], runner_id) for row in rows],
            )
            return self._runner(db, runner_id)

    def reassign_agent(
        self, key: str, runner_id: str | None, expected_runner_id: str | None
    ) -> None:
        with self.connect(write=True) as db:
            self._agent(db, key)
            row = db.execute(
                "SELECT runner_id FROM runner_assignments WHERE agent_key = ?", (key,)
            ).fetchone()
            current = row[0] if row else None
            if current != expected_runner_id:
                raise StorageError("Agent assignment changed", 409)
            if current == runner_id:
                return
            if runner_id is not None:
                runner = self._runner(db, runner_id)
                if len(runner.agent_keys) >= runner.capacity:
                    raise StorageError("Runner is at capacity", 409)
            self._expire_agent_runs(db, key)
            db.execute("DELETE FROM runner_assignments WHERE agent_key = ?", (key,))
            if runner_id is not None:
                db.execute(
                    "INSERT INTO runner_assignments VALUES (?, ?)", (key, runner_id)
                )

    @staticmethod
    def _assigned_worker(db: sqlite3.Connection, key: str, worker_id: UUID) -> None:
        row = db.execute(
            "SELECT r.lease_expires_at FROM runner_assignments a JOIN runners r USING (runner_id) "
            "WHERE a.agent_key = ? AND r.worker_id = ?",
            (key, str(worker_id)),
        ).fetchone()
        if row is None or row[0] <= utc_now().isoformat():
            raise StorageError(
                "Agent run is not owned by this assigned runner worker", 409
            )

    @staticmethod
    def _run(db: sqlite3.Connection, key: str, run_id: UUID) -> AgentRun:
        row = db.execute(
            "SELECT data FROM agent_runs WHERE agent_key = ? AND id = ?",
            (key, str(run_id)),
        ).fetchone()
        if row is None:
            raise StorageError("Agent run not found", 404)
        return AgentRun.model_validate_json(row[0])

    @staticmethod
    def _save_run(db: sqlite3.Connection, run: AgentRun) -> None:
        db.execute(
            """INSERT INTO agent_runs VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                updated_at=excluded.updated_at, data=excluded.data""",
            (
                str(run.id),
                run.agent_key,
                str(run.conversation_id),
                run.status,
                run.updated_at.isoformat(),
                run.model_dump_json(
                    exclude={
                        name
                        for name, empty in (
                            ("references", not run.references),
                            ("trigger_event", run.trigger_event is None),
                        )
                        if empty
                    }
                    or None
                ),
            ),
        )

    @staticmethod
    def _image_job(db: sqlite3.Connection, key: str, job_id: UUID) -> ImageJob:
        row = db.execute(
            "SELECT data FROM image_jobs WHERE agent_key = ? AND id = ?",
            (key, str(job_id)),
        ).fetchone()
        if row is None:
            raise StorageError("Image job not found", 404)
        return ImageJob.model_validate_json(row[0])

    @staticmethod
    def _save_image_job(db: sqlite3.Connection, job: ImageJob) -> None:
        db.execute(
            """INSERT INTO image_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                updated_at=excluded.updated_at, data=excluded.data""",
            (
                str(job.id),
                job.agent_key,
                str(job.run_id),
                str(job.conversation_id),
                str(job.request_id),
                job.status,
                job.updated_at.isoformat(),
                job.model_dump_json(),
            ),
        )

    def _conversation(
        self, db: sqlite3.Connection, key: str, conversation_id: UUID
    ) -> Resource:
        conversation = self._resource(db, key, conversation_id)
        self._active(conversation)
        if conversation.kind != ResourceKind.CONVERSATION:
            raise StorageError("Agent runs require an active conversation", 409)
        return conversation

    def _append_run_message(
        self, db: sqlite3.Connection, key: str, conversation: Resource, message: Message
    ) -> Resource:
        for existing in conversation.messages:
            if existing.id == message.id:
                if existing.model_dump(exclude={"created_at"}) != message.model_dump(
                    exclude={"created_at"}
                ):
                    raise StorageError("This message ID is already in use", 409)
                return conversation
        value = resource_input_adapter.validate_python(
            {
                **conversation.model_dump(exclude=set(ResourceRecord.model_fields)),
                "messages": [*conversation.messages, message],
            }
        )
        return self._update_resource(
            db, key, conversation, value, conversation.version, "message_added"
        )

    @traced("kadan.storage.create_run")
    def create_run(
        self,
        key: str,
        conversation_id: UUID,
        message: Message,
        expected_version: int,
        persona_id: UUID | None = None,
        references: list[RequestReference] | None = None,
    ) -> AgentRun:
        references = references or []
        if len(references) > 3 or len({item.resource_id for item in references}) != len(
            references
        ):
            raise StorageError("Select up to three distinct references", 422)
        if message.role != "user":
            raise StorageError("An agent run must start with a user message", 409)
        with self.connect(write=True) as db:
            conversation = self._conversation(db, key, conversation_id)
            row = db.execute(
                "SELECT data FROM agent_runs WHERE id = ?", (str(message.id),)
            ).fetchone()
            if row is not None:
                existing = AgentRun.model_validate_json(row[0])
                if (
                    existing.agent_key == key
                    and existing.conversation_id == conversation_id
                    and existing.trigger_message_id == message.id
                    and existing.persona_id == persona_id
                    and [
                        (item.resource_id, item.version) for item in existing.references
                    ]
                    == [
                        (item.resource_id, item.expected_version) for item in references
                    ]
                    and any(
                        item.model_dump(exclude={"created_at"})
                        == message.model_dump(exclude={"created_at"})
                        for item in conversation.messages
                    )
                ):
                    return existing
                raise StorageError("This agent run ID is already in use", 409)
            self._check_version(conversation, expected_version)
            if persona_id is not None:
                persona = self._resource(db, key, persona_id)
                self._active(persona)
                if persona.kind != ResourceKind.PERSONA:
                    raise StorageError("Persona must be a persona resource", 409)
            snapshots = []
            for selected in references:
                resource = self._resource(db, key, selected.resource_id)
                self._active(resource)
                if resource.kind not in {ResourceKind.SPACE, ResourceKind.MEMORY}:
                    raise StorageError(
                        "References require Markdown spaces or memories", 409
                    )
                self._check_version(resource, selected.expected_version)
                if (
                    sum(len(item.markdown) for item in snapshots)
                    + len(resource.markdown)
                    > 12000
                ):
                    raise StorageError(
                        "Selected references exceed 12000 characters", 422
                    )
                snapshots.append(
                    ReferenceSnapshot(
                        resource_id=resource.id,
                        kind=resource.kind,
                        title=resource.title,
                        version=resource.version,
                        markdown=resource.markdown,
                    )
                )
            self._append_run_message(db, key, conversation, message)
            now = utc_now()
            run = AgentRun(
                id=message.id,
                trace_parent=carrier().get("traceparent"),
                agent_key=key,
                conversation_id=conversation_id,
                trigger_message_id=message.id,
                persona_id=persona_id,
                references=snapshots,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._save_run(db, run)
            return run

    @traced("kadan.storage.create_event_run")
    def create_event_run(
        self,
        key: str,
        conversation_id: UUID,
        run_id: UUID,
        origin_run_id: UUID | None,
        event: RunEvent,
    ) -> AgentRun:
        """Queue a run woken by an external event, idempotent by its run ID."""
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM agent_runs WHERE id = ?", (str(run_id),)
            ).fetchone()
            if row is not None:
                existing = AgentRun.model_validate_json(row[0])
                if (
                    existing.agent_key == key
                    and existing.conversation_id == conversation_id
                    and existing.trigger_event == event
                ):
                    return existing
                raise StorageError("This agent run ID is already in use", 409)
            self._conversation(db, key, conversation_id)
            if origin_run_id is not None:
                origin = self._run(db, key, origin_run_id)
                if origin.conversation_id != conversation_id:
                    raise StorageError("The event belongs to another conversation", 409)
                persona_id = origin.persona_id
                references = origin.references
            else:
                agent = self._agent(db, key)
                persona_id = agent.persona_id
                references = []
            now = utc_now()
            run = AgentRun(
                id=run_id,
                trace_parent=carrier().get("traceparent"),
                agent_key=key,
                conversation_id=conversation_id,
                trigger_event=event,
                persona_id=persona_id,
                references=references,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._save_run(db, run)
            return run

    def get_run(self, key: str, run_id: UUID) -> AgentRun:
        with self.connect() as db:
            return self._run(db, key, run_id)

    def active_runs(self) -> list[AgentRun]:
        with self.connect() as db:
            return sorted(
                (
                    AgentRun.model_validate_json(row[0])
                    for row in db.execute(
                        "SELECT r.data FROM agent_runs r JOIN resources c "
                        "ON c.id = r.conversation_id AND c.agent_key = r.agent_key "
                        "WHERE r.status IN ('queued', 'running') "
                        "AND c.kind = 'conversation' AND c.deleted = 0"
                    )
                ),
                key=lambda run: (run.created_at, run.id),
            )

    def conversation_runs(self, key: str, conversation_id: UUID) -> list[AgentRun]:
        with self.connect() as db:
            self._conversation(db, key, conversation_id)
            return [
                AgentRun.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM agent_runs WHERE agent_key = ? AND conversation_id = ? ORDER BY updated_at, id",
                    (key, str(conversation_id)),
                )
            ]

    def _owned_run(
        self, db: sqlite3.Connection, run: AgentRun, worker_id: UUID
    ) -> None:
        self._assigned_worker(db, run.agent_key, worker_id)
        if (
            run.status != "running"
            or run.owner_id != worker_id
            or run.lease_expires_at <= utc_now()
        ):
            raise StorageError("Agent run lease is not owned by this worker", 409)

    @traced("kadan.storage.update_run")
    def update_run(
        self,
        key: str,
        run_id: UUID,
        value: AgentRunState,
        expected_version: int,
        worker_id: UUID,
    ) -> AgentRun:
        with self.connect(write=True) as db:
            current = self._run(db, key, run_id)
            self._assigned_worker(db, key, worker_id)
            self._check_version(current, expected_version)
            if value.transcript[: len(current.transcript)] != current.transcript:
                raise StorageError("Persisted agent transcript is immutable", 409)
            now = utc_now()
            claiming = current.status == "queued" or (
                current.status == "running" and current.lease_expires_at <= now
            )
            if claiming:
                if value.status != "running":
                    raise StorageError("Claim the agent run before updating it", 409)
                if (
                    current.status == "running"
                    and value.transcript != current.transcript
                ):
                    raise StorageError(
                        "Recovering a run cannot change its transcript", 409
                    )
            else:
                self._owned_run(db, current, worker_id)
                if value.status not in {"running", "completed", "failed", "cancelled"}:
                    raise StorageError("Invalid agent run state transition", 409)
            if value.status == "running":
                if value.owner_id != worker_id:
                    raise StorageError("A worker must own its agent run lease", 409)
                if not now < value.lease_expires_at <= now + timedelta(minutes=5):
                    raise StorageError(
                        "Agent run lease must expire within five minutes", 409
                    )
                self._conversation(db, key, current.conversation_id)
                for row in db.execute(
                    "SELECT data FROM agent_runs WHERE agent_key = ? AND conversation_id = ? AND status IN ('queued', 'running') AND id != ?",
                    (key, str(current.conversation_id), str(run_id)),
                ):
                    other = AgentRun.model_validate_json(row[0])
                    if other.status == "running" and other.lease_expires_at > now:
                        raise StorageError("Another run owns this conversation", 409)
                    # Heartbeats change updated_at, but cannot move unfinished
                    # work behind a later message after a worker restart.
                    if claiming and (other.created_at, other.id) < (
                        current.created_at,
                        current.id,
                    ):
                        raise StorageError(
                            "An earlier run must finish before this run can claim the conversation",
                            409,
                        )
            updated = AgentRun.model_validate(
                {
                    **current.model_dump(),
                    **value.model_dump(),
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            self._save_run(db, updated)
            return updated

    def cancel_run(self, key: str, run_id: UUID) -> AgentRun:
        with self.connect(write=True) as db:
            current = self._run(db, key, run_id)
            if current.status in {"completed", "failed", "cancelled"}:
                return current
            updated = current.model_copy(
                update={
                    "status": "cancelled",
                    "owner_id": None,
                    "lease_expires_at": None,
                    "version": current.version + 1,
                    "updated_at": utc_now(),
                }
            )
            self._save_run(db, updated)
            return updated

    @traced("kadan.storage.deliver_run_message")
    def deliver_run_message(
        self,
        key: str,
        run_id: UUID,
        worker_id: UUID,
        message: Message,
        image_job_id: UUID | None = None,
    ) -> Resource:
        if message.role != "assistant":
            raise StorageError("The messaging tool delivers assistant messages", 409)
        with self.connect(write=True) as db:
            run = self._run(db, key, run_id)
            conversation = self._conversation(db, key, run.conversation_id)
            job = None
            if image_job_id is not None:
                job = self._image_job(db, key, image_job_id)
                if (
                    job.conversation_id != run.conversation_id
                    or job.status != "completed"
                ):
                    raise StorageError(
                        "Only completed images from this conversation can be delivered",
                        409,
                    )
                message = message.model_copy(
                    update={
                        "attachment_ids": list(
                            dict.fromkeys([*message.attachment_ids, job.asset_id])
                        )
                    }
                )
            if message.id in run.delivered_message_ids:
                return self._append_run_message(db, key, conversation, message)
            self._owned_run(db, run, worker_id)
            if job is not None and job.delivered_message_id is not None:
                return conversation
            conversation = self._append_run_message(db, key, conversation, message)
            now = utc_now()
            self._save_run(
                db,
                run.model_copy(
                    update={
                        "delivered_message_ids": [
                            *run.delivered_message_ids,
                            message.id,
                        ],
                        "version": run.version + 1,
                        "updated_at": now,
                    }
                ),
            )
            if job is not None:
                self._save_image_job(
                    db,
                    job.model_copy(
                        update={
                            "delivered_message_id": message.id,
                            "version": job.version + 1,
                            "updated_at": now,
                        }
                    ),
                )
            return conversation

    @traced("kadan.storage.create_image_job")
    def create_image_job(
        self,
        key: str,
        run_id: UUID,
        job_id: UUID,
        prompt: str,
        request_id: UUID,
        worker_id: UUID,
    ) -> ImageJob:
        if not prompt.strip():
            raise StorageError("Describe the image to create", 409)
        with self.connect(write=True) as db:
            run = self._run(db, key, run_id)
            row = db.execute(
                "SELECT data FROM image_jobs WHERE id = ?", (str(job_id),)
            ).fetchone()
            if row is not None:
                current = ImageJob.model_validate_json(row[0])
                if (
                    current.agent_key == key
                    and current.run_id == run_id
                    and current.prompt == prompt
                    and current.request_id == request_id
                ):
                    return current
                raise StorageError("This image job ID is already in use", 409)
            self._owned_run(db, run, worker_id)
            self._conversation(db, key, run.conversation_id)
            if db.execute(
                "SELECT 1 FROM image_jobs WHERE request_id = ?", (str(request_id),)
            ).fetchone():
                raise StorageError("This inference request ID is already in use", 409)
            now = utc_now()
            job = ImageJob(
                id=job_id,
                trace_parent=carrier().get("traceparent"),
                agent_key=key,
                run_id=run_id,
                conversation_id=run.conversation_id,
                prompt=prompt,
                request_id=request_id,
                event_run_id=uuid5(job_id, "event"),
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._save_image_job(db, job)
            return job

    def get_image_job(self, key: str, job_id: UUID) -> ImageJob:
        with self.connect() as db:
            return self._image_job(db, key, job_id)

    def active_image_jobs(self) -> list[ImageJob]:
        with self.connect() as db:
            return [
                ImageJob.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM image_jobs WHERE status IN ('queued', 'running', 'cancelling') ORDER BY updated_at, id"
                )
            ]

    def conversation_image_jobs(
        self, key: str, conversation_id: UUID
    ) -> list[ImageJob]:
        with self.connect() as db:
            self._conversation(db, key, conversation_id)
            return [
                ImageJob.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM image_jobs WHERE agent_key = ? AND conversation_id = ? ORDER BY updated_at, id",
                    (key, str(conversation_id)),
                )
            ]

    @traced("kadan.storage.image_event")
    def _image_event(self, db: sqlite3.Connection, job: ImageJob) -> None:
        # The terminal result and its wake-up event share one transaction: a
        # restart can neither lose the event nor enqueue it a second time.
        origin = self._run(db, job.agent_key, job.run_id)
        if db.execute(
            "SELECT 1 FROM agent_runs WHERE id = ?", (str(job.event_run_id),)
        ).fetchone():
            raise StorageError("The image event run ID is already in use", 409)
        now = utc_now()
        self._save_run(
            db,
            AgentRun(
                id=job.event_run_id,
                trace_parent=carrier().get("traceparent") or job.trace_parent,
                agent_key=job.agent_key,
                conversation_id=job.conversation_id,
                trigger_job_id=job.id,
                persona_id=origin.persona_id,
                references=origin.references,
                version=1,
                created_at=now,
                updated_at=now,
            ),
        )

    @traced("kadan.storage.update_image_job")
    def update_image_job(
        self,
        key: str,
        job_id: UUID,
        value: ImageJobState,
        expected_version: int,
        worker_id: UUID,
    ) -> ImageJob:
        with self.connect(write=True) as db:
            current = self._image_job(db, key, job_id)
            terminal = {"completed", "failed", "cancelled"}
            if current.status in terminal:
                state = ImageJobState.model_validate(
                    current.model_dump(include=set(ImageJobState.model_fields))
                )
                if state == value:
                    return current
                raise StorageError("Completed image job state is immutable", 409)
            self._check_version(current, expected_version)
            now = utc_now()
            claiming = current.status == "queued" or current.lease_expires_at <= now
            if claiming:
                target = "cancelling" if current.status == "cancelling" else "running"
                if value.status != target:
                    raise StorageError("Claim the image job before updating it", 409)
            elif current.owner_id != worker_id:
                raise StorageError("Image job lease is not owned by this worker", 409)
            elif value.status not in (
                {"cancelling", "cancelled", "failed"}
                if current.status == "cancelling"
                else {"running", "completed", "failed", "cancelled"}
            ):
                raise StorageError("Invalid image job state transition", 409)
            if value.status in {"running", "cancelling"}:
                if value.owner_id != worker_id:
                    raise StorageError("A worker must own its image job lease", 409)
                if not now < value.lease_expires_at <= now + timedelta(minutes=5):
                    raise StorageError(
                        "Image job lease must expire within five minutes", 409
                    )
            if value.upload_status != current.upload_status and not (
                current.upload_status == "not_started"
                and value.upload_status == "pending"
                and current.status == value.status == "running"
                and not claiming
            ):
                raise StorageError("Only storage can settle an image upload", 409)
            if value.status == "completed":
                if value.asset_id != uuid5(job_id, "image"):
                    raise StorageError("Image artwork identity cannot change", 409)
                asset = self._asset(db, key, value.asset_id)
                if asset.deleted_at or not asset.is_image:
                    raise StorageError(
                        "Completed image jobs require an active image asset", 409
                    )
            updated = ImageJob.model_validate(
                {
                    **current.model_dump(),
                    **value.model_dump(),
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            self._save_image_job(db, updated)
            if updated.status in terminal:
                self._image_event(db, updated)
            return updated

    def fail_image_upload(self, key: str, job_id: UUID) -> ImageJob:
        # FileStorage holds the upload's file lock: no old request may register
        # artwork after this transaction closes its persisted upload gate.
        with self.connect(write=True) as db:
            job = self._image_job(db, key, job_id)
            if job.upload_status != "pending":
                return job
            updated = job.model_copy(
                update={
                    "upload_status": "failed",
                    "version": job.version + 1,
                    "updated_at": utc_now(),
                }
            )
            self._save_image_job(db, updated)
            return updated

    def cancel_image_job(self, key: str, job_id: UUID) -> ImageJob:
        with self.connect(write=True) as db:
            current = self._image_job(db, key, job_id)
            if current.status in {"completed", "failed", "cancelled", "cancelling"}:
                return current
            updated = current.model_copy(
                update={
                    "status": "cancelled"
                    if current.status == "queued"
                    else "cancelling",
                    "version": current.version + 1,
                    "updated_at": utc_now(),
                }
            )
            self._save_image_job(db, updated)
            if updated.status == "cancelled":
                self._image_event(db, updated)
            return updated

    def conversation_task_status(
        self, key: str, conversation_id: UUID, offset: int
    ) -> ConversationTaskPage:
        page_size = 5
        with self.connect() as db:
            self._conversation(db, key, conversation_id)
            rows = db.execute(
                """
                SELECT id, 'image' AS kind, status, updated_at,
                       substr(json_extract(data, '$.prompt'), 1, 500) AS request_excerpt,
                       substr(json_extract(data, '$.error'), 1, 1000) AS outcome_excerpt
                FROM image_jobs WHERE agent_key = :key AND conversation_id = :conversation
                ORDER BY updated_at DESC, id DESC
                LIMIT :limit OFFSET :offset
                """,
                {
                    "key": key,
                    "conversation": str(conversation_id),
                    "limit": page_size + 1,
                    "offset": offset,
                },
            ).fetchall()
        return ConversationTaskPage(
            tasks=[
                ConversationTaskStatus.model_validate(dict(row))
                for row in rows[:page_size]
            ],
            next_offset=offset + page_size if len(rows) > page_size else None,
        )

    @staticmethod
    def _agent(db: sqlite3.Connection, key: str) -> Agent:
        row = db.execute("SELECT * FROM agents WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise StorageError("Agent namespace not found", 404)
        return Agent.model_validate(dict(row))

    def get_agent(self, key: str) -> Agent:
        with self.connect() as db:
            return self._agent(db, key)

    def _owned_persona(
        self, db: sqlite3.Connection, key: str
    ) -> MarkdownResource | None:
        agent = self._agent(db, key)
        if agent.persona_id is None:
            return None
        persona = self._resource(db, key, agent.persona_id)
        self._active(persona)
        if persona.kind != ResourceKind.PERSONA:
            raise StorageError(
                "The agent's owned persona must be a persona resource", 409
            )
        return persona

    def get_owned_persona(self, key: str) -> MarkdownResource | None:
        with self.connect() as db:
            return self._owned_persona(db, key)

    def save_owned_persona(
        self, key: str, value: OwnedPersonaUpdate
    ) -> MarkdownResource:
        with self.connect(write=True) as db:
            current = self._owned_persona(db, key)
            if current is None:
                if value.expected_version is not None:
                    raise StorageError(
                        "This agent has no saved persona. Reload before saving.", 409
                    )
                now = utc_now()
                persona = MarkdownResource(
                    kind=ResourceKind.PERSONA,
                    title=value.title,
                    markdown=value.markdown,
                    id=uuid4(),
                    agent_key=key,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                self._save_resource(db, persona)
                db.execute(
                    "UPDATE agents SET persona_id = ? WHERE key = ?",
                    (str(persona.id), key),
                )
                self._activity(
                    db, key, "created", "resource", str(persona.id), persona.title
                )
                return persona
            if value.expected_version is None:
                raise StorageError(
                    "This agent already has a persona. Reload before saving.", 409
                )
            content = MarkdownContent.model_validate(
                {
                    **current.model_dump(exclude=set(ResourceRecord.model_fields)),
                    "title": value.title,
                    "markdown": value.markdown,
                }
            )
            return self._update_resource(
                db, key, current, content, value.expected_version, "updated"
            )

    def list_agents(self) -> list[Agent]:
        with self.connect() as db:
            return [
                Agent.model_validate(dict(row))
                for row in db.execute("SELECT * FROM agents ORDER BY name, key")
            ]

    def rename_agent(self, key: str, value: AgentRename) -> Agent:
        with self.connect(write=True) as db:
            current = self._agent(db, key)
            if current.name == value.name:
                return current
            if current.name != value.expected_name:
                raise StorageError("Agent name changed since it was loaded.", 409)
            db.execute("UPDATE agents SET name = ? WHERE key = ?", (value.name, key))
            self._activity(db, key, "updated", "agent", key, value.name)
            return current.model_copy(update={"name": value.name})

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

    @traced("kadan.storage.delete_agent")
    def delete_agent(self, key: str) -> list[UUID]:
        with self.connect(write=True) as db:
            self._agent(db, key)
            asset_ids = [
                UUID(row[0])
                for row in db.execute(
                    "SELECT id FROM assets WHERE agent_key = ?", (key,)
                ).fetchall()
            ]
            # Clear optional pointers on agents before removing referenced rows.
            db.execute(
                "UPDATE agents SET persona_id = NULL, selected_pet_id = NULL WHERE key = ?",
                (key,),
            )
            db.execute("DELETE FROM image_jobs WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM agent_runs WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM pet_jobs WHERE agent_key = ?", (key,))
            db.execute(
                "DELETE FROM t3_thread_watches WHERE json_extract(data, '$.origin_agent_key') = ?",
                (key,),
            )
            db.execute("DELETE FROM runner_assignments WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM resource_search WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM revisions WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM resources WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM assets WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM activity WHERE agent_key = ?", (key,))
            db.execute("DELETE FROM agents WHERE key = ?", (key,))
            return asset_ids

    @staticmethod
    def _pet(db: sqlite3.Connection, key: str, pet_id: UUID) -> PetJob:
        row = db.execute(
            "SELECT data FROM pet_jobs WHERE agent_key = ? AND id = ?",
            (key, str(pet_id)),
        ).fetchone()
        if row is None:
            raise StorageError("Pet not found in this namespace", 404)
        return PetJob.model_validate_json(row[0])

    @staticmethod
    def _save_pet(db: sqlite3.Connection, pet: PetJob) -> None:
        db.execute(
            """INSERT INTO pet_jobs VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status,
            updated_at=excluded.updated_at, data=excluded.data""",
            (
                str(pet.id),
                pet.agent_key,
                pet.status,
                pet.updated_at.isoformat(),
                pet.model_dump_json(),
            ),
        )

    @staticmethod
    def _require_no_other_active_pet(
        db: sqlite3.Connection, key: str, pet_id: UUID
    ) -> None:
        if db.execute(
            "SELECT 1 FROM pet_jobs WHERE agent_key = ? AND id != ? "
            "AND status IN ('queued', 'running', 'cancelling') LIMIT 1",
            (key, str(pet_id)),
        ).fetchone():
            raise StorageError(
                "Another pet creation or repair is active for this agent. "
                "Wait for it to finish or cancel it first.",
                409,
            )

    def create_pet(
        self,
        key: str,
        pet_id: UUID,
        prompt: str,
        reference_mode: PetReferenceMode = "template_strips",
    ) -> PetJob:
        now = utc_now()
        pet = PetJob(
            id=pet_id,
            trace_parent=carrier().get("traceparent"),
            agent_key=key,
            prompt=prompt,
            reference_mode=reference_mode,
            total_steps=len(pet_step_order(reference_mode)),
            version=1,
            created_at=now,
            updated_at=now,
        )
        with self.connect(write=True) as db:
            self._agent(db, key)
            row = db.execute(
                "SELECT data FROM pet_jobs WHERE id = ?", (str(pet_id),)
            ).fetchone()
            if row is not None:
                current = PetJob.model_validate_json(row[0])
                if (
                    current.agent_key == key
                    and current.prompt == prompt
                    and current.origin == "generated"
                    and current.reference_mode == reference_mode
                ):
                    return current
                raise StorageError("This pet job ID is already in use", 409)
            self._require_no_other_active_pet(db, key, pet_id)
            self._save_pet(db, pet)
        return pet

    def import_pet(
        self, key: str, pet_id: UUID, prompt: str, manifest: PetManifest
    ) -> PetJob:
        now = utc_now()
        pet = PetJob(
            id=pet_id,
            trace_parent=carrier().get("traceparent"),
            agent_key=key,
            prompt=prompt,
            origin="imported",
            status="ready",
            stage="Imported",
            manifest=manifest,
            version=1,
            created_at=now,
            updated_at=now,
        )
        with self.connect(write=True) as db:
            self._agent(db, key)
            row = db.execute(
                "SELECT data FROM pet_jobs WHERE id = ?", (str(pet_id),)
            ).fetchone()
            if row is not None:
                current = PetJob.model_validate_json(row[0])
                if (
                    current.agent_key == key
                    and current.origin == "imported"
                    and current.prompt == prompt
                    and current.manifest == manifest
                ):
                    return current
                raise StorageError("This pet job ID is already in use", 409)
            self._validate_pet_assets(db, key, pet)
            self._save_pet(db, pet)
        return pet

    def revise_pet(
        self,
        key: str,
        pet_id: UUID,
        revision_id: UUID,
        expected_version: int,
        frame_names: list[str],
        reason: str,
    ) -> PetJob:
        reason = reason.strip()
        if not reason or len(reason) > 10000:
            raise StorageError("Describe why the frames need repair")
        if (
            not frame_names
            or len(set(frame_names)) != len(frame_names)
            or any(name in {"base", "character-sheet"} for name in frame_names)
        ):
            raise StorageError(
                "Choose distinct completed frames to repair; references cannot be replaced"
            )
        with self.connect(write=True) as db:
            self._agent(db, key)
            source = self._pet(db, key, pet_id)
            self._check_version(source, expected_version)
            if source.status in {"queued", "running", "cancelling"}:
                raise StorageError("Stop pet creation before repairing frames", 409)
            if source.reference_mode == "template_strips":
                rejected_names = {name.split("/", 1)[0] for name in frame_names}
                if rejected_names & {"base", "character-sheet"}:
                    raise StorageError(
                        "Choose distinct completed frames to repair; references cannot be replaced"
                    )
            else:
                rejected_names = set(frame_names)
            if not rejected_names <= {step.name for step in source.steps}:
                raise StorageError("Only completed frames can be repaired")
            rejections = [
                *source.rejections,
                *(
                    PetRejection(step=step, reason=reason)
                    for step in source.steps
                    if step.name in rejected_names
                ),
            ]
            existing = db.execute(
                "SELECT data FROM pet_jobs WHERE id = ?", (str(revision_id),)
            ).fetchone()
            if existing is not None:
                current = PetJob.model_validate_json(existing[0])
                if (
                    current.agent_key == key
                    and current.parent_pet_id == source.id
                    and current.rejections == rejections
                    and current.prompt == source.prompt
                ):
                    return current
                raise StorageError("This pet job ID is already in use", 409)
            self._require_no_other_active_pet(db, key, revision_id)
            now = utc_now()
            steps = [step for step in source.steps if step.name not in rejected_names]
            revision = PetJob(
                id=revision_id,
                trace_parent=carrier().get("traceparent"),
                agent_key=key,
                prompt=source.prompt,
                reference_mode=source.reference_mode,
                total_steps=source.total_steps,
                parent_pet_id=source.id,
                rejections=rejections,
                steps=steps,
                completed_steps=len(steps),
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._validate_pet_assets(db, key, revision)
            self._save_pet(db, revision)
            return revision

    def get_pet(self, key: str, pet_id: UUID) -> PetJob:
        with self.connect() as db:
            self._agent(db, key)
            return self._pet(db, key, pet_id)

    def list_pets(self, key: str) -> list[PetJob]:
        with self.connect() as db:
            self._agent(db, key)
            return [
                PetJob.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM pet_jobs WHERE agent_key = ? ORDER BY updated_at DESC, id",
                    (key,),
                )
            ]

    def active_pet_jobs(self) -> list[PetJob]:
        with self.connect() as db:
            return [
                PetJob.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM pet_jobs WHERE status IN ('queued', 'running', 'cancelling') ORDER BY updated_at, id"
                )
            ]

    def _validate_pet_assets(
        self, db: sqlite3.Connection, key: str, pet: PetState
    ) -> None:
        provenance_ids = {
            step.provenance_asset_id
            for step in [*pet.steps, *(rejection.step for rejection in pet.rejections)]
        }
        for asset_id in provenance_ids:
            asset = self._asset(db, key, asset_id)
            if asset.deleted_at or asset.media_type != "application/json":
                raise StorageError(
                    "Pet provenance must reference active JSON assets in this namespace"
                )
        for asset_id in pet.asset_references() - provenance_ids:
            asset = self._asset(db, key, asset_id)
            if asset.deleted_at or not asset.is_image:
                raise StorageError(
                    "Pet artwork must reference active raster images in this namespace"
                )

    def update_pet(
        self,
        key: str,
        pet_id: UUID,
        value: PetState,
        expected_version: int,
        worker_id: UUID | None = None,
    ) -> PetJob:
        with self.connect(write=True) as db:
            current = self._pet(db, key, pet_id)
            self._check_version(current, expected_version)
            now = utc_now()
            transitions = {
                "queued": {"running", "cancelled"},
                "running": {
                    "running",
                    "cancelling",
                    "cancelled",
                    "interrupted",
                    "failed",
                    "ready",
                },
                "cancelling": {"cancelling", "cancelled", "interrupted", "failed"},
                "interrupted": {"queued"},
                "failed": {"queued"},
                "cancelled": {"queued"},
                "ready": set(),
            }
            if value.status not in transitions[current.status]:
                raise StorageError("Invalid pet job state transition", 409)
            if value.status == "queued":
                self._require_no_other_active_pet(db, key, pet_id)
            updated_steps = {step.name: step for step in value.steps}
            if any(updated_steps.get(step.name) != step for step in current.steps):
                raise StorageError("Completed pet steps are immutable", 409)
            if (
                value.origin != current.origin
                or value.reference_mode != current.reference_mode
                or value.total_steps != current.total_steps
                or value.parent_pet_id != current.parent_pet_id
                or value.rejections != current.rejections
            ):
                raise StorageError("Pet revision history is immutable", 409)
            if len(value.steps) > len(current.steps) + 1:
                raise StorageError(
                    "Pet progress must persist one completed step at a time", 409
                )
            active = current.status in {"running", "cancelling"}
            expired = active and current.lease_expires_at <= now
            cancellation = (
                active
                and value.status == "cancelling"
                and value.cancel_requested
                and worker_id != current.owner_id
            )
            if cancellation:
                # A user can request cancellation, but cannot alter worker progress.
                unchanged = {
                    "steps",
                    "completed_steps",
                    "pending_step_name",
                    "pending_request_id",
                    "pending_seed",
                    "additional_pending_frames",
                    "owner_id",
                    "lease_expires_at",
                    "manifest",
                    "error",
                }
                if any(
                    getattr(current, field) != getattr(value, field)
                    for field in unchanged
                ):
                    raise StorageError(
                        "Cancellation cannot modify generation progress", 409
                    )
            elif expired and (
                value.status == "interrupted"
                or (value.status == "cancelled" and current.cancel_requested)
            ):
                if (
                    value.steps != current.steps
                    or value.pending_step_name != current.pending_step_name
                    or value.pending_request_id != current.pending_request_id
                    or value.pending_seed != current.pending_seed
                    or value.additional_pending_frames
                    != current.additional_pending_frames
                ):
                    raise StorageError(
                        "Recovery cannot modify completed or pending steps", 409
                    )
            elif active and (expired or worker_id != current.owner_id):
                raise StorageError("Pet job lease is not owned by this worker", 409)
            if (
                current.cancel_requested
                and value.status != "queued"
                and not value.cancel_requested
            ):
                raise StorageError(
                    "Cancellation requests cannot be cleared by worker updates", 409
                )
            if value.status in {"running", "cancelling"} and not cancellation:
                if worker_id is None or value.owner_id != worker_id:
                    raise StorageError("A worker must own its pet job lease", 409)
                if not now < value.lease_expires_at <= now + timedelta(minutes=5):
                    raise StorageError(
                        "Pet job lease must expire within five minutes", 409
                    )
            if (
                current.status in {"queued", "interrupted", "failed", "cancelled"}
                and value.steps != current.steps
            ):
                raise StorageError(
                    "Claiming or resuming a pet cannot alter completed steps", 409
                )
            # Validate new references only: cancellation and failure must still work
            # after someone has moved already completed artwork to trash.
            if len(value.steps) > len(current.steps) or value.status in {
                "ready",
                "queued",
            }:
                self._validate_pet_assets(db, key, value)
            updated = PetJob.model_validate(
                {
                    **current.model_dump(),
                    **value.model_dump(),
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            self._save_pet(db, updated)
            return updated

    def selected_pet(self, key: str) -> PetJob | None:
        with self.connect() as db:
            agent = self._agent(db, key)
            return (
                self._pet(db, key, agent.selected_pet_id)
                if agent.selected_pet_id
                else None
            )

    def select_pet(self, key: str, pet_id: UUID | None) -> PetJob | None:
        with self.connect(write=True) as db:
            self._agent(db, key)
            pet = self._pet(db, key, pet_id) if pet_id is not None else None
            if pet:
                if pet.status != "ready":
                    raise StorageError("Only ready pets can be selected", 409)
                self._validate_pet_assets(db, key, pet)
            db.execute(
                "UPDATE agents SET selected_pet_id = ? WHERE key = ?",
                (str(pet_id) if pet_id else None, key),
            )
            return pet

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
        *,
        kinds: list[ResourceKind] | None = None,
    ) -> list[Resource]:
        with self.connect() as db:
            self._agent(db, key)
            clauses, params = ["r.agent_key = ?", "r.deleted = ?"], [key, int(trash)]
            if kind:
                clauses.append("r.kind = ?")
                params.append(kind.value)
            if kinds is not None:
                if not kinds:
                    return []
                placeholders = ", ".join("?" for _ in kinds)
                clauses.append(f"r.kind IN ({placeholders})")
                params.extend(kind.value for kind in kinds)
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
    def _check_version(
        current: Resource | Asset | PetJob | AgentRun | ImageJob,
        expected_version: int,
    ) -> None:
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
        allowed = parent_kinds(value.kind)
        visited = {resource_id}
        while parent_id:
            if parent_id in visited:
                raise StorageError(
                    "A folder or Space cannot contain itself or one of its ancestors"
                )
            visited.add(parent_id)
            parent = self._resource(db, key, parent_id)
            if parent.kind not in allowed or parent.deleted_at:
                raise StorageError(PARENT_ERRORS[allowed])
            # Each ancestor must itself sit where its own kind is allowed.
            allowed = parent_kinds(parent.kind)
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

    def _clear_conversation_memory_snapshots(
        self, db: sqlite3.Connection, key: str
    ) -> None:
        rows = db.execute(
            "SELECT id, data FROM resources WHERE agent_key = ? AND kind = ? AND deleted = 0",
            (key, ResourceKind.CONVERSATION.value),
        ).fetchall()
        for row in rows:
            conv = resource_adapter.validate_json(row["data"])
            if (
                conv.core_memories_snapshot is not None
                or conv.core_memories_refreshed_at is not None
            ):
                cleared = conv.model_copy(
                    update={
                        "core_memories_snapshot": None,
                        "core_memories_refreshed_at": None,
                    }
                )
                db.execute(
                    "UPDATE resources SET data = ? WHERE id = ?",
                    (cleared.model_dump_json(), str(conv.id)),
                )

    def create_resource(
        self,
        key: str,
        value: ResourceInput,
        *,
        preserve_memory_snapshot: bool = False,
    ) -> Resource:
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
            if resource.kind == ResourceKind.MEMORY and not preserve_memory_snapshot:
                self._clear_conversation_memory_snapshots(db, key)
            self._activity(
                db, key, "created", "resource", str(resource.id), resource.title
            )
        return resource

    def save_chat_result(
        self, key: str, conversation_id: UUID, message_id: UUID, title: str
    ) -> Resource:
        """Keep one editable document per saved reply, including after a lost receipt."""
        document_id = uuid5(conversation_id, f"chat-result:{message_id}")
        with self.connect(write=True) as db:
            self._agent(db, key)
            conversation = self._resource(db, key, conversation_id)
            if (
                conversation.kind != ResourceKind.CONVERSATION
                or conversation.deleted_at
            ):
                raise StorageError("An active conversation is required", 409)
            message = next(
                (item for item in conversation.messages if item.id == message_id), None
            )
            if message is None:
                raise StorageError("Saved reply not found", 404)
            if message.role != "assistant" or not message.content.strip():
                raise StorageError(
                    "Choose a saved assistant reply containing text", 409
                )
            existing = db.execute(
                "SELECT data FROM resources WHERE agent_key = ? AND id = ?",
                (key, str(document_id)),
            ).fetchone()
            if existing:
                document = resource_adapter.validate_json(existing[0])
                if document.deleted_at:
                    raise StorageError(
                        "This result is in trash. Restore it in Tsuchi.", 409
                    )
                return document
            now = utc_now()
            document = resource_adapter.validate_python(
                {
                    "id": document_id,
                    "agent_key": key,
                    "kind": "space",
                    "title": title,
                    "markdown": message.content,
                    "attachment_ids": message.attachment_ids,
                    "sources": message.sources,
                    "tags": [
                        "chat-result",
                        f"conversation:{conversation_id}",
                        f"message:{message_id}",
                    ],
                    "version": 1,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            self._validate_links(db, key, document, document_id)
            self._save_resource(db, document)
            self._activity(
                db, key, "created", "resource", str(document_id), document.title
            )
            return document

    def update_resource(
        self,
        key: str,
        resource_id: UUID,
        value: ResourceInput,
        expected_version: int,
        *,
        preserve_memory_snapshot: bool = False,
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            updated = self._update_resource(
                db, key, current, value, expected_version, "updated"
            )
            if updated.kind == ResourceKind.MEMORY and not preserve_memory_snapshot:
                self._clear_conversation_memory_snapshots(db, key)
            return updated

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
        content = value.model_dump()
        if "pinned" not in value.model_fields_set:
            # Editors that predate pinning must not unpin a record.
            content["pinned"] = current.pinned
        if current.kind == ResourceKind.CONVERSATION:
            for field_name in (
                "compacted_summary",
                "compacted_through_message_id",
                "compacted_at",
                "memory_extracted_through_message_id",
                "core_memories_snapshot",
                "core_memories_refreshed_at",
            ):
                if field_name not in value.model_fields_set:
                    content[field_name] = getattr(current, field_name)
        updated = resource_adapter.validate_python(
            {
                **content,
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
        self,
        key: str,
        resource_id: UUID,
        expected_version: int,
        *,
        preserve_memory_snapshot: bool = False,
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            self._check_version(current, expected_version)
            self._active(current)
            if db.execute(
                "SELECT 1 FROM resources WHERE agent_key = ? AND parent_id = ? AND deleted = 0 LIMIT 1",
                (key, str(resource_id)),
            ).fetchone() or (
                current.kind == ResourceKind.FOLDER
                and db.execute(
                    "SELECT 1 FROM assets WHERE agent_key = ? AND deleted = 0 AND json_extract(data, '$.folder_id') = ? LIMIT 1",
                    (key, str(resource_id)),
                ).fetchone()
            ):
                raise StorageError(
                    "Move or trash this folder's contents first"
                    if current.kind == ResourceKind.FOLDER
                    else "Move or trash this Space's active children first",
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
            self._save_resource(db, updated)
            if updated.kind == ResourceKind.MEMORY and not preserve_memory_snapshot:
                self._clear_conversation_memory_snapshots(db, key)
            self._activity(
                db, key, "trashed", "resource", str(resource_id), current.title
            )
            return updated

    def restore_resource(
        self,
        key: str,
        resource_id: UUID,
        expected_version: int,
        *,
        preserve_memory_snapshot: bool = False,
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
            if updated.kind == ResourceKind.MEMORY and not preserve_memory_snapshot:
                self._clear_conversation_memory_snapshots(db, key)
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
        self,
        key: str,
        resource_id: UUID,
        version: int,
        expected_version: int,
        *,
        preserve_memory_snapshot: bool = False,
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
            # Restoring text does not change whether the record is pinned.
            value = resource_input_adapter.validate_python(
                snapshot.model_dump(exclude={*ResourceRecord.model_fields, "pinned"})
            )
            updated = self._update_resource(
                db, key, current, value, expected_version, "version_restored"
            )
            if updated.kind == ResourceKind.MEMORY and not preserve_memory_snapshot:
                self._clear_conversation_memory_snapshots(db, key)
            return updated

    def append_message(
        self, key: str, resource_id: UUID, message: Message, expected_version: int
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            if current.kind != ResourceKind.CONVERSATION:
                raise StorageError("Messages can only be added to conversations")
            if any(existing.id == message.id for existing in current.messages):
                raise StorageError("This message ID already exists", 409)
            value = resource_input_adapter.validate_python(
                {
                    **current.model_dump(exclude=set(ResourceRecord.model_fields)),
                    "messages": [*current.messages, message],
                }
            )
            return self._update_resource(
                db, key, current, value, expected_version, "message_added"
            )

    def update_conversation_compaction(
        self,
        key: str,
        resource_id: UUID,
        value: ConversationCompactionUpdate,
    ) -> Resource:
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            self._active(current)
            if current.kind != ResourceKind.CONVERSATION:
                raise StorageError("Only conversations can be compacted", 409)
            self._check_version(current, value.expected_version)
            if (
                value.require_idle
                and db.execute(
                    "SELECT 1 FROM agent_runs WHERE agent_key = ? AND conversation_id = ? "
                    "AND status IN ('queued', 'running') LIMIT 1",
                    (key, str(resource_id)),
                ).fetchone()
            ):
                raise StorageError(
                    "Conversation has active work and cannot be idle-compacted", 409
                )
            message_indices = {
                message.id: index for index, message in enumerate(current.messages)
            }
            target_index = message_indices.get(value.compacted_through_message_id)
            if target_index is None:
                raise StorageError(
                    "Compaction anchor message was not found in this conversation", 409
                )
            if current.compacted_through_message_id is not None:
                current_index = message_indices.get(
                    current.compacted_through_message_id
                )
                if current_index is not None and target_index < current_index:
                    raise StorageError(
                        "Compaction cannot move backwards to an earlier message", 409
                    )
            now = utc_now()
            updates: dict[str, object] = {
                "compacted_summary": value.compacted_summary,
                "compacted_through_message_id": value.compacted_through_message_id,
                "compacted_at": value.compacted_at or now,
                "version": current.version + 1,
                "updated_at": now,
            }
            if "core_memories_snapshot" in value.model_fields_set:
                updates["core_memories_snapshot"] = value.core_memories_snapshot
                updates["core_memories_refreshed_at"] = (
                    (value.core_memories_refreshed_at or now)
                    if value.core_memories_snapshot is not None
                    else None
                )
            updated = current.model_copy(update=updates)
            self._save_resource(db, updated)
            self._activity(
                db, key, "compacted", "resource", str(resource_id), updated.title
            )
            return updated

    def update_conversation_memory_state(
        self,
        key: str,
        resource_id: UUID,
        value: ConversationMemoryStateUpdate,
    ) -> Resource:
        """Advance memory extraction cursor and/or epoch-frozen Core Memories snapshot in place."""
        with self.connect(write=True) as db:
            current = self._resource(db, key, resource_id)
            self._active(current)
            if current.kind != ResourceKind.CONVERSATION:
                raise StorageError("Memory state applies only to conversations", 409)
            if value.expected_version is not None:
                self._check_version(current, value.expected_version)
            if (
                value.require_idle
                and db.execute(
                    "SELECT 1 FROM agent_runs WHERE agent_key = ? AND conversation_id = ? "
                    "AND status IN ('queued', 'running') LIMIT 1",
                    (key, str(resource_id)),
                ).fetchone()
            ):
                raise StorageError(
                    "Conversation has active work and cannot update memory state",
                    409,
                )
            now = utc_now()
            updates: dict[str, object] = {}
            if (
                "memory_extracted_through_message_id" in value.model_fields_set
                and value.memory_extracted_through_message_id is not None
            ):
                message_indices = {
                    message.id: index for index, message in enumerate(current.messages)
                }
                target_index = message_indices.get(
                    value.memory_extracted_through_message_id
                )
                if target_index is None:
                    raise StorageError(
                        "Memory cursor message was not found in this conversation",
                        409,
                    )
                if current.memory_extracted_through_message_id is not None:
                    current_index = message_indices.get(
                        current.memory_extracted_through_message_id
                    )
                    if current_index is not None and target_index < current_index:
                        raise StorageError(
                            "Memory cursor cannot move backwards to an earlier message",
                            409,
                        )
                updates["memory_extracted_through_message_id"] = (
                    value.memory_extracted_through_message_id
                )
            if "core_memories_snapshot" in value.model_fields_set:
                if not (
                    value.only_if_snapshot_unset
                    and current.core_memories_snapshot is not None
                ):
                    updates["core_memories_snapshot"] = value.core_memories_snapshot
                    updates["core_memories_refreshed_at"] = (
                        (value.core_memories_refreshed_at or now)
                        if value.core_memories_snapshot is not None
                        else None
                    )
            if not updates:
                return current
            updated = current.model_copy(update=updates)
            # Update internal tracking metadata in place without bumping resource
            # version so concurrent user chat sends never hit spurious 409 conflicts.
            db.execute(
                "UPDATE resources SET data = ? WHERE id = ?",
                (updated.model_dump_json(), str(resource_id)),
            )
            return updated

    @staticmethod
    def _asset(db: sqlite3.Connection, key: str, asset_id: UUID) -> Asset:
        row = db.execute(
            "SELECT data FROM assets WHERE agent_key = ? AND id = ?",
            (key, str(asset_id)),
        ).fetchone()
        if row is None:
            raise StorageError("Asset not found in this namespace", 404)
        return Asset.model_validate_json(row[0])

    def _active_folder(
        self, db: sqlite3.Connection, key: str, folder_id: UUID
    ) -> Resource | None:
        row = db.execute(
            "SELECT data FROM resources WHERE agent_key = ? AND id = ?",
            (key, str(folder_id)),
        ).fetchone()
        if row is None:
            return None
        folder = resource_adapter.validate_json(row[0])
        if folder.kind != ResourceKind.FOLDER or folder.deleted_at:
            return None
        return folder

    def _require_folder(
        self, db: sqlite3.Connection, key: str, folder_id: UUID | None
    ) -> None:
        if folder_id is not None and self._active_folder(db, key, folder_id) is None:
            raise StorageError("Folder must be an active folder in this namespace")

    def get_asset(self, key: str, asset_id: UUID) -> Asset:
        with self.connect() as db:
            self._agent(db, key)
            return self._asset(db, key, asset_id)

    def list_assets(
        self,
        key: str,
        images: bool = False,
        trash: bool = False,
        query: str = "",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Asset]:
        with self.connect() as db:
            self._agent(db, key)
            rows = db.execute(
                "SELECT data FROM assets WHERE agent_key = ? AND deleted = ? AND (? = 0 OR is_image = 1) ORDER BY updated_at DESC, id",
                (key, int(trash), int(images)),
            )
            assets = (Asset.model_validate_json(row[0]) for row in rows)
            tokens = query.casefold().split()
            matches = (
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
            )
            return list(
                islice(matches, offset, None if limit is None else offset + limit)
            )

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

    def create_asset(self, asset: Asset, image_job_id: UUID | None = None) -> Asset:
        with self.connect(write=True) as db:
            self._agent(db, asset.agent_key)
            job = None
            if image_job_id is not None:
                job = self._image_job(db, asset.agent_key, image_job_id)
                if job.upload_status not in {"pending", "completed"}:
                    raise StorageError("This image upload is closed", 409)
                if asset.id != uuid5(job.id, "image") or not asset.is_image:
                    raise StorageError(
                        "Image uploads require their reserved artwork", 409
                    )
            row = db.execute(
                "SELECT data FROM assets WHERE id = ?", (str(asset.id),)
            ).fetchone()
            if row is not None:
                current = Asset.model_validate_json(row[0])
                replay_fields = {
                    "agent_key",
                    "filename",
                    "media_type",
                    "size_bytes",
                    "sha256",
                    "is_image",
                    "title",
                    "tags",
                    "sources",
                }
                if current.deleted_at is None and all(
                    getattr(current, field) == getattr(asset, field)
                    for field in replay_fields
                ):
                    asset = current
                else:
                    raise StorageError("This asset ID is already in use", 409)
            else:
                if job is not None and job.upload_status == "completed":
                    raise StorageError(
                        "Completed image artwork cannot be recreated", 409
                    )
                self._require_folder(db, asset.agent_key, asset.folder_id)
                self._save_asset(db, asset)
                self._activity(
                    db, asset.agent_key, "uploaded", "asset", str(asset.id), asset.title
                )
            if job is not None and job.upload_status == "pending":
                self._save_image_job(
                    db,
                    job.model_copy(
                        update={
                            "upload_status": "completed",
                            "version": job.version + 1,
                            "updated_at": utc_now(),
                        }
                    ),
                )
        return asset

    def update_asset(
        self, key: str, asset_id: UUID, value: AssetMetadata, expected_version: int
    ) -> Asset:
        with self.connect(write=True) as db:
            current = self._asset(db, key, asset_id)
            self._check_version(current, expected_version)
            self._active(current)
            # Omitted pin and folder fields keep their saved values.
            kept = {"pinned", "folder_id"} - value.model_fields_set
            if "folder_id" not in kept:
                self._require_folder(db, key, value.folder_id)
            updated = Asset.model_validate(
                {
                    **current.model_dump(),
                    **value.model_dump(exclude=kept),
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
            agent = self._agent(db, key)
            if agent.selected_pet_id:
                pet = self._pet(db, key, agent.selected_pet_id)
                if asset_id in pet.asset_references():
                    db.execute(
                        "UPDATE agents SET selected_pet_id = NULL WHERE key = ?", (key,)
                    )
            self._activity(db, key, "trashed", "asset", str(asset_id), current.title)
            return updated

    def restore_asset(self, key: str, asset_id: UUID, expected_version: int) -> Asset:
        with self.connect(write=True) as db:
            current = self._asset(db, key, asset_id)
            self._check_version(current, expected_version)
            if not current.deleted_at:
                raise StorageError("This asset is not in trash", 409)
            folder_id = current.folder_id
            if folder_id and self._active_folder(db, key, folder_id) is None:
                # A file whose folder is gone comes back at the top level.
                folder_id = None
            updated = current.model_copy(
                update={
                    "folder_id": folder_id,
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
            pet_jobs=[
                PetJob.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM pet_jobs WHERE agent_key = ? ORDER BY id", (key,)
                )
            ],
            agent_runs=[
                AgentRun.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM agent_runs WHERE agent_key = ? ORDER BY id",
                    (key,),
                )
            ],
            image_jobs=[
                ImageJob.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT data FROM image_jobs WHERE agent_key = ? ORDER BY id",
                    (key,),
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

    def list_mcp_servers(self) -> list[McpServer]:
        with self.connect() as db:
            return [
                McpServer.model_validate_json(row[0])
                for row in db.execute("SELECT data FROM mcp_servers ORDER BY slug")
            ]

    def get_mcp_server(self, server_id: UUID) -> McpServer:
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM mcp_servers WHERE id = ?", (str(server_id),)
            ).fetchone()
            if not row:
                raise StorageError("MCP server not found", 404)
            return McpServer.model_validate_json(row[0])

    def create_mcp_server(self, value: McpServerInput) -> McpServer:
        with self.connect(write=True) as db:
            if db.execute(
                "SELECT 1 FROM mcp_servers WHERE slug = ?", (value.slug,)
            ).fetchone():
                raise StorageError(
                    f"An integration with identifier '{value.slug}' already exists", 409
                )
            now = utc_now()
            server = McpServer(
                id=uuid4(),
                version=1,
                created_at=now,
                updated_at=now,
                **value.model_dump(),
            )
            db.execute(
                "INSERT INTO mcp_servers (id, slug, updated_at, data) VALUES (?, ?, ?, ?)",
                (
                    str(server.id),
                    server.slug,
                    now.isoformat(),
                    server.model_dump_json(),
                ),
            )
            return server

    def update_mcp_server(self, server_id: UUID, value: McpServerUpdate) -> McpServer:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM mcp_servers WHERE id = ?", (str(server_id),)
            ).fetchone()
            if not row:
                raise StorageError("MCP server not found", 404)
            current = McpServer.model_validate_json(row[0])
            if current.version != value.expected_version:
                raise StorageError("MCP server changed; reload before saving", 409)
            if (
                value.slug != current.slug
                and db.execute(
                    "SELECT 1 FROM mcp_servers WHERE slug = ? AND id != ?",
                    (value.slug, str(server_id)),
                ).fetchone()
            ):
                raise StorageError(
                    f"An integration with identifier '{value.slug}' already exists", 409
                )
            now = utc_now()
            updated = McpServer(
                id=current.id,
                version=current.version + 1,
                created_at=current.created_at,
                updated_at=now,
                **value.model_dump(exclude={"expected_version"}),
            )
            db.execute(
                "UPDATE mcp_servers SET slug = ?, updated_at = ?, data = ? WHERE id = ?",
                (
                    updated.slug,
                    now.isoformat(),
                    updated.model_dump_json(),
                    str(server_id),
                ),
            )
            return updated

    def delete_mcp_server(self, server_id: UUID) -> None:
        with self.connect(write=True) as db:
            cursor = db.execute(
                "DELETE FROM mcp_servers WHERE id = ?", (str(server_id),)
            )
            if cursor.rowcount == 0:
                raise StorageError("MCP server not found", 404)
            db.execute(
                "DELETE FROM mcp_oauth_states WHERE server_id = ?", (str(server_id),)
            )

    def save_mcp_oauth_state(self, value: McpOAuthState) -> McpOAuthState:
        with self.connect(write=True) as db:
            now = utc_now()
            db.execute(
                "DELETE FROM mcp_oauth_states WHERE expires_at <= ?", (now.isoformat(),)
            )
            db.execute(
                "INSERT OR REPLACE INTO mcp_oauth_states (state, server_id, expires_at, data) "
                "VALUES (?, ?, ?, ?)",
                (
                    value.state,
                    str(value.server_id),
                    value.expires_at.isoformat(),
                    value.model_dump_json(),
                ),
            )
            return value

    def consume_mcp_oauth_state(self, state: str) -> McpOAuthState:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM mcp_oauth_states WHERE state = ?", (state,)
            ).fetchone()
            if not row:
                raise StorageError("Authorization session expired or not found", 404)
            db.execute("DELETE FROM mcp_oauth_states WHERE state = ?", (state,))
            saved = McpOAuthState.model_validate_json(row[0])
            if saved.expires_at <= utc_now():
                raise StorageError("Authorization session expired", 410)
            return saved

    def upsert_t3_watch(self, value: T3ThreadWatchInput) -> T3ThreadWatch:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM t3_thread_watches WHERE id = ?", (str(value.id),)
            ).fetchone()
            if row:
                return T3ThreadWatch.model_validate_json(row[0])
            now = utc_now()
            watch = T3ThreadWatch(
                **value.model_dump(),
                status="watching",
                last_pending_request_id=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            db.execute(
                "INSERT INTO t3_thread_watches (id, status, updated_at, data) VALUES (?, ?, ?, ?)",
                (str(watch.id), watch.status, now.isoformat(), watch.model_dump_json()),
            )
            return watch

    def list_t3_watches(
        self, status: Literal["watching", "delivered", "dropped"] | None = None
    ) -> list[T3ThreadWatch]:
        with self.connect() as db:
            if status is None:
                rows = db.execute(
                    "SELECT data FROM t3_thread_watches ORDER BY updated_at"
                )
            else:
                rows = db.execute(
                    "SELECT data FROM t3_thread_watches WHERE status = ? ORDER BY updated_at",
                    (status,),
                )
            return [T3ThreadWatch.model_validate_json(row[0]) for row in rows]

    def update_t3_watch(
        self, watch_id: UUID, value: T3ThreadWatchUpdate
    ) -> T3ThreadWatch:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM t3_thread_watches WHERE id = ?", (str(watch_id),)
            ).fetchone()
            if not row:
                raise StorageError("T3 thread watch not found", 404)
            current = T3ThreadWatch.model_validate_json(row[0])
            if current.version != value.expected_version:
                raise StorageError("T3 thread watch changed; reload before saving", 409)
            now = utc_now()
            updated = current.model_copy(
                update={
                    "status": value.status,
                    "run_id": value.run_id
                    if value.run_id is not None
                    else current.run_id,
                    "thread_link": value.thread_link
                    if value.thread_link is not None
                    else current.thread_link,
                    "thread_title": value.thread_title
                    if value.thread_title is not None
                    else current.thread_title,
                    "last_pending_request_id": value.last_pending_request_id,
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            db.execute(
                "UPDATE t3_thread_watches SET status = ?, updated_at = ?, data = ? WHERE id = ?",
                (
                    updated.status,
                    now.isoformat(),
                    updated.model_dump_json(),
                    str(watch_id),
                ),
            )
            return updated

    # ---- Group Chat durable storage ----

    @staticmethod
    def _group_chat_room(db: sqlite3.Connection, room_id: UUID) -> GroupChatRoom:
        row = db.execute(
            "SELECT data FROM group_chat_rooms WHERE id = ?", (str(room_id),)
        ).fetchone()
        if not row:
            raise StorageError("Group chat room not found", 404)
        return GroupChatRoom.model_validate_json(row[0])

    @staticmethod
    def _save_group_chat_room(db: sqlite3.Connection, room: GroupChatRoom) -> None:
        db.execute(
            """INSERT INTO group_chat_rooms (id, updated_at, data) VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, data=excluded.data""",
            (str(room.id), room.updated_at.isoformat(), room.model_dump_json()),
        )

    @staticmethod
    def _group_chat_participant(
        db: sqlite3.Connection, room_id: UUID, participant_id: UUID
    ) -> GroupChatParticipant:
        row = db.execute(
            "SELECT data FROM group_chat_participants WHERE room_id = ? AND id = ?",
            (str(room_id), str(participant_id)),
        ).fetchone()
        if not row:
            raise StorageError("Group chat participant not found", 404)
        return GroupChatParticipant.model_validate_json(row[0])

    @staticmethod
    def _derive_group_chat_handle(
        db: sqlite3.Connection,
        room_id: UUID,
        display_name: str,
        exclude_id: UUID | None = None,
    ) -> str:
        raw = re.sub(r"[^a-z0-9_-]+", "-", display_name.strip().lower()).strip("-_")
        if not raw or not re.match(r"^[a-z0-9]", raw):
            raw = "agent"
        base = raw[:40]
        candidate = base
        suffix = 2
        while True:
            if exclude_id is None:
                taken = db.execute(
                    "SELECT 1 FROM group_chat_participants WHERE room_id = ? AND handle = ?",
                    (str(room_id), candidate),
                ).fetchone()
            else:
                taken = db.execute(
                    "SELECT 1 FROM group_chat_participants WHERE room_id = ? AND handle = ? AND id != ?",
                    (str(room_id), candidate, str(exclude_id)),
                ).fetchone()
            if not taken:
                return candidate
            candidate = f"{base[: 47 - len(str(suffix))]}-{suffix}"
            suffix += 1

    @staticmethod
    def _revoke_participant_subscriptions(
        db: sqlite3.Connection, participant_id: UUID, reason: str, now: datetime
    ) -> None:
        rows = db.execute(
            "SELECT data FROM group_chat_event_subscriptions WHERE participant_id = ? AND status = 'active'",
            (str(participant_id),),
        ).fetchall()
        for row in rows:
            sub = GroupChatEventSubscription.model_validate_json(row[0])
            revoked = sub.model_copy(
                update={
                    "status": "revoked",
                    "last_error": reason,
                    "version": sub.version + 1,
                    "updated_at": now,
                }
            )
            db.execute(
                "UPDATE group_chat_event_subscriptions SET status = ?, updated_at = ?, data = ? WHERE id = ?",
                (
                    revoked.status,
                    now.isoformat(),
                    revoked.model_dump_json(),
                    str(revoked.id),
                ),
            )
        pending_rows = db.execute(
            "SELECT data FROM group_chat_webhook_deliveries WHERE participant_id = ? AND status = 'pending'",
            (str(participant_id),),
        ).fetchall()
        for row in pending_rows:
            delivery = GroupChatWebhookDelivery.model_validate_json(row[0])
            dropped = delivery.model_copy(
                update={
                    "status": "dropped",
                    "last_error": reason,
                    "version": delivery.version + 1,
                    "updated_at": now,
                }
            )
            db.execute(
                "UPDATE group_chat_webhook_deliveries SET status = ?, updated_at = ?, data = ? WHERE id = ?",
                (
                    dropped.status,
                    now.isoformat(),
                    dropped.model_dump_json(),
                    str(dropped.id),
                ),
            )

    def get_or_create_default_group_chat_room(self) -> GroupChatRoom:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM group_chat_rooms WHERE id = ?",
                (str(DEFAULT_GROUP_CHAT_ROOM_ID),),
            ).fetchone()
            if row:
                return GroupChatRoom.model_validate_json(row[0])
            now = utc_now()
            room = GroupChatRoom(
                id=DEFAULT_GROUP_CHAT_ROOM_ID,
                title="Group Chat",
                consecutive_agent_turns=0,
                version=1,
                created_at=now,
                updated_at=now,
            )
            self._save_group_chat_room(db, room)
            return room

    def get_group_chat_room(self, room_id: UUID) -> GroupChatRoom:
        with self.connect() as db:
            return self._group_chat_room(db, room_id)

    def update_group_chat_room(
        self, room_id: UUID, value: GroupChatRoomUpdate
    ) -> GroupChatRoom:
        with self.connect(write=True) as db:
            current = self._group_chat_room(db, room_id)
            if current.version != value.expected_version:
                raise StorageError("Group chat room changed; reload before saving", 409)
            now = utc_now()
            updated = current.model_copy(
                update={
                    "title": value.title if value.title is not None else current.title,
                    "max_consecutive_agent_turns": (
                        value.max_consecutive_agent_turns
                        if value.max_consecutive_agent_turns is not None
                        else current.max_consecutive_agent_turns
                    ),
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            self._save_group_chat_room(db, updated)
            return updated

    def list_group_chat_participants(self, room_id: UUID) -> list[GroupChatParticipant]:
        with self.connect() as db:
            self._group_chat_room(db, room_id)
            rows = db.execute(
                "SELECT data FROM group_chat_participants WHERE room_id = ? "
                "ORDER BY CASE kind WHEN 'kadan_agent' THEN 0 ELSE 1 END, handle",
                (str(room_id),),
            ).fetchall()
            return [GroupChatParticipant.model_validate_json(row[0]) for row in rows]

    def create_group_chat_participant(
        self, room_id: UUID, value: GroupChatParticipantInput
    ) -> GroupChatParticipant:
        with self.connect(write=True) as db:
            self._group_chat_room(db, room_id)
            now = utc_now()
            if value.kind == "kadan_agent":
                existing_rows = db.execute(
                    "SELECT data FROM group_chat_participants WHERE room_id = ? AND kind = 'kadan_agent'",
                    (str(room_id),),
                ).fetchall()
                for row in existing_rows:
                    existing = GroupChatParticipant.model_validate_json(row[0])
                    if (
                        existing.agent_key == value.agent_key
                        and existing.revoked_at is None
                    ):
                        if existing.display_name != value.display_name or (
                            value.description
                            and existing.description != value.description
                        ):
                            updated = existing.model_copy(
                                update={
                                    "display_name": value.display_name,
                                    "description": value.description
                                    or existing.description,
                                    "version": existing.version + 1,
                                    "updated_at": now,
                                }
                            )
                            db.execute(
                                "UPDATE group_chat_participants SET updated_at = ?, data = ? WHERE id = ?",
                                (
                                    now.isoformat(),
                                    updated.model_dump_json(),
                                    str(updated.id),
                                ),
                            )
                            return updated
                        return existing

            if (
                value.token_hash is not None
                and db.execute(
                    "SELECT 1 FROM group_chat_participants WHERE token_hash = ?",
                    (value.token_hash,),
                ).fetchone()
            ):
                raise StorageError("This participant token is already in use", 409)

            handle = value.handle
            if handle is not None:
                if db.execute(
                    "SELECT 1 FROM group_chat_participants WHERE room_id = ? AND handle = ?",
                    (str(room_id), handle),
                ).fetchone():
                    raise StorageError(
                        f"A participant with handle '@{handle}' already exists", 409
                    )
            else:
                handle = self._derive_group_chat_handle(db, room_id, value.display_name)

            participant = GroupChatParticipant(
                id=value.id or uuid4(),
                room_id=room_id,
                handle=handle,
                display_name=value.display_name,
                kind=value.kind,
                agent_key=value.agent_key,
                description=value.description,
                enabled=value.enabled,
                revoked_at=None,
                last_seen_at=None,
                cursor_seq=0,
                version=1,
                created_at=now,
                updated_at=now,
            )
            db.execute(
                """INSERT INTO group_chat_participants
                (id, room_id, handle, kind, token_hash, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(participant.id),
                    str(room_id),
                    participant.handle,
                    participant.kind,
                    value.token_hash,
                    now.isoformat(),
                    participant.model_dump_json(),
                ),
            )
            return participant

    def update_group_chat_participant(
        self,
        room_id: UUID,
        participant_id: UUID,
        value: GroupChatParticipantUpdate,
    ) -> GroupChatParticipant:
        with self.connect(write=True) as db:
            self._group_chat_room(db, room_id)
            current = self._group_chat_participant(db, room_id, participant_id)
            if current.version != value.expected_version:
                raise StorageError("Participant changed; reload before saving", 409)
            if current.revoked_at is not None:
                raise StorageError("Revoked participants cannot be modified", 409)
            if value.revoke and current.kind == "kadan_agent":
                raise StorageError("Kadan's built-in agent cannot be revoked", 409)
            if value.token_hash is not None and current.kind != "external_agent":
                raise StorageError("Only external agents use bearer tokens", 409)

            now = utc_now()
            next_handle = current.handle
            if value.handle is not None and value.handle != current.handle:
                if db.execute(
                    "SELECT 1 FROM group_chat_participants WHERE room_id = ? AND handle = ? AND id != ?",
                    (str(room_id), value.handle, str(participant_id)),
                ).fetchone():
                    raise StorageError(
                        f"A participant with handle '@{value.handle}' already exists",
                        409,
                    )
                next_handle = value.handle

            if (
                value.token_hash is not None
                and db.execute(
                    "SELECT 1 FROM group_chat_participants WHERE token_hash = ? AND id != ?",
                    (value.token_hash, str(participant_id)),
                ).fetchone()
            ):
                raise StorageError("This participant token is already in use", 409)

            next_enabled = (
                False
                if value.revoke
                else (value.enabled if value.enabled is not None else current.enabled)
            )
            next_revoked_at = now if value.revoke else current.revoked_at
            updated = current.model_copy(
                update={
                    "display_name": (
                        value.display_name
                        if value.display_name is not None
                        else current.display_name
                    ),
                    "handle": next_handle,
                    "description": (
                        value.description
                        if value.description is not None
                        else current.description
                    ),
                    "enabled": next_enabled,
                    "revoked_at": next_revoked_at,
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            if value.revoke:
                db.execute(
                    "UPDATE group_chat_participants SET handle = ?, token_hash = NULL, updated_at = ?, data = ? WHERE id = ?",
                    (
                        updated.handle,
                        now.isoformat(),
                        updated.model_dump_json(),
                        str(participant_id),
                    ),
                )
                self._revoke_participant_subscriptions(
                    db, participant_id, "Participant was revoked", now
                )
            elif value.token_hash is not None:
                db.execute(
                    "UPDATE group_chat_participants SET handle = ?, token_hash = ?, updated_at = ?, data = ? WHERE id = ?",
                    (
                        updated.handle,
                        value.token_hash,
                        now.isoformat(),
                        updated.model_dump_json(),
                        str(participant_id),
                    ),
                )
            else:
                db.execute(
                    "UPDATE group_chat_participants SET handle = ?, updated_at = ?, data = ? WHERE id = ?",
                    (
                        updated.handle,
                        now.isoformat(),
                        updated.model_dump_json(),
                        str(participant_id),
                    ),
                )
            if not value.revoke and current.enabled and not updated.enabled:
                self._revoke_participant_subscriptions(
                    db, participant_id, "Participant was paused", now
                )
            return updated

    def touch_group_chat_participant(
        self,
        room_id: UUID,
        participant_id: UUID,
        value: GroupChatParticipantTouch,
    ) -> GroupChatParticipant:
        with self.connect(write=True) as db:
            self._group_chat_room(db, room_id)
            current = self._group_chat_participant(db, room_id, participant_id)
            updates: dict[str, object] = {}
            if value.last_seen_at is not None:
                updates["last_seen_at"] = value.last_seen_at
            if value.cursor_seq is not None and value.cursor_seq > current.cursor_seq:
                updates["cursor_seq"] = value.cursor_seq
            if not updates:
                return current
            updated = current.model_copy(update=updates)
            db.execute(
                "UPDATE group_chat_participants SET data = ? WHERE id = ?",
                (updated.model_dump_json(), str(participant_id)),
            )
            return updated

    def authenticate_group_chat_token(self, token_hash: str) -> GroupChatParticipant:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM group_chat_participants WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if not row:
                raise StorageError(
                    "Invalid group chat bearer token", 401, code="invalid_token"
                )
            participant = GroupChatParticipant.model_validate_json(row[0])
            if participant.revoked_at is not None:
                raise StorageError(
                    "This agent's access has been revoked",
                    403,
                    code="participant_revoked",
                )
            if not participant.enabled:
                raise StorageError(
                    "This agent is currently paused", 403, code="participant_paused"
                )
            now = utc_now()
            updated = participant.model_copy(update={"last_seen_at": now})
            db.execute(
                "UPDATE group_chat_participants SET data = ? WHERE id = ?",
                (updated.model_dump_json(), str(participant.id)),
            )
            return updated

    def list_group_chat_messages(
        self,
        room_id: UUID,
        *,
        after_seq: int | None = None,
        before_seq: int | None = None,
        limit: int = 50,
    ) -> GroupChatMessagesPage:
        with self.connect() as db:
            room = self._group_chat_room(db, room_id)
            latest_row = db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM group_chat_messages WHERE room_id = ?",
                (str(room_id),),
            ).fetchone()
            latest_seq = int(latest_row[0]) if latest_row else 0
            fetch_limit = limit + 1
            if after_seq is not None and before_seq is not None:
                rows = db.execute(
                    "SELECT data FROM group_chat_messages WHERE room_id = ? AND seq > ? AND seq < ? ORDER BY seq ASC LIMIT ?",
                    (str(room_id), after_seq, before_seq, fetch_limit),
                ).fetchall()
                has_more = len(rows) > limit
                messages = [
                    GroupChatMessage.model_validate_json(row[0]) for row in rows[:limit]
                ]
            elif after_seq is not None:
                rows = db.execute(
                    "SELECT data FROM group_chat_messages WHERE room_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                    (str(room_id), after_seq, fetch_limit),
                ).fetchall()
                has_more = len(rows) > limit
                messages = [
                    GroupChatMessage.model_validate_json(row[0]) for row in rows[:limit]
                ]
            elif before_seq is not None:
                rows = db.execute(
                    "SELECT data FROM group_chat_messages WHERE room_id = ? AND seq < ? ORDER BY seq DESC LIMIT ?",
                    (str(room_id), before_seq, fetch_limit),
                ).fetchall()
                has_more = len(rows) > limit
                messages = [
                    GroupChatMessage.model_validate_json(row[0])
                    for row in reversed(rows[:limit])
                ]
            else:
                rows = db.execute(
                    "SELECT data FROM group_chat_messages WHERE room_id = ? ORDER BY seq DESC LIMIT ?",
                    (str(room_id), fetch_limit),
                ).fetchall()
                has_more = len(rows) > limit
                messages = [
                    GroupChatMessage.model_validate_json(row[0])
                    for row in reversed(rows[:limit])
                ]
            return GroupChatMessagesPage(
                room=room,
                messages=messages,
                latest_seq=latest_seq,
                has_more=has_more,
            )

    def create_group_chat_message(
        self, room_id: UUID, value: GroupChatMessageInput
    ) -> GroupChatMessageWriteResult:
        with self.connect(write=True) as db:
            room = self._group_chat_room(db, room_id)
            sender_scope = (
                "user" if value.sender_kind == "user" else str(value.participant_id)
            )
            if value.client_request_id is not None:
                existing_row = db.execute(
                    "SELECT data FROM group_chat_messages WHERE room_id = ? AND sender_scope = ? AND client_request_id = ?",
                    (str(room_id), sender_scope, value.client_request_id),
                ).fetchone()
                if existing_row is not None:
                    existing = GroupChatMessage.model_validate_json(existing_row[0])
                    if (
                        existing.content != value.content
                        or existing.reply_to_message_id != value.reply_to_message_id
                        or existing.sender_kind != value.sender_kind
                        or existing.participant_id != value.participant_id
                    ):
                        raise StorageError(
                            "This client_request_id was already used for a different message",
                            409,
                            code="idempotency_conflict",
                        )
                    return GroupChatMessageWriteResult(
                        message=existing,
                        room=room,
                        effective_handoff_to_participant_ids=list(
                            existing.handoff_to_participant_ids
                        ),
                        turn_cap_reached=room.consecutive_agent_turns
                        >= room.max_consecutive_agent_turns,
                        handoff_suppressed=existing.handoff_suppressed,
                        idempotent_replay=True,
                        enqueued_delivery_ids=[],
                    )

            sender_participant: GroupChatParticipant | None = None
            if value.sender_kind != "user":
                assert value.participant_id is not None
                sender_participant = self._group_chat_participant(
                    db, room_id, value.participant_id
                )
                if sender_participant.kind != value.sender_kind:
                    raise StorageError(
                        "Sender kind does not match participant record", 409
                    )
                if sender_participant.revoked_at is not None:
                    raise StorageError(
                        "This agent's access has been revoked",
                        403,
                        code="participant_revoked",
                    )
                if not sender_participant.enabled:
                    raise StorageError(
                        "This agent is currently paused",
                        403,
                        code="participant_paused",
                    )
                if room.consecutive_agent_turns >= room.max_consecutive_agent_turns:
                    raise StorageError(
                        "Agents reached the consecutive turn limit. Waiting for the user to send a message.",
                        409,
                        code="agent_turn_cap_reached",
                    )

            if (
                value.reply_to_message_id is not None
                and not db.execute(
                    "SELECT 1 FROM group_chat_messages WHERE room_id = ? AND id = ?",
                    (str(room_id), str(value.reply_to_message_id)),
                ).fetchone()
            ):
                raise StorageError("Reply target message was not found", 404)

            requested_handoff = list(dict.fromkeys(value.handoff_to_participant_ids))
            if (
                value.participant_id is not None
                and value.participant_id in requested_handoff
            ):
                raise StorageError(
                    "An agent cannot hand off a turn to itself",
                    422,
                    code="self_handoff_not_allowed",
                )
            for target_id in requested_handoff:
                target = self._group_chat_participant(db, room_id, target_id)
                if target.revoked_at is not None or not target.enabled:
                    raise StorageError(
                        f"Participant '@{target.handle}' is not active",
                        409,
                        code="participant_inactive",
                    )

            new_turns = (
                0 if value.sender_kind == "user" else room.consecutive_agent_turns + 1
            )
            turn_cap_reached = new_turns >= room.max_consecutive_agent_turns
            if turn_cap_reached:
                effective_handoff: list[UUID] = []
                handoff_suppressed = bool(requested_handoff)
            else:
                effective_handoff = requested_handoff
                handoff_suppressed = False

            next_seq_row = db.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM group_chat_messages WHERE room_id = ?",
                (str(room_id),),
            ).fetchone()
            seq = int(next_seq_row[0])
            now = utc_now()
            message = GroupChatMessage(
                id=value.id or uuid4(),
                room_id=room_id,
                seq=seq,
                sender_kind=value.sender_kind,
                participant_id=value.participant_id,
                sender_name=value.sender_name,
                content=value.content,
                handoff_to_participant_ids=effective_handoff,
                handoff_suppressed=handoff_suppressed,
                reply_to_message_id=value.reply_to_message_id,
                client_request_id=value.client_request_id,
                created_at=now,
            )
            db.execute(
                """INSERT INTO group_chat_messages
                (id, room_id, seq, sender_scope, client_request_id, created_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(message.id),
                    str(room_id),
                    seq,
                    sender_scope,
                    value.client_request_id,
                    now.isoformat(),
                    message.model_dump_json(),
                ),
            )
            updated_room = room.model_copy(
                update={
                    "consecutive_agent_turns": new_turns,
                    "version": room.version + 1,
                    "updated_at": now,
                }
            )
            self._save_group_chat_room(db, updated_room)

            if sender_participant is not None:
                touched = sender_participant.model_copy(update={"last_seen_at": now})
                db.execute(
                    "UPDATE group_chat_participants SET data = ? WHERE id = ?",
                    (touched.model_dump_json(), str(sender_participant.id)),
                )

            enqueued_delivery_ids: list[UUID] = []
            if effective_handoff:
                recent_rows = db.execute(
                    "SELECT data FROM group_chat_messages WHERE room_id = ? AND seq <= ? ORDER BY seq DESC LIMIT 10",
                    (str(room_id), seq),
                ).fetchall()
                recent_messages = [
                    GroupChatMessage.model_validate_json(row[0])
                    for row in reversed(recent_rows)
                ]
                recent_context = [
                    {
                        "seq": item.seq,
                        "id": str(item.id),
                        "sender_kind": item.sender_kind,
                        "sender_name": item.sender_name,
                        "content": item.content[:2000],
                        "created_at": item.created_at.isoformat(),
                    }
                    for item in recent_messages
                ]
                for target_id in effective_handoff:
                    sub_rows = db.execute(
                        "SELECT data FROM group_chat_event_subscriptions "
                        "WHERE room_id = ? AND participant_id = ? AND event_name = 'group_chat.turn_requested' AND status = 'active'",
                        (str(room_id), str(target_id)),
                    ).fetchall()
                    for sub_row in sub_rows:
                        sub = GroupChatEventSubscription.model_validate_json(sub_row[0])
                        if sub.expires_at <= now:
                            continue
                        room_filter = sub.filters.get("room_id")
                        if room_filter is not None and str(room_filter) != str(room_id):
                            continue
                        event_id = uuid5(sub.id, f"group-chat:{room_id}:{seq}")
                        delivery = GroupChatWebhookDelivery(
                            id=event_id,
                            subscription_id=sub.id,
                            room_id=room_id,
                            participant_id=target_id,
                            seq=seq,
                            message_id=message.id,
                            payload={
                                "room_id": str(room_id),
                                "room_title": updated_room.title,
                                "seq": seq,
                                "message": {
                                    "id": str(message.id),
                                    "seq": seq,
                                    "sender_kind": message.sender_kind,
                                    "participant_id": (
                                        str(message.participant_id)
                                        if message.participant_id
                                        else None
                                    ),
                                    "sender_name": message.sender_name,
                                    "content": message.content[:8000],
                                    "handoff_to_participant_ids": [
                                        str(pid) for pid in effective_handoff
                                    ],
                                    "reply_to_message_id": (
                                        str(message.reply_to_message_id)
                                        if message.reply_to_message_id
                                        else None
                                    ),
                                    "created_at": message.created_at.isoformat(),
                                },
                                "recent_messages": recent_context,
                                "consecutive_agent_turns": updated_room.consecutive_agent_turns,
                                "max_consecutive_agent_turns": updated_room.max_consecutive_agent_turns,
                                "turn_cap_reached": turn_cap_reached,
                            },
                            attempts=0,
                            next_attempt_at=now,
                            status="pending",
                            last_error=None,
                            version=1,
                            created_at=now,
                            updated_at=now,
                        )
                        cursor = db.execute(
                            """INSERT OR IGNORE INTO group_chat_webhook_deliveries
                            (id, subscription_id, room_id, participant_id, seq, status, next_attempt_at, updated_at, data)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                str(delivery.id),
                                str(delivery.subscription_id),
                                str(room_id),
                                str(target_id),
                                seq,
                                delivery.status,
                                now.isoformat(),
                                now.isoformat(),
                                delivery.model_dump_json(),
                            ),
                        )
                        if cursor.rowcount > 0:
                            enqueued_delivery_ids.append(delivery.id)

            return GroupChatMessageWriteResult(
                message=message,
                room=updated_room,
                effective_handoff_to_participant_ids=effective_handoff,
                turn_cap_reached=turn_cap_reached,
                handoff_suppressed=handoff_suppressed,
                idempotent_replay=False,
                enqueued_delivery_ids=enqueued_delivery_ids,
            )

    def list_group_chat_subscriptions(
        self,
        room_id: UUID,
        *,
        participant_id: UUID | None = None,
        status: GroupChatSubscriptionStatus | None = None,
    ) -> list[GroupChatEventSubscription]:
        with self.connect() as db:
            self._group_chat_room(db, room_id)
            clauses = ["room_id = ?"]
            params: list[object] = [str(room_id)]
            if participant_id is not None:
                clauses.append("participant_id = ?")
                params.append(str(participant_id))
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            rows = db.execute(
                f"SELECT data FROM group_chat_event_subscriptions WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC",
                tuple(params),
            ).fetchall()
            return [
                GroupChatEventSubscription.model_validate_json(row[0]) for row in rows
            ]

    def get_group_chat_subscription(
        self, subscription_id: UUID
    ) -> GroupChatEventSubscription:
        with self.connect() as db:
            row = db.execute(
                "SELECT data FROM group_chat_event_subscriptions WHERE id = ?",
                (str(subscription_id),),
            ).fetchone()
            if not row:
                raise StorageError("Group chat event subscription not found", 404)
            return GroupChatEventSubscription.model_validate_json(row[0])

    def create_group_chat_subscription(
        self, room_id: UUID, value: GroupChatEventSubscriptionInput
    ) -> GroupChatEventSubscription:
        with self.connect(write=True) as db:
            self._group_chat_room(db, room_id)
            participant = self._group_chat_participant(
                db, room_id, value.participant_id
            )
            if participant.kind != "external_agent":
                raise StorageError(
                    "Only external agents can register webhook event subscriptions", 409
                )
            if participant.revoked_at is not None or not participant.enabled:
                raise StorageError(
                    "Paused or revoked participants cannot subscribe to events", 403
                )
            now = utc_now()
            sub = GroupChatEventSubscription(
                id=value.id or uuid4(),
                room_id=room_id,
                participant_id=value.participant_id,
                event_name=value.event_name,
                filters=value.filters,
                callback_url=value.callback_url,
                signing_secret=value.signing_secret,
                status="active",
                expires_at=value.expires_at,
                last_delivered_seq=0,
                last_error=None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            db.execute(
                """INSERT INTO group_chat_event_subscriptions
                (id, room_id, participant_id, event_name, status, expires_at, updated_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(sub.id),
                    str(room_id),
                    str(sub.participant_id),
                    sub.event_name,
                    sub.status,
                    sub.expires_at.isoformat(),
                    now.isoformat(),
                    sub.model_dump_json(),
                ),
            )
            return sub

    def update_group_chat_subscription(
        self, subscription_id: UUID, value: GroupChatEventSubscriptionUpdate
    ) -> GroupChatEventSubscription:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM group_chat_event_subscriptions WHERE id = ?",
                (str(subscription_id),),
            ).fetchone()
            if not row:
                raise StorageError("Group chat event subscription not found", 404)
            current = GroupChatEventSubscription.model_validate_json(row[0])
            if current.version != value.expected_version:
                raise StorageError(
                    "Group chat event subscription changed; reload before saving", 409
                )
            now = utc_now()
            updates: dict[str, object] = {
                "version": current.version + 1,
                "updated_at": now,
            }
            if value.status is not None:
                updates["status"] = value.status
            if value.expires_at is not None:
                updates["expires_at"] = value.expires_at
            if value.last_delivered_seq is not None:
                updates["last_delivered_seq"] = max(
                    current.last_delivered_seq, value.last_delivered_seq
                )
            if "last_error" in value.model_fields_set:
                updates["last_error"] = value.last_error
            updated = current.model_copy(update=updates)
            db.execute(
                "UPDATE group_chat_event_subscriptions SET status = ?, expires_at = ?, updated_at = ?, data = ? WHERE id = ?",
                (
                    updated.status,
                    updated.expires_at.isoformat(),
                    now.isoformat(),
                    updated.model_dump_json(),
                    str(subscription_id),
                ),
            )
            return updated

    def revoke_group_chat_subscription(
        self, subscription_id: UUID, *, participant_id: UUID | None = None
    ) -> GroupChatEventSubscription | None:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM group_chat_event_subscriptions WHERE id = ?",
                (str(subscription_id),),
            ).fetchone()
            if not row:
                return None
            current = GroupChatEventSubscription.model_validate_json(row[0])
            if participant_id is not None and current.participant_id != participant_id:
                raise StorageError("Subscription belongs to another participant", 403)
            now = utc_now()
            if current.status != "revoked":
                updated = current.model_copy(
                    update={
                        "status": "revoked",
                        "version": current.version + 1,
                        "updated_at": now,
                    }
                )
                db.execute(
                    "UPDATE group_chat_event_subscriptions SET status = ?, updated_at = ?, data = ? WHERE id = ?",
                    (
                        updated.status,
                        now.isoformat(),
                        updated.model_dump_json(),
                        str(subscription_id),
                    ),
                )
            else:
                updated = current
            pending_rows = db.execute(
                "SELECT data FROM group_chat_webhook_deliveries WHERE subscription_id = ? AND status = 'pending'",
                (str(subscription_id),),
            ).fetchall()
            for prow in pending_rows:
                delivery = GroupChatWebhookDelivery.model_validate_json(prow[0])
                dropped = delivery.model_copy(
                    update={
                        "status": "dropped",
                        "last_error": "Subscription unsubscribed",
                        "version": delivery.version + 1,
                        "updated_at": now,
                    }
                )
                db.execute(
                    "UPDATE group_chat_webhook_deliveries SET status = ?, updated_at = ?, data = ? WHERE id = ?",
                    (
                        dropped.status,
                        now.isoformat(),
                        dropped.model_dump_json(),
                        str(dropped.id),
                    ),
                )
            return updated

    def list_group_chat_webhook_deliveries(
        self,
        *,
        status: GroupChatDeliveryStatus | None = None,
        due_before: datetime | None = None,
        room_id: UUID | None = None,
        participant_id: UUID | None = None,
        limit: int = 100,
    ) -> list[GroupChatWebhookDelivery]:
        with self.connect() as db:
            clauses: list[str] = []
            params: list[object] = []
            if status is not None:
                clauses.append("status = ?")
                params.append(status)
            if due_before is not None:
                clauses.append("next_attempt_at <= ?")
                params.append(due_before.astimezone(UTC).isoformat())
            if room_id is not None:
                clauses.append("room_id = ?")
                params.append(str(room_id))
            if participant_id is not None:
                clauses.append("participant_id = ?")
                params.append(str(participant_id))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)
            rows = db.execute(
                f"SELECT data FROM group_chat_webhook_deliveries {where} ORDER BY next_attempt_at ASC, seq ASC LIMIT ?",
                tuple(params),
            ).fetchall()
            return [
                GroupChatWebhookDelivery.model_validate_json(row[0]) for row in rows
            ]

    def update_group_chat_webhook_delivery(
        self, delivery_id: UUID, value: GroupChatWebhookDeliveryUpdate
    ) -> GroupChatWebhookDelivery:
        with self.connect(write=True) as db:
            row = db.execute(
                "SELECT data FROM group_chat_webhook_deliveries WHERE id = ?",
                (str(delivery_id),),
            ).fetchone()
            if not row:
                raise StorageError("Webhook delivery not found", 404)
            current = GroupChatWebhookDelivery.model_validate_json(row[0])
            if current.version != value.expected_version:
                raise StorageError(
                    "Webhook delivery changed; reload before saving", 409
                )
            now = utc_now()
            updated = current.model_copy(
                update={
                    "status": value.status,
                    "attempts": value.attempts,
                    "next_attempt_at": (
                        value.next_attempt_at
                        if value.next_attempt_at is not None
                        else current.next_attempt_at
                    ),
                    "last_error": value.last_error,
                    "version": current.version + 1,
                    "updated_at": now,
                }
            )
            db.execute(
                "UPDATE group_chat_webhook_deliveries SET status = ?, next_attempt_at = ?, updated_at = ?, data = ? WHERE id = ?",
                (
                    updated.status,
                    updated.next_attempt_at.isoformat(),
                    now.isoformat(),
                    updated.model_dump_json(),
                    str(delivery_id),
                ),
            )
            if updated.status == "delivered":
                sub_row = db.execute(
                    "SELECT data FROM group_chat_event_subscriptions WHERE id = ?",
                    (str(updated.subscription_id),),
                ).fetchone()
                if sub_row:
                    sub = GroupChatEventSubscription.model_validate_json(sub_row[0])
                    sub_updated = sub.model_copy(
                        update={
                            "last_delivered_seq": max(
                                sub.last_delivered_seq, updated.seq
                            ),
                            "last_error": None,
                            "version": sub.version + 1,
                            "updated_at": now,
                        }
                    )
                    db.execute(
                        "UPDATE group_chat_event_subscriptions SET updated_at = ?, data = ? WHERE id = ?",
                        (
                            now.isoformat(),
                            sub_updated.model_dump_json(),
                            str(sub.id),
                        ),
                    )
            return updated
