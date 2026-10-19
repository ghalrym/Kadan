import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4, uuid5

from kadan_tracing import carrier, traced

from .config import StorageError
from .models import (
    Activity,
    Agent,
    AgentInput,
    AgentRun,
    AgentRunState,
    Asset,
    ExportManifest,
    ExportRevision,
    ImageJob,
    ImageJobState,
    Message,
    Metadata,
    PetJob,
    PetManifest,
    PetRejection,
    PetState,
    Resource,
    ResourceInput,
    ResourceKind,
    ResourceRecord,
    ResourceVersion,
    Runner,
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
            """)
            # Serialize the additive migration for existing storage volumes.
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(agents)")}
            if "selected_pet_id" not in columns:
                db.execute("ALTER TABLE agents ADD COLUMN selected_pet_id TEXT")
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
                run.model_dump_json(),
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
    ) -> AgentRun:
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
            self._append_run_message(db, key, conversation, message)
            now = utc_now()
            run = AgentRun(
                id=message.id,
                trace_parent=carrier().get("traceparent"),
                agent_key=key,
                conversation_id=conversation_id,
                trigger_message_id=message.id,
                persona_id=persona_id,
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
                        "SELECT data FROM agent_runs WHERE status IN ('queued', 'running')"
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
                if value.transcript[: len(current.transcript)] != current.transcript:
                    raise StorageError("Persisted agent transcript is immutable", 409)
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
            if value.status == "completed":
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

    def create_pet(self, key: str, pet_id: UUID, prompt: str) -> PetJob:
        now = utc_now()
        pet = PetJob(
            id=pet_id,
            trace_parent=carrier().get("traceparent"),
            agent_key=key,
            prompt=prompt,
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
                if current.agent_key == key and current.prompt == prompt:
                    return current
                raise StorageError("This pet job ID is already in use", 409)
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
            or "base" in frame_names
        ):
            raise StorageError(
                "Choose distinct completed frames to repair; the base cannot be replaced"
            )
        with self.connect(write=True) as db:
            self._agent(db, key)
            source = self._pet(db, key, pet_id)
            self._check_version(source, expected_version)
            if source.status in {"queued", "running", "cancelling"}:
                raise StorageError("Stop pet creation before repairing frames", 409)
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
            now = utc_now()
            steps = [step for step in source.steps if step.name not in rejected_names]
            revision = PetJob(
                id=revision_id,
                trace_parent=carrier().get("traceparent"),
                agent_key=key,
                prompt=source.prompt,
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
            updated_steps = {step.name: step for step in value.steps}
            if any(updated_steps.get(step.name) != step for step in current.steps):
                raise StorageError("Completed pet steps are immutable", 409)
            if (
                value.origin != current.origin
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
    def _check_version(
        current: Resource | Asset | PetJob | AgentRun | ImageJob, expected_version: int
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
                    return current
                raise StorageError("This asset ID is already in use", 409)
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
