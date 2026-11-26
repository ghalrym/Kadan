"""Convert the original runner journal inside the storage upgrade transaction."""

import json
import re
import sqlite3

from .models import (
    MigratedMarker,
    RunCall,
    RunMessage,
    RunObservation,
    RunResult,
    ToolCommand,
    TranscriptEntry,
    UnrecoveredRecord,
)


def _command(text: str) -> ToolCommand:
    value = json.loads(text)
    name = value.pop("tool")
    if name == "computer" and "operation" in value:
        name = value.pop("operation").replace(".", "_")
    return ToolCommand(tool=name, arguments=value.get("arguments", value))


def _action(row: dict, index: int, previous: TranscriptEntry | None) -> TranscriptEntry:
    role, text = row["role"], row["content"]
    if role == "assistant":
        return RunCall(call_id=f"legacy_{index}", command=_command(text))
    if (
        role == "user"
        and text.startswith("Tool result: ")
        and isinstance(previous, RunCall)
    ):
        return RunResult(
            call_id=previous.call_id,
            data=json.loads(text.removeprefix("Tool result: ")),
        )
    if role == "user" and text.startswith("Rejected tool arguments: "):
        rejection = json.loads(text.removeprefix("Rejected tool arguments: "))
        return RunObservation(role="rejection", name=rejection["tool"], data=rejection)
    # Early runners recorded a generic malformed-generation retry with no call.
    # It carries no restorable arguments or command and cannot be replayed.
    if text.startswith(
        "Your response was not a valid command. Return exactly one JSON object"
    ):
        return MigratedMarker()
    raise ValueError("Unrecognized legacy action journal row")


def migrate_run(record: dict, conversation: dict) -> list[TranscriptEntry]:
    try:
        return _run_entries(record, conversation)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        # One damaged historical journal must not prevent unrelated work. Its
        # original rows remain inspectable and an active run fails before replay.
        return [UnrecoveredRecord(original=row) for row in record["transcript"]]


def _run_entries(record: dict, conversation: dict) -> list[TranscriptEntry]:
    rows = record["transcript"]
    target = record.get("trigger_message_id")
    event = record.get("trigger_event")
    if record.get("trigger_job_id"):
        trigger = "Image job event: "
    elif event:
        trigger = {
            "codex": "Codex thread event: ",
            "computer": "Computer task event: ",
            "t3": "T3 Code thread event: ",
        }[event["kind"]]
    else:
        trigger = f"Respond to user message {target} in the conversation above. Other user messages have their own queued runs."
    boundary = next(
        (
            i
            for i, row in enumerate(rows)
            if row["role"] == "user" and row["content"].startswith(trigger)
        ),
        None,
    )
    if boundary is None:
        # Only the canonical history/trigger journal proves which rows were
        # operations. A historical message containing command JSON is still text.
        return [UnrecoveredRecord(original=row) for row in rows]
    originals = {message["id"]: message for message in conversation.get("messages", [])}
    result: list[TranscriptEntry] = []
    past_target = False
    for index, row in enumerate(rows):
        role, text = row["role"], row["content"]
        if index > boundary:
            item = _action(row, index, result[-1])
        elif index == boundary:
            if target or (not record.get("trigger_job_id") and not event):
                item = MigratedMarker()
            else:
                if event:
                    name = {
                        "computer": "computer_task_event",
                        "codex": "codex_thread_event",
                        "t3": "t3_thread_event",
                    }[event["kind"]]
                    # The durable event is authoritative and can be plain text.
                    try:
                        detail = json.loads(event["content"])
                    except ValueError:
                        detail = event["content"]
                    value = detail if isinstance(detail, dict) else {"detail": detail}
                else:
                    name = "image_job_event"
                    value, _ = json.JSONDecoder().raw_decode(text.removeprefix(trigger))
                item = RunObservation(name=name, data=value)
        elif past_target:
            item = MigratedMarker()
        else:
            match = re.match(r"^\[message ([\da-f-]{36})\] ", text)
            if role != "system" and match:
                ident = match[1]
                body = text[match.end() :]
                original = originals.get(ident, {})
                attachments = original.get("attachment_ids", [])
                if attachments:
                    suffix = " [Attached assets: " + ", ".join(attachments) + "]"
                    if body == original.get("content", "") + suffix:
                        body = original["content"]
                    else:
                        attachments = []
                item = RunMessage(role=role, content=body, attachment_ids=attachments)
                past_target = ident == target and role == "user"
            else:
                item = RunMessage(role=role, content=text)
        result.append(item)
    return result


def upgrade_transcripts(db: sqlite3.Connection) -> None:
    if db.execute("PRAGMA user_version").fetchone()[0] >= 1:
        return
    run_keys = db.execute(
        "SELECT id, conversation_id FROM agent_runs WHERE json_array_length(data, '$.transcript') > 0 ORDER BY conversation_id"
    ).fetchall()
    last_conversation = None
    conversation = {}
    for key, conversation_id in run_keys:
        if conversation_id != last_conversation:
            row = db.execute(
                "SELECT data FROM resources WHERE id=?", (conversation_id,)
            ).fetchone()
            conversation = json.loads(row[0]) if row else {}
            last_conversation = conversation_id
        raw = db.execute("SELECT data FROM agent_runs WHERE id=?", (key,)).fetchone()[0]
        record = json.loads(raw)
        entries = migrate_run(record, conversation)
        record["transcript"] = [entry.model_dump(mode="json") for entry in entries]
        db.execute("UPDATE agent_runs SET data=? WHERE id=?", (json.dumps(record), key))
    db.execute("PRAGMA user_version=1")
