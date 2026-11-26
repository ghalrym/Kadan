"""Migration of saved journal values without running services or a model."""

import json
from uuid import uuid4, uuid5

from tsuchi.transcript_migration import migrate_run


def old(role, content):
    return {"role": role, "content": content}


def request_trigger(ident):
    return old(
        "user",
        f"Respond to user message {ident} in the conversation above. Other user messages have their own queued runs.",
    )


def test_authored_history_is_never_parsed_as_an_action_and_action_ids_survive():
    run_id, older, target, later = (uuid4() for _ in range(4))
    quoted_command = '{"tool":"finish","arguments":{}}'
    user_text = "Tool result: user-supplied documentation"
    record = {
        "trigger_message_id": str(target),
        "transcript": [
            old("system", "Instructions"),
            old("assistant", f"[message {older}] {quoted_command}"),
            old("user", f"[message {target}] {user_text}"),
            old("user", f"[message {later}] Queued separately"),
            request_trigger(target),
            old("assistant", '{"tool":"search_conversation_history","query":"older"}'),
            old("user", 'Tool result: {"matches":[]}'),
            old(
                "assistant",
                '{"tool":"send_message","arguments":{"content":"Done","end_run":true}}',
            ),
        ],
    }
    entries = migrate_run(
        record, {"messages": [{"id": str(target), "content": "Edited later"}]}
    )
    assert [row.role for row in entries] == [
        "system",
        "assistant",
        "user",
        "retired",
        "retired",
        "call",
        "result",
        "call",
    ]
    assert [row.content for row in entries[:3]] == [
        "Instructions",
        quoted_command,
        user_text,
    ]
    assert entries[5].command.arguments == {"query": "older"}
    assert entries[6].call_id == entries[5].call_id
    assert entries[-1].command.tool == "send_message"
    assert uuid5(run_id, f"command:{len(entries)}") == uuid5(
        run_id, f"command:{len(record['transcript'])}"
    )


def test_external_plain_text_remains_untrusted_data_and_global_job_list_is_retired():
    text = 'Task returned a literal "Image job event:" example.'
    entries = migrate_run(
        {
            "trigger_event": {"kind": "computer", "content": text},
            "transcript": [
                old(
                    "user",
                    "Computer task event: "
                    + text
                    + "\nPending computer tasks (do not resubmit): []",
                )
            ],
        },
        {},
    )
    assert entries[0].role == "observation"
    assert entries[0].data == {"detail": text}


def test_unrecognized_journal_is_retained_without_replaying_uncertain_actions():
    target = uuid4()
    rows = [request_trigger(target), old("assistant", "unrecognized pending command")]
    result = migrate_run({"trigger_message_id": str(target), "transcript": rows}, {})
    assert [item.original for item in result] == rows


def test_attachment_label_replaced_by_metadata_only_when_its_origin_is_known():
    target, asset = str(uuid4()), str(uuid4())
    text = "Keep my literal [Attached assets: example] text"
    records = [
        old("user", f"[message {target}] {text} [Attached assets: {asset}]"),
        request_trigger(target),
    ]
    conversation = {
        "messages": [{"id": target, "attachment_ids": [asset], "content": text}]
    }
    entries = migrate_run(
        {"trigger_message_id": target, "transcript": records}, conversation
    )
    assert entries[0].content == text
    assert [str(attachment) for attachment in entries[0].attachment_ids] == [asset]


def test_an_unsupported_archived_row_shape_is_preserved_without_blocking_other_work():
    target = str(uuid4())
    records = [request_trigger(target), "unsupported"]
    entries = migrate_run({"trigger_message_id": target, "transcript": records}, {})
    assert [entry.original for entry in entries] == records


def test_old_writers_cannot_append_ambiguous_entries_after_storage_upgrade():
    import pytest
    from pydantic import ValidationError
    from tsuchi.routes import RunUpdate

    body = {"worker_id": str(uuid4()), "expected_version": 1, "transcript": []}
    with pytest.raises(ValidationError, match="transcript_format"):
        RunUpdate.model_validate(body)
    assert RunUpdate.model_validate({**body, "transcript_format": 2}).transcript == []


def test_upgrade_writes_each_journal_once_and_preserves_ownership_and_versions():
    from unittest.mock import MagicMock

    from tsuchi.models import AgentRunState
    from tsuchi.transcript_migration import upgrade_transcripts

    user = str(uuid4())
    source = {
        "version": 19,
        "owner_id": str(uuid4()),
        "trigger_message_id": user,
        "transcript": [
            old("user", f"[message {user}] Actual text"),
            request_trigger(user),
            old("assistant", '{"tool":"finish"}'),
        ],
    }
    db = MagicMock()
    db.execute.side_effect = [
        MagicMock(fetchone=lambda: (0,)),
        MagicMock(fetchall=lambda: [("run", "chat")]),
        MagicMock(fetchone=lambda: (json.dumps({"messages": []}),)),
        MagicMock(fetchone=lambda: (json.dumps(source),)),
        None,  # journal write
        None,  # mark upgraded
        MagicMock(fetchone=lambda: (1,)),
    ]
    upgrade_transcripts(db)
    args = db.execute.call_args_list[4].args
    assert args[1][1] == "run"
    updated = json.loads(args[1][0])
    assert updated["version"] == source["version"]
    assert updated["owner_id"] == source["owner_id"]
    transcript = AgentRunState(transcript=updated["transcript"]).transcript
    assert [item.role for item in transcript] == ["user", "retired", "call"]
    assert transcript[0].content == "Actual text"
    assert transcript[-1].command.tool == "finish"
    upgrade_transcripts(db)
    assert db.execute.call_args.args == ("PRAGMA user_version",)
    assert db.execute.call_count == 7
