"""Computer-side Codex script with every process boundary replaced."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "sandbox" / "codex_threads.py"
SPEC = importlib.util.spec_from_file_location("kadan_codex_threads", SOURCE)
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "CODEX_HOME", tmp_path / "codex")
    monkeypatch.setattr(script, "STATE", tmp_path / "state")
    script.STATE.mkdir()
    return tmp_path


def rollout(home, thread_id, cwd, entries):
    path = home / "codex" / "sessions" / "2026" / "10" / "30"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"rollout-2026-10-30T12-00-00-{thread_id}.jsonl"
    meta = {"type": "session_meta", "payload": {"id": thread_id, "cwd": cwd}}
    file.write_text("\n".join(json.dumps(item) for item in [meta, *entries]) + "\n")
    return file


ORIGIN = {
    "agent_key": "guide",
    "conversation_id": "00000000-0000-4000-8000-000000000001",
    "run_id": "00000000-0000-4000-8000-000000000002",
}


def request(**fields):
    return {"request_id": str(uuid4()), "origin": ORIGIN, **fields}


def event(kind, message):
    return {"type": "event_msg", "payload": {"type": kind, "message": message}}


def test_thread_reads_messages_and_commands(home):
    thread_id = str(uuid4())
    rollout(
        home,
        thread_id,
        "/home/gem/project",
        [
            event("user_message", "Fix the  parser\nplease"),
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": ["bash", "-lc", "pytest -q"]}),
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "custom_tool_call", "name": "apply_patch"},
            },
            {"type": "response_item", "payload": {"type": "reasoning"}},
            event("agent_message", "Fixed it."),
            "not an object",
        ],
    )
    with patch.object(script, "runs", return_value={}):
        thread = script.thread(thread_id)
    assert thread["title"] == "Fix the parser please"
    assert thread["cwd"] == "/home/gem/project"
    assert thread["running"] is False
    assert thread["items"] == [
        {"kind": "user", "text": "Fix the  parser\nplease"},
        {"kind": "command", "text": "Ran pytest -q"},
        {"kind": "command", "text": "Edited files"},
        {"kind": "assistant", "text": "Fixed it."},
    ]


def test_threads_mark_kadan_runs_as_running(home):
    first, second = str(uuid4()), str(uuid4())
    rollout(home, first, "/a", [event("user_message", "One")])
    rollout(home, second, "/b", [event("user_message", "Two")])
    with (
        patch.object(script, "runs", return_value={second: {}}),
        patch.object(script, "setup", return_value={"state": "signed_in"}),
    ):
        listed = {row["id"]: row["running"] for row in script.threads()["threads"]}
    assert listed == {first: False, second: True}


def test_unknown_thread_is_rejected(home):
    with pytest.raises(ValueError, match="isn't on this computer"):
        script.thread(str(uuid4()))


def test_login_prompt_reads_link_and_code_through_colours(home):
    script.STATE.mkdir(exist_ok=True)
    (script.STATE / "login.log").write_text(
        "1. Open \x1b[94mhttps://auth.example/codex/device\x1b[0m\n"
        "2. Enter this one-time code\n   \x1b[94mAB12-CD34\x1b[0m\n"
    )
    assert script.login_prompt() == {
        "url": "https://auth.example/codex/device",
        "code": "AB12-CD34",
    }


@pytest.mark.parametrize(
    ("code", "login_alive", "state"),
    [(0, False, "signed_in"), (1, False, "signed_out"), (1, True, "signing_in")],
)
def test_setup_reports_sign_in_state(home, code, login_alive, state):
    with (
        patch.object(script.shutil, "which", return_value="/usr/bin/codex"),
        patch.object(
            script.subprocess, "run", return_value=SimpleNamespace(returncode=code)
        ),
        patch.object(script, "alive", return_value=login_alive),
        patch.object(script, "terminate"),
        patch.object(script, "login_prompt", return_value={"url": None, "code": None}),
    ):
        assert script.setup()["state"] == state


def test_missing_codex_is_reported(home):
    with patch.object(script.shutil, "which", return_value=None):
        assert script.setup()["state"] == "missing"


def save(request_id, **fields):
    record = {
        "pid": 10,
        "start": "1",
        "incarnation": script.incarnation(),
        "cwd": "/p",
        "origin": ORIGIN,
        **fields,
    }
    script.write_json(script.STATE / "runs" / f"{request_id}.json", record)
    return record


def test_reserve_persists_before_launch_and_rejects_duplicate_resume(home):
    thread_id = str(uuid4())
    project = home / "project"
    project.mkdir()
    rollout(home, thread_id, str(project), [])
    req = request(prompt="More", thread_id=thread_id)

    def launch(command, output, cwd):
        saved = script.read_json(script.STATE / "runs" / f"{req['request_id']}.json")
        assert saved["origin"] == ORIGIN
        assert saved["resume_thread_id"] == thread_id
        assert saved["command"][:3] == ["codex", "exec", "resume"]
        assert command[-2:] == ["--worker", req["request_id"]]
        return {"pid": 10, "start": "1", "incarnation": script.incarnation()}

    with (
        patch.object(script, "setup", return_value={"state": "signed_in"}),
        patch.object(script, "launch", side_effect=launch),
        patch.object(script, "alive", return_value=True),
    ):
        script.reserve(req, req["request_id"])
        with pytest.raises(ValueError):
            other = request(prompt="Again", thread_id=thread_id)
            script.reserve(other, other["request_id"])


def test_acknowledged_start_retains_receipt_and_cannot_launch_twice(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    save(
        request_id,
        thread_id=thread_id,
        cleanup_confirmed=True,
        status="completed",
        message="Done",
    )
    assert script.finished()["runs"][0]["message"] == "Done"
    script.acknowledge(request_id)
    assert script.finished() == {"runs": []}
    with patch.object(script, "launch") as launch:
        assert script.start({"request_id": request_id, "origin": ORIGIN}) == {
            "thread_id": thread_id,
            "cwd": "/p",
            "running": False,
        }
    launch.assert_not_called()


def test_uncertain_cleanup_is_not_delivered_and_blocks_new_work(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    save(request_id, thread_id=thread_id)
    with patch.object(script, "alive", return_value=False):
        assert script.finished() == {"runs": []}
        assert thread_id in script.runs()
        req = request(prompt="Again", cwd=str(home))
        with pytest.raises(ValueError, match="cleanup is unconfirmed"):
            script.reserve(req, req["request_id"])
    with pytest.raises(ValueError, match="still working"):
        script.acknowledge(request_id)


def test_computer_restart_releases_uncertain_work_and_reports_failure(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    save(request_id, thread_id=thread_id, incarnation="old-computer")
    assert script.runs() == {}
    assert script.finished()["runs"][0]["status"] == "failed"


def test_stop_waits_for_cleanup_confirmation(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    save(request_id, thread_id=thread_id)

    def finish(_seconds):
        record = script.read_json(script.STATE / "runs" / f"{request_id}.json")
        assert record["stopped"]
        save(request_id, **{**record, "cleanup_confirmed": True})

    with patch.object(script.time, "sleep", side_effect=finish):
        assert script.stop(thread_id) == {"thread_id": thread_id, "running": False}


def test_stop_failure_keeps_thread_working(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    save(request_id, thread_id=thread_id)
    with patch.object(script.time, "monotonic", side_effect=[0, 8]):
        with pytest.raises(ValueError, match="cleanup is unconfirmed"):
            script.stop(thread_id)
    assert thread_id in script.runs()


@pytest.fixture
def supervisor(home, monkeypatch):
    from unittest.mock import Mock

    request_id, thread_id = str(uuid4()), str(uuid4())
    save(request_id, command=["codex", "exec"], cwd=str(home))
    prompt = script.STATE / "runs" / f"{request_id}.prompt"
    prompt.write_text("Build it")
    process = Mock()
    process.poll.return_value = 0

    def popen(*args, **kwargs):
        assert kwargs["stdin"].read() == b"Build it"
        kwargs["stdout"].write(
            (
                json.dumps({"type": "thread.started", "thread_id": thread_id})
                + "\n"
                + json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "Done"},
                    }
                )
                + "\n"
                + json.dumps({"type": "turn.completed"})
                + "\n"
            ).encode()
        )
        return process

    launch = Mock(side_effect=popen)
    monkeypatch.setattr(script.subprocess, "Popen", launch)
    monkeypatch.setattr(script.signal, "signal", Mock())
    monkeypatch.setattr(
        script.ctypes, "CDLL", Mock(return_value=SimpleNamespace(prctl=lambda *args: 0))
    )
    monkeypatch.setattr(script, "alive", lambda record: False)
    cleanup = Mock(return_value=True)
    monkeypatch.setattr(script, "cleanup", cleanup)
    return SimpleNamespace(
        request_id=request_id,
        thread_id=thread_id,
        process=process,
        launch=launch,
        cleanup=cleanup,
    )


def test_supervisor_publishes_only_after_cleanup(supervisor):
    def cleanup():
        assert script.finished() == {"runs": []}
        return True

    supervisor.cleanup.side_effect = cleanup
    script.worker(supervisor.request_id)
    result = script.finished()["runs"][0]
    assert result["status"] == "completed"
    assert result["message"] == "Done"
    assert result["thread_id"] == supervisor.thread_id


def test_supervisor_nonzero_exit_overrides_completion_log(supervisor):
    supervisor.process.poll.return_value = 1
    script.worker(supervisor.request_id)
    assert script.finished()["runs"][0]["status"] == "failed"


def test_supervisor_unconfirmed_descendants_block_delivery(supervisor):
    supervisor.cleanup.return_value = False
    script.worker(supervisor.request_id)
    assert script.finished() == {"runs": []}
    assert supervisor.thread_id in script.runs()


def test_supervisor_cannot_spawn_without_descendant_ownership(supervisor):
    with patch.object(
        script.ctypes, "CDLL", return_value=SimpleNamespace(prctl=lambda *args: -1)
    ):
        script.worker(supervisor.request_id)
    supervisor.launch.assert_not_called()
    record = script.read_json(script.STATE / "runs" / f"{supervisor.request_id}.json")
    assert record["cleanup_confirmed"]
    assert record["status"] == "failed"


def test_cleanup_signals_only_matching_descendants(home):

    with (
        patch.object(script, "children_of", side_effect=[{77: "old"}, {}, {}]),
        patch.object(script.os, "pidfd_open", return_value=9),
        patch.object(script.os, "close") as close,
        patch.object(script, "process_start", return_value="reused"),
        patch.object(script.signal, "pidfd_send_signal") as send,
        patch.object(script, "reap"),
    ):
        assert script.cleanup()
    send.assert_not_called()
    close.assert_called_once_with(9)


def test_new_project_is_created_only_after_sign_in(home):
    req = request(prompt="Build", cwd=str(home / "new"))
    with patch.object(script, "setup", return_value={"state": "signed_out"}):
        with pytest.raises(ValueError, match="isn't signed in"):
            script.reserve(req, req["request_id"])
    assert not (home / "new").exists()
    with (
        patch.object(script, "setup", return_value={"state": "signed_in"}),
        patch.object(script, "launch", return_value={"pid": 10}),
    ):
        script.reserve(req, req["request_id"])
    assert (home / "new").is_dir()


@pytest.mark.parametrize("stopped,expected", [(False, "failed"), (True, "stopped")])
def test_outcome_retains_failure_even_after_an_agent_message(home, stopped, expected):
    log = home / "events.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Working"},
                },
                {"type": "turn.failed", "error": {"message": "Usage limit"}},
            ]
        )
    )
    status, message = script.outcome(log, stopped)
    assert status == expected
    if not stopped:
        assert "Usage limit" in message


def test_supervisor_observes_durable_stop_and_confirms_cleanup(supervisor):
    supervisor.process.poll.return_value = None
    path = script.STATE / "runs" / f"{supervisor.request_id}.json"
    record = script.read_json(path)
    script.write_json(path, {**record, "stopped": True})
    script.worker(supervisor.request_id)
    supervisor.launch.assert_not_called()
    assert script.read_json(path)["cleanup_confirmed"]


def test_cleanup_escalates_when_descendant_survives_term(home):
    signals = []
    with (
        patch.object(
            script, "children_of", side_effect=[{77: "1"}, {77: "1"}, {77: "1"}, {}]
        ),
        patch.object(script.os, "pidfd_open", return_value=9),
        patch.object(script.os, "close"),
        patch.object(script, "process_start", return_value="1"),
        patch.object(
            script.signal,
            "pidfd_send_signal",
            side_effect=lambda fd, sig: signals.append(sig),
        ),
        patch.object(script, "reap"),
        patch.object(script.time, "monotonic", side_effect=[0, 3, 3]),
    ):
        assert script.cleanup()
    assert signals == [script.signal.SIGTERM, script.signal.SIGKILL]


def test_retry_without_thread_id_reports_uncertainty_without_relaunch(home):
    request_id = str(uuid4())
    record = save(request_id)
    log = script.STATE / "runs" / f"{request_id}.jsonl"
    log.write_text("existing output\n")
    with (
        patch.object(script, "launch") as launch,
        patch.object(script.time, "monotonic", side_effect=[0, 21]),
        pytest.raises(ValueError, match="start is unconfirmed"),
    ):
        script.start({"request_id": request_id, "origin": ORIGIN})
    launch.assert_not_called()
    assert script.read_json(script.STATE / "runs" / f"{request_id}.json") == record
    assert log.read_text() == "existing output\n"


def test_retry_waits_for_existing_supervisor_to_publish_thread(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    save(request_id)
    with (
        patch.object(script, "launch") as launch,
        patch.object(
            script.time,
            "sleep",
            side_effect=lambda seconds: save(request_id, thread_id=thread_id),
        ),
    ):
        result = script.start({"request_id": request_id, "origin": ORIGIN})
    assert result == {"thread_id": thread_id, "cwd": "/p", "running": True}
    launch.assert_not_called()
