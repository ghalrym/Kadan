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
    script.STATE.mkdir()
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


def test_start_launches_background_run_and_returns_thread(home, tmp_path):
    thread_id = str(uuid4())
    project = tmp_path / "project"
    project.mkdir()
    launched = {}

    def launch(command, output, cwd, stdin):
        launched.update(command=command, cwd=cwd, prompt=stdin.read().decode())
        output.write_text(
            json.dumps({"type": "thread.started", "thread_id": thread_id})
        )
        return {"pid": 10, "start": "1"}

    with (
        patch.object(script, "setup", return_value={"state": "signed_in"}),
        patch.object(script, "launch", side_effect=launch),
        patch.object(script, "alive", return_value=True),
    ):
        started = script.start(request(prompt="Do it", cwd=str(project)))
        assert script.runs()[thread_id]["pid"] == 10
    assert started == {"thread_id": thread_id, "cwd": str(project), "running": True}
    assert launched["command"][:2] == ["codex", "exec"]
    assert launched["command"][-1] == "-"
    assert launched["cwd"] == str(project)
    assert launched["prompt"] == "Do it"
    assert not list((script.STATE / "runs").glob("*.prompt"))


def test_resume_uses_thread_folder(home, tmp_path):
    thread_id = str(uuid4())
    project = tmp_path / "project"
    project.mkdir()
    rollout(home, thread_id, str(project), [])
    launched = {}

    def launch(command, output, cwd, stdin):
        launched.update(command=command, cwd=cwd)
        output.write_text(
            json.dumps({"type": "thread.started", "thread_id": thread_id})
        )
        return {"pid": 10, "start": "1"}

    with (
        patch.object(script, "setup", return_value={"state": "signed_in"}),
        patch.object(script, "launch", side_effect=launch),
        patch.object(script, "alive", return_value=True),
        patch.object(script, "runs", return_value={}),
    ):
        script.start(request(prompt="More", thread_id=thread_id, cwd="/elsewhere"))
    assert launched["command"][:3] == ["codex", "exec", "resume"]
    assert launched["command"][-2:] == [thread_id, "-"]
    assert launched["cwd"] == str(project)


def test_start_refuses_running_thread(home):
    thread_id = str(uuid4())
    with (
        patch.object(script, "runs", return_value={thread_id: {}}),
        pytest.raises(ValueError, match="already running"),
    ):
        script.start(request(prompt="More", thread_id=thread_id))


def test_start_requires_sign_in(home, tmp_path):
    with (
        patch.object(script, "setup", return_value={"state": "signed_out"}),
        pytest.raises(ValueError, match="isn't signed in"),
    ):
        script.start(request(prompt="Do it", cwd=str(tmp_path)))


def write_record(request_id, log, **record):
    runs = script.STATE / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{request_id}.json").write_text(
        json.dumps({"pid": 10, "start": "1", "cwd": "/p", "origin": ORIGIN, **record})
    )
    (runs / f"{request_id}.jsonl").write_text(
        "\n".join(json.dumps(line) for line in log) + "\n"
    )


def agent_message(text):
    return {"type": "item.completed", "item": {"type": "agent_message", "text": text}}


@pytest.mark.parametrize(
    ("log", "record", "status", "message"),
    [
        (
            [agent_message("Built it."), {"type": "turn.completed"}],
            {},
            "completed",
            "Built it.",
        ),
        (
            [{"type": "turn.failed", "error": {"message": "Usage limit reached"}}],
            {},
            "failed",
            "Usage limit reached",
        ),
        ([agent_message("Halfway")], {"stopped": True}, "stopped", "Halfway"),
        ([], {}, "failed", "Codex exited before finishing its turn."),
    ],
)
def test_finished_runs_report_their_outcome_until_acknowledged(
    home, log, record, status, message
):
    request_id, thread_id = str(uuid4()), str(uuid4())
    write_record(request_id, log, thread_id=thread_id, **record)
    with patch.object(script, "alive", return_value=False):
        assert script.finished() == {
            "runs": [
                {
                    "request_id": request_id,
                    "thread_id": thread_id,
                    "cwd": "/p",
                    "origin": ORIGIN,
                    "status": status,
                    "message": message,
                }
            ]
        }
        assert script.runs() == {}
        script.acknowledge(request_id)
        assert script.finished() == {"runs": []}
    assert list((script.STATE / "runs").iterdir()) == []


def test_running_runs_are_not_finished(home):
    write_record(str(uuid4()), [], thread_id=str(uuid4()))
    with patch.object(script, "alive", return_value=True):
        assert script.finished() == {"runs": []}


def test_retried_start_returns_the_original_run(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    write_record(request_id, [], thread_id=thread_id)
    with (
        patch.object(script, "launch") as launch,
        patch.object(script, "alive", return_value=False),
    ):
        started = script.start(
            {"request_id": request_id, "origin": ORIGIN, "prompt": "Do it"}
        )
    assert started == {"thread_id": thread_id, "cwd": "/p", "running": False}
    launch.assert_not_called()


def test_start_requires_its_origin(home, tmp_path):
    with pytest.raises(ValueError, match="conversation that started"):
        script.start(
            {"request_id": str(uuid4()), "prompt": "Do it", "cwd": str(tmp_path)}
        )


def test_failed_start_leaves_no_record_to_report(home, tmp_path):
    with (
        patch.object(script, "setup", return_value={"state": "signed_in"}),
        patch.object(script, "launch", return_value={"pid": 10, "start": "1"}),
        patch.object(script, "alive", return_value=False),
        pytest.raises(ValueError, match="exited before starting"),
    ):
        script.start(request(prompt="Do it", cwd=str(tmp_path)))
    assert list((script.STATE / "runs").glob("*.json")) == []


def test_stop_marks_the_run_stopped_and_terminates_it(home):
    request_id, thread_id = str(uuid4()), str(uuid4())
    write_record(request_id, [], thread_id=thread_id)
    alive = iter([True, False])
    with (
        patch.object(script, "alive", side_effect=lambda record: next(alive, False)),
        patch.object(script, "terminate") as terminate,
    ):
        assert script.stop(thread_id) == {"thread_id": thread_id, "running": False}
    terminate.assert_called_once()
    saved = json.loads((script.STATE / "runs" / f"{request_id}.json").read_text())
    assert saved["stopped"] is True


def test_login_that_exits_without_a_code_reports_failure(home):
    with (
        patch.object(script, "setup", return_value={"state": "signed_out"}),
        patch.object(script, "launch", return_value={"pid": 10, "start": "1"}),
        pytest.raises(ValueError, match="stopped before it gave a code"),
    ):
        script.login()


def test_new_thread_creates_missing_project_folder(home, tmp_path):
    thread_id = str(uuid4())
    project = tmp_path / "new" / "hello-world"

    def launch(command, output, cwd, stdin):
        output.write_text(
            json.dumps({"type": "thread.started", "thread_id": thread_id})
        )
        return {"pid": 10, "start": "1"}

    with (
        patch.object(script, "setup", return_value={"state": "signed_in"}),
        patch.object(script, "launch", side_effect=launch),
        patch.object(script, "alive", return_value=True),
    ):
        script.start(request(prompt="Make it", cwd=str(project)))
    assert project.is_dir()


def test_folder_is_not_created_when_signed_out(home, tmp_path):
    project = tmp_path / "never"
    with (
        patch.object(script, "setup", return_value={"state": "signed_out"}),
        pytest.raises(ValueError, match="isn't signed in"),
    ):
        script.start(request(prompt="Make it", cwd=str(project)))
    assert not project.exists()


def test_login_returns_existing_device_prompt():
    prompt = {"state": "signing_in", "message": "Enter the code", "login": {"url": "https://example.com", "code": "ABCD-EFGH"}}
    with patch.object(script, "setup", return_value=prompt), patch.object(script, "launch") as launch:
        assert script.login() == prompt
        launch.assert_not_called()


def test_login_starts_device_authorization():
    with patch.object(script, "setup", return_value={"state": "signed_out"}), patch.object(script, "launch", return_value={"pid": 3}), patch.object(script, "write_json") as write, patch.object(script, "wait_for_login", return_value={"state": "signing_in"}):
        assert script.login() == {"state": "signing_in"}
        assert write.call_args.args[1] == {"pid": 3}
