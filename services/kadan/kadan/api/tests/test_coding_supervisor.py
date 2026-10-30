"""Supervisor unit tests replace every external process boundary."""

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "sandbox" / "coding_jobs.py"
SPEC = importlib.util.spec_from_file_location("kadan_coding_supervisor", SOURCE)
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)


@pytest.fixture
def store(tmp_path):
    with (
        patch.object(supervisor, "incarnation", return_value="computer-a"),
        patch.object(supervisor, "process_start", return_value="100"),
        patch.object(supervisor, "setup_status", return_value={"state": "ready"}),
    ):
        yield supervisor.Jobs(tmp_path)


def request(tmp_path, **values):
    return {
        "request_id": str(uuid4()),
        "prompt": "Make the requested code change.",
        "cwd": str(tmp_path),
        **values,
    }


def launch(store, body):
    with patch.object(
        supervisor.subprocess, "Popen", return_value=SimpleNamespace(pid=123)
    ):
        return store.start(body)


def finish(store, result, status="completed", session=True):
    job = store.get(result["id"])
    job.status = status
    job.cleanup_confirmed = True
    job.session_id = str(uuid4()) if session else None
    store.save(job)
    return job


def test_retry_returns_same_job_without_launching_again(store, tmp_path):
    body = request(tmp_path)
    result = launch(store, body)
    with patch.object(supervisor.subprocess, "Popen") as spawn:
        assert store.start(body)["id"] == body["request_id"] == result["id"]
        spawn.assert_not_called()
    persisted = (tmp_path / f"{result['id']}.json").read_text()
    assert body["prompt"] not in persisted


def test_retry_rejects_changed_payload(store, tmp_path):
    body = request(tmp_path)
    launch(store, body)
    with pytest.raises(ValueError, match="different coding instructions"):
        store.start({**body, "prompt": "Different instructions"})


def test_shared_computer_accepts_only_one_active_job(store, tmp_path):
    launch(store, request(tmp_path))
    with pytest.raises(ValueError, match="active coding job"):
        store.start(request(tmp_path))


def test_cancel_is_durable_and_repeatable_without_signalling_worker(store, tmp_path):
    result = launch(store, request(tmp_path))
    with patch.object(supervisor.os, "kill") as kill:
        assert store.cancel(result["id"])["status"] == "stopping"
        assert store.cancel(result["id"])["status"] == "stopping"
        assert store.get(result["id"]).cancel_requested
        kill.assert_not_called()


def test_finished_cancellation_does_not_change_result(store, tmp_path):
    result = launch(store, request(tmp_path))
    finish(store, result)
    assert store.cancel(result["id"])["status"] == "completed"


def test_dead_supervisor_requires_restart_before_new_job(store, tmp_path):
    result = launch(store, request(tmp_path))
    with patch.object(supervisor, "process_start", return_value=None):
        assert store.status(result["id"])["status"] == "cleanup_required"
        with pytest.raises(ValueError, match="requires cleanup"):
            store.start(request(tmp_path))
    assert not store.get(result["id"]).cleanup_confirmed


def test_restart_confirms_old_job_interruption_and_allows_next_job(store, tmp_path):
    result = launch(store, request(tmp_path))
    with patch.object(supervisor, "incarnation", return_value="computer-b"):
        assert store.status(result["id"])["status"] == "interrupted"
        assert store.get(result["id"]).cleanup_confirmed
        assert launch(store, request(tmp_path))["status"] == "starting"


def test_startup_spawn_uncertainty_blocks_after_grace(store, tmp_path):
    result = launch(store, request(tmp_path))
    job = store.get(result["id"])
    job.worker_pid = None
    store.save(job)
    assert store.status(job.id)["status"] == "starting"
    with patch.object(supervisor.time, "time", return_value=job.launched_at + 16):
        assert store.status(job.id)["status"] == "cleanup_required"


def test_spawn_failure_can_release_slot(store, tmp_path):
    with patch.object(
        supervisor.subprocess, "Popen", side_effect=OSError("private diagnostic")
    ):
        result = store.start(request(tmp_path))
    assert result["status"] == "failed"
    assert store.get(result["id"]).cleanup_confirmed
    assert "private" not in json.dumps(result)
    assert not list(tmp_path.glob("*.prompt"))
    assert launch(store, request(tmp_path))["status"] == "starting"


def test_resume_uses_only_recorded_session_and_original_directory(store, tmp_path):
    result = launch(store, request(tmp_path))
    previous = finish(store, result)
    resumed = launch(store, request(tmp_path, resume_job_id=previous.id))
    job = store.get(resumed["id"])
    assert job.session_id == previous.session_id
    args = supervisor.command(job)
    assert args[:3] == ["codex", "exec", "resume"]
    assert args[-2:] == [previous.session_id, "-"]
    assert "--last" not in args
    assert 'sandbox_mode="workspace-write"' in args


def test_resume_rejects_changed_directory_and_missing_session(store, tmp_path):
    result = launch(store, request(tmp_path))
    previous = finish(store, result)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="working directory"):
        store.start(request(other, resume_job_id=previous.id))
    finish(store, result, session=False)
    with pytest.raises(ValueError, match="recorded Codex session"):
        store.start(request(tmp_path, resume_job_id=previous.id))


def test_events_are_bounded_and_paged(store, tmp_path):
    result = launch(store, request(tmp_path))
    job = store.get(result["id"])
    for index in range(150):
        job.event("progress", "x" * 1000)
    store.save(job)
    first = store.status(job.id)
    assert len(store.get(job.id).events) == 100
    assert len(first["events"]) == 20
    assert all(len(event["message"]) == 300 for event in first["events"])
    second = store.status(job.id, first["next_cursor"])
    assert second["events"][0]["sequence"] == first["next_cursor"] + 1


def test_setup_distinguishes_missing_package_auth_and_supported_version():
    with patch.object(supervisor.shutil, "which", return_value=None):
        assert supervisor.setup_status()["state"] == "missing"
    with (
        patch.object(supervisor.shutil, "which", return_value="/bin/codex"),
        patch.object(
            supervisor.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(returncode=0, stdout=b"codex-cli 0.162.1\n"),
                SimpleNamespace(returncode=1),
            ],
        ) as run,
    ):
        assert supervisor.setup_status()["state"] == "auth_required"
        assert run.call_args.args[0] == ["/bin/codex", "login", "status"]
        assert run.call_args.kwargs["stderr"] == supervisor.subprocess.DEVNULL
        assert run.call_args.kwargs["env"]["CODEX_HOME"] == "/home/gem/.codex"
    with (
        patch.object(supervisor.shutil, "which", return_value="/bin/codex"),
        patch.object(
            supervisor.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=b"codex-cli 0.139.0\n"),
        ) as run,
    ):
        assert supervisor.setup_status()["state"] == "unavailable"
        assert run.call_count == 1


def test_progress_never_persists_raw_command_output(store, tmp_path):
    job = store.get(launch(store, request(tmp_path))["id"])
    supervisor.record_event(
        job,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "private-command",
                "aggregated_output": "private-secret",
            },
        },
    )
    assert "private" not in json.dumps(job.events)
    text = supervisor.record_event(
        job,
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "Fixed the bug. api_key=private-secret " + "x" * 3000,
            },
        },
    )
    assert text.startswith("Fixed the bug. api_key=[redacted]")
    assert "private-secret" not in text
    assert len(text) == 2000


@pytest.mark.parametrize("outcome", ["cancelled", "timed_out", "interrupted"])
def test_monitor_obeys_cancellation_timeout_and_interrupt(store, tmp_path, outcome):
    job = store.get(launch(store, request(tmp_path))["id"])
    if outcome == "cancelled":
        store.cancel(job.id)
    elif outcome == "timed_out":
        job.timeout_seconds = 0
    process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 8))
    with (
        patch.object(supervisor.os, "set_blocking"),
        patch.object(supervisor.selectors, "DefaultSelector"),
    ):
        result = supervisor.monitor(
            store, job, process, lambda: outcome == "interrupted"
        )
    assert result[0] == outcome


def test_cleanup_escalates_and_confirms_reaped_detached_children():
    with (
        patch.object(
            supervisor,
            "children_of",
            side_effect=[{5: "100"}, {5: "100"}, {5: "100"}, {}],
        ),
        patch.object(supervisor, "process_start", return_value="100"),
        patch.object(supervisor, "reap"),
        patch.object(supervisor.os, "pidfd_open", return_value=50),
        patch.object(supervisor.signal, "pidfd_send_signal") as send,
        patch.object(supervisor.os, "close"),
        patch.object(supervisor.time, "monotonic", side_effect=[0, 3, 3]),
        patch.object(supervisor.time, "sleep"),
    ):
        assert supervisor.cleanup()
    assert send.call_args_list[0].args == (50, signal.SIGTERM)
    assert send.call_args_list[1].args == (50, signal.SIGKILL)


def test_worker_never_claims_completion_before_confirmed_cleanup(store, tmp_path):
    result = launch(store, request(tmp_path))
    with (
        patch.object(supervisor, "Jobs", return_value=store),
        patch.object(supervisor.signal, "signal"),
        patch.object(
            supervisor.ctypes,
            "CDLL",
            return_value=SimpleNamespace(prctl=lambda *args: 0),
        ),
        patch.object(supervisor.subprocess, "Popen", return_value=Mock()),
        patch.object(
            supervisor,
            "monitor",
            return_value=("completed", "Finished changing files."),
        ),
        patch.object(supervisor, "cleanup", return_value=False),
    ):
        supervisor.worker(result["id"])
    assert store.get(result["id"]).status == "cleanup_required"
    assert not store.get(result["id"]).cleanup_confirmed
    assert not list(tmp_path.glob("*.prompt"))


def test_new_jobs_use_explicit_model_and_workspace_policy(store, tmp_path):
    job = store.get(launch(store, request(tmp_path))["id"])
    args = supervisor.command(job)
    assert "gpt-6.1-sol" in args
    assert 'model_reasoning_effort="high"' in args
    assert 'approval_policy="never"' in args
    assert args[args.index("--sandbox") + 1] == "workspace-write"
    assert "--ignore-user-config" in args


@pytest.mark.parametrize("failed", [False, True])
def test_monitor_parses_fragmented_events_without_retaining_tool_output(
    store, tmp_path, failed
):
    job = store.get(launch(store, request(tmp_path))["id"])
    session = str(uuid4())
    records = [
        {"type": "thread.started", "thread_id": session},
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "aggregated_output": "private-output",
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Updated code and unit tests."},
        },
        {"type": "turn.failed" if failed else "turn.completed"},
    ]
    data = ("\n".join(json.dumps(record) for record in records) + "\n").encode()
    stream = SimpleNamespace(fileno=lambda: 8)

    class Selector:
        def __init__(self):
            self.open = True

        def register(self, *args):
            pass

        def unregister(self, *args):
            self.open = False

        def select(self, *args):
            return [(SimpleNamespace(fileobj=stream), None)] if self.open else []

        def get_map(self):
            return {8: stream} if self.open else {}

        def close(self):
            pass

    process = SimpleNamespace(stdout=stream, poll=lambda: 0)
    with (
        patch.object(supervisor.os, "set_blocking"),
        patch.object(supervisor.os, "read", side_effect=[data[:8], data[8:], b""]),
        patch.object(supervisor.selectors, "DefaultSelector", Selector),
    ):
        status, message = supervisor.monitor(store, job, process)
    assert status == ("failed" if failed else "completed")
    if not failed:
        assert message == "Updated code and unit tests."
    assert store.get(job.id).session_id == session
    assert "private-output" not in json.dumps(store.get(job.id).events)


def test_cancel_before_worker_launches_codex_still_confirms_cleanup(store, tmp_path):
    result = launch(store, request(tmp_path))
    store.cancel(result["id"])
    with (
        patch.object(supervisor, "Jobs", return_value=store),
        patch.object(supervisor.signal, "signal"),
        patch.object(
            supervisor.ctypes,
            "CDLL",
            return_value=SimpleNamespace(prctl=lambda *args: 0),
        ),
        patch.object(supervisor.subprocess, "Popen") as spawn,
        patch.object(supervisor, "cleanup", return_value=True) as cleanup,
    ):
        supervisor.worker(result["id"])
    spawn.assert_not_called()
    cleanup.assert_called_once_with()
    assert store.get(result["id"]).status == "cancelled"
    assert store.get(result["id"]).cleanup_confirmed
    assert not list(tmp_path.glob("*.prompt"))


def test_list_events_show_latest_progress_and_cursor_reports_dropped_history(
    store, tmp_path
):
    job = store.get(launch(store, request(tmp_path))["id"])
    for index in range(150):
        job.event("progress", f"Update {index}")
    store.save(job)
    latest = store.status(None)["jobs"][0]
    assert latest["events"][-1]["message"] == "Update 149"
    assert latest["events_truncated"]
    first_retained = store.status(job.id, after=5)
    assert first_retained["events_truncated"]
    next_page = store.status(job.id, after=first_retained["next_cursor"])
    assert not next_page["events_truncated"]


def test_cleanup_never_signals_reused_unrelated_pid():
    with (
        patch.object(supervisor, "children_of", side_effect=[{5: "original"}, {}]),
        patch.object(supervisor, "process_start", return_value="replacement"),
        patch.object(supervisor, "reap"),
        patch.object(supervisor.os, "pidfd_open", return_value=50),
        patch.object(supervisor.signal, "pidfd_send_signal") as send,
        patch.object(supervisor.os, "close") as close,
    ):
        assert supervisor.cleanup()
    send.assert_not_called()
    close.assert_called_once_with(50)


def test_cleanup_picks_up_descendants_reparented_after_parent_stops():
    scans = [{5: "parent"}, {6: "detached-child"}, {6: "detached-child"}, {}]
    with (
        patch.object(supervisor, "children_of", side_effect=scans),
        patch.object(
            supervisor, "process_start", side_effect=["parent", "detached-child"]
        ),
        patch.object(supervisor, "reap"),
        patch.object(supervisor.os, "pidfd_open", side_effect=[50, 60]),
        patch.object(supervisor.signal, "pidfd_send_signal") as send,
        patch.object(supervisor.os, "close"),
        patch.object(supervisor.time, "sleep"),
    ):
        assert supervisor.cleanup()
    assert [call.args for call in send.call_args_list] == [
        (50, signal.SIGTERM),
        (60, signal.SIGTERM),
    ]


def test_cleanup_failure_keeps_job_blocked_until_restart():
    with (
        patch.object(supervisor, "children_of", return_value={5: "parent"}),
        patch.object(supervisor, "process_start", return_value="parent"),
        patch.object(supervisor, "reap"),
        patch.object(supervisor.os, "pidfd_open", return_value=50),
        patch.object(supervisor.signal, "pidfd_send_signal"),
        patch.object(supervisor.os, "close"),
        patch.object(supervisor.time, "monotonic", side_effect=[0, 3, 3, 7]),
    ):
        assert not supervisor.cleanup()


def test_repeated_sigterm_during_cleanup_finishes_interrupted_with_ownership_confirmed(
    store, tmp_path
):
    result = launch(store, request(tmp_path))
    handlers = {}

    def install(sig, handler):
        handlers[sig] = handler

    def clean():
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return True

    with (
        patch.object(supervisor, "Jobs", return_value=store),
        patch.object(supervisor.signal, "signal", side_effect=install),
        patch.object(
            supervisor.ctypes,
            "CDLL",
            return_value=SimpleNamespace(prctl=lambda *args: 0),
        ),
        patch.object(supervisor.subprocess, "Popen", return_value=Mock()) as spawn,
        patch.object(
            supervisor,
            "monitor",
            return_value=("completed", "Finished changing files."),
        ),
        patch.object(supervisor, "cleanup", side_effect=clean),
    ):
        supervisor.worker(result["id"])
    assert store.get(result["id"]).status == "interrupted"
    assert store.get(result["id"]).cleanup_confirmed
    assert spawn.call_args.kwargs["env"]["CODEX_HOME"] == "/home/gem/.codex"


def test_repeated_cancellation_during_cleanup_does_not_interrupt_cleanup(
    store, tmp_path
):
    result = launch(store, request(tmp_path))

    def clean():
        assert store.cancel(result["id"])["status"] == "stopping"
        assert store.cancel(result["id"])["status"] == "stopping"
        assert not store.get(result["id"]).cleanup_confirmed
        return True

    with (
        patch.object(supervisor, "Jobs", return_value=store),
        patch.object(supervisor.signal, "signal"),
        patch.object(
            supervisor.ctypes,
            "CDLL",
            return_value=SimpleNamespace(prctl=lambda *args: 0),
        ),
        patch.object(supervisor.subprocess, "Popen", return_value=Mock()),
        patch.object(
            supervisor,
            "monitor",
            return_value=("completed", "Finished changing files."),
        ),
        patch.object(supervisor, "cleanup", side_effect=clean),
    ):
        supervisor.worker(result["id"])
    assert store.get(result["id"]).status == "cancelled"
    assert store.get(result["id"]).cleanup_confirmed


def test_zombie_worker_is_not_treated_as_running():
    stat = "123 (worker name) " + " ".join(["Z", "1"] + ["0"] * 17 + ["100"])
    with patch.object(supervisor.Path, "read_text", return_value=stat):
        assert supervisor.process_start(123) is None


def test_retry_reads_original_job_after_project_directory_removed(store, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    body = request(project)
    result = launch(store, body)
    finish(store, result)
    project.rmdir()
    with patch.object(supervisor.subprocess, "Popen") as spawn:
        assert store.start(body)["id"] == result["id"]
        spawn.assert_not_called()


def test_retry_reads_original_job_after_directory_symlink_retargeted(store, tmp_path):
    first, second, link = (tmp_path / name for name in ("first", "second", "link"))
    first.mkdir()
    second.mkdir()
    link.symlink_to(first, target_is_directory=True)
    body = request(link)
    result = launch(store, body)
    finish(store, result)
    link.unlink()
    link.symlink_to(second, target_is_directory=True)
    with patch.object(supervisor.subprocess, "Popen") as spawn:
        assert store.start(body)["cwd"] == str(first)
        spawn.assert_not_called()


def test_saved_sign_in_check_does_not_claim_model_entitlement_or_run_inference():
    with (
        patch.object(supervisor.shutil, "which", return_value="/bin/codex"),
        patch.object(
            supervisor.subprocess,
            "run",
            side_effect=[
                SimpleNamespace(returncode=0, stdout=b"codex-cli 0.162.1\n"),
                SimpleNamespace(returncode=0),
            ],
        ) as run,
    ):
        result = supervisor.setup_status()
    assert result["state"] == "ready"
    assert "saved sign-in" in result["message"]
    assert "model access and available usage" in result["message"]
    assert [call.args[0] for call in run.call_args_list] == [
        ["/bin/codex", "--version"],
        ["/bin/codex", "login", "status"],
    ]


def test_setup_failure_gives_recovery_without_exposing_diagnostics():
    with (
        patch.object(supervisor.shutil, "which", return_value="/bin/codex"),
        patch.object(
            supervisor.subprocess,
            "run",
            side_effect=supervisor.subprocess.TimeoutExpired(
                "private-command", 5, stderr=b"private-secret"
            ),
        ),
    ):
        result = supervisor.setup_status()
    assert result["state"] == "unavailable"
    assert "Refresh" in result["message"]
    assert "person who runs Kadan" in result["message"]
    assert "private" not in json.dumps(result)
