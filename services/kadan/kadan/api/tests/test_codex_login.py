import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(params=["codex_login", "codex_threads"])
def login(request):
    path = Path(__file__).parents[3] / "sandbox" / f"{request.param}.py"
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_login_returns_existing_device_prompt(login):
    prompt = {
        "state": "signing_in",
        "message": "Enter the code",
        "login": {"url": "https://example.com", "code": "ABCD-EFGH"},
    }
    with (
        patch.object(login, "setup", return_value=prompt),
        patch.object(login, "launch") as launch,
    ):
        assert login.login() == prompt
        launch.assert_not_called()


def test_login_starts_device_authorization(login):
    with (
        patch.object(login, "setup", return_value={"state": "signed_out"}),
        patch.object(login, "launch", return_value={"pid": 3}),
        patch.object(login, "write_json") as write,
    ):
        assert login.login()["state"] == "signing_in"
        assert write.call_args.args[1] == {"pid": 3}


def test_prompt_extracts_link_and_code(login, tmp_path):
    (tmp_path / "login.log").write_text(
        "Open https://example.com/device and enter ABCD-EFGH"
    )
    with patch.object(login, "STATE", tmp_path):
        assert login.login_prompt() == {
            "url": "https://example.com/device",
            "code": "ABCD-EFGH",
        }


def test_dead_or_unidentified_process_is_never_alive(login):
    with (
        patch.object(login, "process_start", return_value=None),
        patch.object(login, "incarnation", return_value="boot"),
    ):
        assert not login.alive({"pid": 123, "start": None, "incarnation": "boot"})
        assert not login.alive({"pid": 123, "start": "10", "incarnation": "boot"})


def test_process_from_previous_computer_is_not_terminated(login):
    with (
        patch.object(login, "process_start", return_value="10"),
        patch.object(login, "incarnation", return_value="new"),
        patch.object(login.os, "killpg") as kill,
    ):
        login.terminate({"pid": 123, "start": "10", "incarnation": "old"})
        kill.assert_not_called()


def test_logout_stops_pending_login_before_removing_credentials(login):
    from types import SimpleNamespace

    calls = []
    with (
        patch.object(login, "login_record", return_value={"pid": 123}),
        patch.object(
            login, "terminate", side_effect=lambda record: calls.append("stop")
        ),
        patch.object(
            login.subprocess,
            "run",
            side_effect=lambda *args, **kwargs: (
                calls.append("logout") or SimpleNamespace(returncode=0)
            ),
        ),
        patch.object(login, "setup", return_value={"state": "signed_out"}),
    ):
        assert login.logout()["state"] == "signed_out"
        assert calls == ["stop", "logout"]


def test_failed_logout_does_not_report_success(login):
    import pytest
    from types import SimpleNamespace

    with (
        patch.object(login, "terminate"),
        patch.object(
            login.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ),
    ):
        with pytest.raises(ValueError, match="Sign-out failed"):
            login.logout()


def test_signed_in_status_cleans_pending_login(login):
    from types import SimpleNamespace

    with (
        patch.object(login.shutil, "which", return_value="codex"),
        patch.object(
            login.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ),
        patch.object(login, "login_record", return_value={"pid": 123}),
        patch.object(login, "terminate") as terminate,
    ):
        assert login.setup()["state"] == "signed_in"
        terminate.assert_called_once_with({"pid": 123})


def test_pending_login_status_reuses_saved_prompt(login):
    from types import SimpleNamespace

    prompt = {"url": "https://example.com", "code": "ABCD-EFGH"}
    with (
        patch.object(login.shutil, "which", return_value="codex"),
        patch.object(
            login.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ),
        patch.object(login, "alive", return_value=True),
        patch.object(login, "login_prompt", return_value=prompt),
    ):
        assert login.setup()["login"] == prompt


def test_failed_or_expired_login_returns_to_signed_out(login):
    from types import SimpleNamespace

    with (
        patch.object(login.shutil, "which", return_value="codex"),
        patch.object(
            login.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ),
        patch.object(login, "alive", return_value=False),
        patch.object(login, "login_prompt") as prompt,
    ):
        assert login.setup()["state"] == "signed_out"
        prompt.assert_not_called()
