"""Desktop launcher unit tests replace all X11 and application processes."""

import fcntl
import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "sandbox/desktop/launch.py"
SPEC = importlib.util.spec_from_file_location("desktop_launcher", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".cache").mkdir()
    with patch.object(launcher.Path, "home", return_value=tmp_path):
        yield tmp_path


def test_existing_minimized_app_is_activated_without_spawning(home):
    with (
        patch.object(
            launcher.subprocess, "run", return_value=Mock(returncode=0)
        ) as run,
        patch.object(launcher.subprocess, "Popen") as spawn,
    ):
        launcher.launch("browser")
    assert run.call_args_list[0].args[0] == ["wmctrl", "-k", "off"]
    assert run.call_args_list[1].args[0] == ["wmctrl", "-xa", "Chromium-browser"]
    spawn.assert_not_called()


def test_closed_browser_reopens_using_the_existing_profile(home):
    with (
        patch.object(launcher, "focus", side_effect=[False, False, True]),
        patch.object(launcher.time, "sleep"),
        patch.object(launcher.subprocess, "Popen") as spawn,
    ):
        launcher.launch("browser")
    spawn.assert_called_once()
    assert spawn.call_args.args[0] == [
        "/usr/local/bin/browser",
        f"--user-data-dir={home / '.config/browser'}",
        "--new-window",
        "about:blank",
    ]


def test_repeated_click_during_startup_does_not_launch_another_app(home):
    def repeat_click(*args, **kwargs):
        launcher.launch("terminal")

    with (
        patch.object(launcher, "focus", side_effect=[False, True]) as focus,
        patch.object(launcher.subprocess, "Popen", side_effect=repeat_click) as spawn,
    ):
        launcher.launch("terminal")
    spawn.assert_called_once()
    assert focus.call_count == 2


def test_startup_timeout_releases_lock_for_retry(home):
    with (
        patch.object(launcher, "focus", return_value=False),
        patch.object(launcher.time, "monotonic", side_effect=[0, 16]),
        patch.object(launcher.subprocess, "Popen"),
        pytest.raises(RuntimeError, match="Try again from Applications"),
    ):
        launcher.launch("terminal")
    with (home / ".cache/kadan-terminal.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_launch_error_releases_lock_for_retry(home):
    with (
        patch.object(launcher, "focus", return_value=False),
        patch.object(launcher.subprocess, "Popen", side_effect=OSError("unavailable")),
        pytest.raises(OSError),
    ):
        launcher.launch("terminal")
    with patch.object(launcher, "focus", return_value=True):
        launcher.launch("terminal")
