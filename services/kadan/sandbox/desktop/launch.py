"""Focus an existing app or open it once, including during repeated clicks."""

import fcntl
import subprocess
import sys
import time
from pathlib import Path


def focus(window_class: str) -> bool:
    subprocess.run(["wmctrl", "-k", "off"], check=True)
    return subprocess.run(["wmctrl", "-xa", window_class], check=False).returncode == 0


def launch(app: str) -> None:
    window_class, command = {
        "browser": (
            "Chromium-browser",
            [
                "/usr/local/bin/browser",
                f"--user-data-dir={Path.home() / '.config/browser'}",
                "--new-window",
                "about:blank",
            ],
        ),
        "terminal": (
            "tilix",
            [
                "tilix",
                "--window-style=disable-csd",
                "--working-directory",
                str(Path.home()),
            ],
        ),
    }[app]
    # The lock lives outside the profile and is released even if launch fails.
    with (Path.home() / ".cache" / f"kadan-{app}.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if focus(window_class):
            return
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if focus(window_class):
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"{app.capitalize()} did not open. Try again from Applications."
        )


if __name__ == "__main__":
    try:
        Path.home().joinpath(".cache").mkdir(exist_ok=True)
        launch(sys.argv[1])
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        subprocess.run(
            ["xmessage", "-center", "-buttons", "Close", str(error)], check=False
        )
