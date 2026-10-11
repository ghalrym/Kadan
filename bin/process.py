"""Run repository commands without a shell."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def run(arguments: list[str | Path], *, cwd: Path = ROOT) -> None:
    subprocess.run([str(argument) for argument in arguments], cwd=cwd, check=True)
