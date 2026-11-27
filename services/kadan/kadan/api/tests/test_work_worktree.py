"""Unit tests for the Git worktree helper, repository cloning, and folder browser."""

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from kadan.api.config import ServiceError
from kadan.work.worktree import WorktreeManager


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    (repo / "README.md").write_text("# Demo\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.mark.anyio
async def test_inspect_and_browse_directory(tmp_path):
    repo = tmp_path / "sample-repo"
    plain = tmp_path / "notes"
    _init_git_repo(repo)
    plain.mkdir()

    mgr = WorktreeManager(base_dir=tmp_path / "work-data")

    info = await mgr.inspect_directory(str(repo))
    assert info["is_git_repo"] is True
    assert info["default_branch"] == "main"
    assert info["name"] == "sample-repo"

    plain_info = await mgr.inspect_directory(str(plain))
    assert plain_info["is_git_repo"] is False

    browsed = await mgr.browse_directory(str(tmp_path))
    names = {entry["name"]: entry["is_git_repo"] for entry in browsed["entries"]}
    assert names["sample-repo"] is True
    assert names["notes"] is False


@pytest.mark.anyio
async def test_create_worktree_with_setup_and_cleanup(tmp_path):
    repo = tmp_path / "my-app"
    _init_git_repo(repo)
    mgr = WorktreeManager(base_dir=tmp_path / "work-data")
    thread_id = uuid4()

    wt = await mgr.create_worktree(
        repo,
        thread_id,
        base_ref="main",
        setup_script="echo setup-ok > .setup_ran && echo done",
    )
    assert wt.path.is_dir()
    assert (wt.path / "README.md").is_file()
    assert (wt.path / ".setup_ran").read_text().strip() == "setup-ok"
    assert wt.setup_output == "done"
    assert wt.branch.startswith("kadan/work-")
    assert wt.base_ref == "main"

    await mgr.remove_worktree(repo, wt.path, delete_branch=wt.branch)
    assert not wt.path.exists()
    branches = subprocess.run(
        ["git", "branch", "--list", wt.branch],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branches == ""


@pytest.mark.anyio
async def test_worktree_rejected_for_non_git_folder(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    mgr = WorktreeManager(base_dir=tmp_path / "work-data")

    with pytest.raises(ServiceError) as exc_info:
        await mgr.create_worktree(plain, uuid4())
    assert exc_info.value.code == "not_a_git_repo"


@pytest.mark.anyio
async def test_clone_local_repository(tmp_path):
    source = tmp_path / "upstream"
    _init_git_repo(source)
    mgr = WorktreeManager(base_dir=tmp_path / "work-data")

    dest, branch, title = await mgr.clone_repository(f"file://{source}")
    assert dest.is_dir()
    assert (dest / "README.md").is_file()
    assert branch == "main"
    assert title == "upstream"
