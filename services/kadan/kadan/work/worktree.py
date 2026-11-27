"""Git worktree lifecycle, repository cloning, and UI folder browser for native Work."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from ..api.config import ServiceError

GIT_TIMEOUT_SECONDS = 120
SETUP_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class WorktreeInfo:
    path: Path
    branch: str
    base_ref: str
    setup_output: str | None = None


def _sanitize_slug(raw: str, fallback: str = "project") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip("-.")
    return cleaned[:64] or fallback


def _default_work_base_dir() -> Path:
    configured = os.environ.get("KADAN_WORK_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".kadan" / "work").resolve()


async def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = GIT_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ServiceError(
            "Git is not installed on this server, so repository branches cannot be managed.",
            503,
            "git_unavailable",
        ) from exc
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise ServiceError("Git operation timed out.", 504, "git_timeout") from exc
    return (
        proc.returncode or 0,
        stdout_b.decode(errors="replace").strip(),
        stderr_b.decode(errors="replace").strip(),
    )


class WorktreeManager:
    """Manages local project inspection, repo cloning, and isolated Git worktrees."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = (base_dir or _default_work_base_dir()).resolve()
        self.repos_dir = self.base_dir / "repos"
        self.worktrees_dir = self.base_dir / "worktrees"

    async def inspect_directory(self, path_str: str) -> dict[str, object]:
        path = Path(path_str).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise ServiceError(
                "That folder could not be found on this server.",
                404,
                "folder_not_found",
            )
        is_git = False
        branch: str | None = None
        rc, out, _ = await _run_git(["rev-parse", "--is-inside-work-tree"], cwd=path)
        if rc == 0 and out == "true":
            is_git = True
            branch = await self.detect_default_branch(path)
        return {
            "path": str(path),
            "name": path.name or str(path),
            "is_git_repo": is_git,
            "default_branch": branch or "main",
        }

    async def detect_default_branch(self, repo_path: Path) -> str:
        rc, out, _ = await _run_git(
            ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=repo_path
        )
        if rc == 0 and out:
            return out.removeprefix("origin/").strip() or "main"
        rc, out, _ = await _run_git(["branch", "--show-current"], cwd=repo_path)
        if rc == 0 and out:
            return out.strip()
        for candidate in ("main", "master"):
            rc, _, _ = await _run_git(
                ["rev-parse", "--verify", f"refs/heads/{candidate}"], cwd=repo_path
            )
            if rc == 0:
                return candidate
        return "main"

    async def browse_directory(self, path_str: str | None = None) -> dict[str, object]:
        target = (
            Path(path_str).expanduser().resolve()
            if path_str and path_str.strip()
            else Path.home().resolve()
        )
        if not target.exists() or not target.is_dir():
            target = Path.home().resolve()
        entries: list[dict[str, object]] = []
        try:
            children = sorted(
                (entry for entry in target.iterdir() if entry.is_dir()),
                key=lambda p: (p.name.startswith("."), p.name.lower()),
            )
        except OSError as exc:
            raise ServiceError(
                "Kadan does not have permission to open that folder.",
                403,
                "folder_unreadable",
            ) from exc
        for child in children[:200]:
            if child.name.startswith("."):
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_git_repo": (child / ".git").exists(),
                }
            )
        parent = str(target.parent) if target.parent != target else None
        is_git = (target / ".git").exists()
        default_branch = await self.detect_default_branch(target) if is_git else None
        return {
            "current_path": str(target),
            "parent_path": parent,
            "is_git_repo": is_git,
            "default_branch": default_branch,
            "entries": entries,
        }

    async def clone_repository(
        self, url: str, *, title: str | None = None
    ) -> tuple[Path, str, str]:
        cleaned_url = url.strip()
        if not cleaned_url:
            raise ServiceError(
                "Enter a repository address to clone.", 422, "invalid_clone_url"
            )
        parsed = urlsplit(cleaned_url)
        is_ssh = cleaned_url.startswith("git@") and ":" in cleaned_url
        if not is_ssh and parsed.scheme not in {"http", "https", "ssh", "file"}:
            raise ServiceError(
                "Use an HTTPS, SSH, or Git repository address.",
                422,
                "invalid_clone_url",
            )
        raw_name = (
            title.strip()
            if title and title.strip()
            else cleaned_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        )
        slug = _sanitize_slug(raw_name, "repo")
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        dest = self.repos_dir / slug
        suffix = 1
        while dest.exists():
            suffix += 1
            dest = self.repos_dir / f"{slug}-{suffix}"
        rc, _, stderr = await _run_git(["clone", cleaned_url, str(dest)])
        if rc != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise ServiceError(
                f"Could not clone that repository: {stderr or 'git clone failed.'}",
                422,
                "clone_failed",
            )
        default_branch = await self.detect_default_branch(dest)
        display_title = title.strip() if title and title.strip() else raw_name or slug
        return dest.resolve(), default_branch, display_title

    async def create_worktree(
        self,
        project_root: Path,
        thread_id: UUID,
        *,
        base_ref: str | None = None,
        branch: str | None = None,
        setup_script: str | None = None,
    ) -> WorktreeInfo:
        root = project_root.expanduser().resolve()
        if not root.is_dir():
            raise ServiceError(
                "The project folder no longer exists on disk.",
                404,
                "project_missing",
            )
        rc, out, _ = await _run_git(["rev-parse", "--is-inside-work-tree"], cwd=root)
        if rc != 0 or out != "true":
            raise ServiceError(
                "Isolated branches require a Git repository. Switch this thread to use the project folder directly.",
                422,
                "not_a_git_repo",
            )
        resolved_base = (base_ref or "").strip() or await self.detect_default_branch(
            root
        )
        short_id = thread_id.hex[:8]
        resolved_branch = (
            branch.strip() if branch and branch.strip() else f"kadan/work-{short_id}"
        )
        project_slug = _sanitize_slug(root.name, "project")
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = (self.worktrees_dir / f"{project_slug}-{short_id}").resolve()

        # Verify the requested base ref exists; fall back to HEAD if needed.
        rc, _, _ = await _run_git(["rev-parse", "--verify", resolved_base], cwd=root)
        start_point = resolved_base if rc == 0 else "HEAD"

        # Check if branch already exists; reuse or create new branch.
        rc_branch, _, _ = await _run_git(
            ["rev-parse", "--verify", f"refs/heads/{resolved_branch}"], cwd=root
        )
        if rc_branch == 0:
            add_args = ["worktree", "add", str(worktree_path), resolved_branch]
        else:
            add_args = [
                "worktree",
                "add",
                "-b",
                resolved_branch,
                str(worktree_path),
                start_point,
            ]
        rc, _, stderr = await _run_git(add_args, cwd=root)
        if rc != 0:
            raise ServiceError(
                f"Could not create an isolated branch copy: {stderr or 'git worktree add failed.'}",
                422,
                "worktree_create_failed",
            )

        setup_output: str | None = None
        if setup_script and setup_script.strip():
            setup_output = await self._run_setup_script(worktree_path, setup_script)

        return WorktreeInfo(
            path=worktree_path,
            branch=resolved_branch,
            base_ref=start_point,
            setup_output=setup_output,
        )

    async def _run_setup_script(self, worktree_path: Path, script: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            script,
            cwd=str(worktree_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout_b, _ = await asyncio.wait_for(
                proc.communicate(), timeout=SETUP_TIMEOUT_SECONDS
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ServiceError(
                "Project setup steps timed out while preparing the workspace.",
                504,
                "setup_timeout",
            ) from exc
        output = stdout_b.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise ServiceError(
                f"Project setup steps failed: {output[-1000:] or f'exit code {proc.returncode}'}",
                422,
                "setup_failed",
            )
        return output[-4000:]

    async def remove_worktree(
        self,
        project_root: Path,
        worktree_path: Path,
        *,
        delete_branch: str | None = None,
    ) -> None:
        root = project_root.expanduser().resolve()
        wt = worktree_path.expanduser().resolve()
        # Safety: never delete the project root itself.
        if wt == root or root in wt.parents and wt == root:
            return
        if wt.exists() and root.is_dir():
            rc, _, _ = await _run_git(
                ["worktree", "remove", "--force", str(wt)], cwd=root
            )
            if rc != 0 and wt.exists():
                shutil.rmtree(wt, ignore_errors=True)
                await _run_git(["worktree", "prune"], cwd=root)
        elif wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        if delete_branch and delete_branch.startswith("kadan/work-") and root.is_dir():
            await _run_git(["branch", "-D", delete_branch], cwd=root)
