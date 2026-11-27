"""Local folder inspection and UI folder browser for Kadan native Work."""

from __future__ import annotations

from pathlib import Path

from ..api.config import ServiceError


async def inspect_directory(path_str: str) -> dict[str, str]:
    """Validate that a local project folder exists and return its normalized path and name."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ServiceError(
            "That folder could not be found on this server.",
            404,
            "folder_not_found",
        )
    return {
        "path": str(path),
        "name": path.name or str(path),
    }


async def browse_directory(path_str: str | None = None) -> dict[str, object]:
    """List visible child directories so users can pick a local folder in the web UI."""
    target = (
        Path(path_str).expanduser().resolve()
        if path_str and path_str.strip()
        else Path.home().resolve()
    )
    if not target.exists() or not target.is_dir():
        target = Path.home().resolve()
    entries: list[dict[str, str]] = []
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
            }
        )
    parent = str(target.parent) if target.parent != target else None
    return {
        "current_path": str(target),
        "parent_path": parent,
        "entries": entries,
    }
