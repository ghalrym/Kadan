"""Unit tests for the local folder inspection and UI folder browser helper."""

from __future__ import annotations

import pytest

from kadan.api.config import ServiceError
from kadan.work.filesystem import browse_directory, inspect_directory


@pytest.mark.anyio
async def test_inspect_and_browse_directory(tmp_path):
    project_a = tmp_path / "alpha"
    project_b = tmp_path / "beta"
    hidden = tmp_path / ".hidden"
    file_item = tmp_path / "readme.txt"
    project_a.mkdir()
    project_b.mkdir()
    hidden.mkdir()
    file_item.write_text("hi")

    info = await inspect_directory(str(project_a))
    assert info["path"] == str(project_a.resolve())
    assert info["name"] == "alpha"

    browsed = await browse_directory(str(tmp_path))
    assert browsed["current_path"] == str(tmp_path.resolve())
    names = [entry["name"] for entry in browsed["entries"]]
    assert names == ["alpha", "beta"]


@pytest.mark.anyio
async def test_inspect_missing_folder_raises_404(tmp_path):
    with pytest.raises(ServiceError) as exc_info:
        await inspect_directory(str(tmp_path / "does-not-exist"))
    assert exc_info.value.status == 404
    assert exc_info.value.code == "folder_not_found"
