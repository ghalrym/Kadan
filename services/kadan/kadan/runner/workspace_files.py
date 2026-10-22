"""Copy completed computer files into the running agent's durable Workspace."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from tempfile import TemporaryFile
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from ..api.config import ServiceError
from ..api.pydantic_models.resources import AgentKey
from ..api.sandbox_client import OPERATION_SECONDS

if TYPE_CHECKING:
    from ..api.services import LocalServices


class WorkspaceFile(BaseModel):
    id: UUID
    agent_key: AgentKey
    title: str = Field(min_length=1, max_length=500)
    filename: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    version: int = Field(ge=1, strict=True)
    deleted_at: datetime | None


def file_name(path: str) -> str:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(character for character in name if character.isprintable())[
        :255
    ].strip()
    # Multipart encodes quotes as %22; use a stable display name at both ends.
    name = name.replace('"', "_")
    return name if name not in {"", ".", ".."} else "attachment"


class WorkspaceFiles:
    def __init__(self, services: LocalServices, agent_key: str):
        self.services = services
        self.agent_key = agent_key
        self.path = f"/api/agents/{agent_key}/assets"

    def checked(self, value: object, file_id: UUID) -> WorkspaceFile:
        try:
            asset = WorkspaceFile.model_validate(value)
        except ValidationError as error:
            raise ServiceError("Tsuchi returned invalid file metadata.", 502) from error
        if asset.id != file_id or asset.agent_key != self.agent_key:
            raise ServiceError("Tsuchi returned a different Workspace file.", 502)
        if asset.deleted_at is not None:
            raise ServiceError(
                "This file is in Tsuchi Trash. Restore it there before continuing.", 409
            )
        return asset

    def summary(self, asset: WorkspaceFile) -> dict:
        return {
            "file_id": str(asset.id),
            **asset.model_dump(
                mode="json",
                include={"title", "filename", "media_type", "size_bytes", "sha256"},
            ),
            "download_url": f"{self.path}/{asset.id}/content?download=true",
        }

    async def read(self, file_id: UUID) -> dict:
        value = await self.services.json_request(
            "Tsuchi", "GET", f"{self.path}/{file_id}"
        )
        return self.summary(self.checked(value, file_id))

    async def save(
        self,
        file_id: UUID,
        source_path: str,
        title: str | None,
        check_ownership: Callable[[], Awaitable[object]],
        *,
        replayed: bool,
    ) -> dict:
        await check_ownership()
        if replayed:
            try:
                return {"saved": True, **await self.read(file_id)}
            except ServiceError as error:
                raise ServiceError(
                    "The earlier file save could not be confirmed. The source has not been read or uploaded again. "
                    "Use read_workspace_file with this file_id to check before starting another save. "
                    + str(error),
                    error.status,
                ) from error
        filename = file_name(source_path)
        with TemporaryFile("w+b") as output:
            downloaded = await self.services.sandbox.download_file(source_path, output)
            await check_ownership()
            try:
                async with asyncio.timeout(OPERATION_SECONDS):
                    value = await self.services.json_request(
                        "Tsuchi",
                        "POST",
                        self.path,
                        timeout=OPERATION_SECONDS,
                        files={"file": (filename, output, "application/octet-stream")},
                        data={"title": title or filename, "asset_id": str(file_id)},
                    )
                asset = self.checked(value, file_id)
                if (
                    asset.filename != filename
                    or asset.size_bytes != downloaded.size_bytes
                    or asset.sha256 != downloaded.sha256
                ):
                    raise ServiceError(
                        "Tsuchi did not confirm the complete file contents.", 502
                    )
            except (ServiceError, TimeoutError) as error:
                raise ServiceError(
                    "The file save is unconfirmed and may have completed. Use read_workspace_file "
                    "with this file_id before trying another save. " + str(error),
                    getattr(error, "status", 503),
                ) from error
        return {"saved": True, **self.summary(asset)}
