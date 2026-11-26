"""Read and list file metadata in the running agent's durable Workspace."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from ..api.config import ServiceError
from ..api.pydantic_models.resources import AgentKey

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


class WorkspaceFileListing(WorkspaceFile):
    created_at: datetime
    updated_at: datetime


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

    async def list(self, *, query: str = "", limit: int = 20, offset: int = 0) -> dict:
        rows = await self.services.json_request(
            "Tsuchi",
            "GET",
            self.path,
            params={"q": query, "trash": "false", "limit": limit + 1, "offset": offset},
        )
        if not isinstance(rows, list) or len(rows) > limit + 1:
            raise ServiceError("Tsuchi returned an invalid file list.", 502)
        assets = []
        for row in rows:
            try:
                asset = WorkspaceFileListing.model_validate(row)
            except ValidationError as error:
                raise ServiceError(
                    "Tsuchi returned invalid file metadata.", 502
                ) from error
            self.checked(asset, asset.id)
            assets.append(asset)
        if len({asset.id for asset in assets}) != len(assets):
            raise ServiceError("Tsuchi returned duplicate file identities.", 502)
        return {
            "files": [
                {
                    **self.summary(asset),
                    **asset.model_dump(
                        mode="json", include={"created_at", "updated_at", "version"}
                    ),
                }
                for asset in assets[:limit]
            ],
            "next_offset": offset + limit if len(assets) > limit else None,
        }
