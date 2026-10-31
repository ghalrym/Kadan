"""Copy completed computer files into the running agent's durable Workspace."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from datetime import datetime
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..api.config import ServiceError
from ..api.pydantic_models.resources import AgentKey
from ..api.sandbox_client import MAX_FILE_BYTES, OPERATION_SECONDS

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

    async def download(self, asset: WorkspaceFile, output: BinaryIO) -> None:
        size = 0
        digest = hashlib.sha256()
        try:
            async with asyncio.timeout(OPERATION_SECONDS):
                async with self.services.client.stream(
                    "GET",
                    self.services.settings.tsuchi_url
                    + f"{self.path}/{asset.id}/content",
                    headers={"Accept-Encoding": "identity"},
                    timeout=httpx.Timeout(OPERATION_SECONDS, connect=3),
                ) as response:
                    if response.status_code != 200:
                        raise ServiceError(
                            f"Tsuchi could not download the file (HTTP {response.status_code}).",
                            502,
                        )
                    if (
                        response.headers.get("content-encoding", "identity").lower()
                        != "identity"
                    ):
                        raise ServiceError(
                            "Tsuchi returned an encoded file download.", 502
                        )
                    length = response.headers.get("content-length")
                    if length is not None:
                        try:
                            if not length.isascii() or not length.isdigit():
                                raise ValueError
                            if int(length) != asset.size_bytes:
                                raise ValueError
                        except ValueError as error:
                            raise ServiceError(
                                "Tsuchi's file length does not match its saved metadata.",
                                502,
                            ) from error
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        size += len(chunk)
                        if size > min(asset.size_bytes, MAX_FILE_BYTES):
                            raise ServiceError(
                                "Tsuchi returned more file bytes than expected.", 502
                            )
                        output.write(chunk)
                        digest.update(chunk)
                    if size != asset.size_bytes or digest.hexdigest() != asset.sha256:
                        raise ServiceError(
                            "Tsuchi's file contents do not match its saved metadata.",
                            502,
                        )
        except (httpx.RequestError, TimeoutError) as error:
            raise ServiceError(
                "Tsuchi did not complete the file download.", 503
            ) from error
        output.seek(0)

    async def existing_copy(self, asset: WorkspaceFile, path: str) -> bool:
        with TemporaryFile("w+b") as output:
            try:
                downloaded = await self.services.sandbox.download_file(path, output)
            except ServiceError as error:
                if error.code == "file_not_found":
                    return False
                raise
        if (
            downloaded.size_bytes != asset.size_bytes
            or downloaded.sha256 != asset.sha256
        ):
            raise ServiceError(
                f"The computer file at {path} differs from the saved file. It has been left unchanged.",
                409,
            )
        return True

    async def open(
        self,
        file_id: UUID,
        action_id: UUID,
        check_ownership: Callable[[], Awaitable[object]],
        *,
        replayed: bool,
    ) -> dict:
        await check_ownership()
        value = await self.services.json_request(
            "Tsuchi", "GET", f"{self.path}/{file_id}"
        )
        asset = self.checked(value, file_id)
        if asset.size_bytes > MAX_FILE_BYTES:
            raise ServiceError("Files cannot exceed 100 MiB.", 413, "file_too_large")
        name = (
            file_name(asset.filename)
            .encode("utf-8")[:160]
            .decode("utf-8", errors="ignore")
        )
        path = f"/tmp/kadan-workspace-{action_id}-{name}"
        result = {"opened": True, "path": path, **self.summary(asset)}
        if replayed:
            try:
                if await self.existing_copy(asset, path):
                    return result
                raise ServiceError("The computer copy is missing.", 404)
            except ServiceError as error:
                raise ServiceError(
                    f"The earlier copy to {path} could not be confirmed. Nothing was uploaded again. "
                    + str(error),
                    error.status,
                ) from error
        with TemporaryFile("w+b") as source:
            await self.download(asset, source)
            if await self.existing_copy(asset, path):
                return result
            await check_ownership()
            upload_error = None
            try:
                await self.services.sandbox.upload_file(path, source)
            except (ServiceError, OSError) as error:
                upload_error = error
            try:
                if not await self.existing_copy(asset, path):
                    raise ServiceError(
                        "The computer copy is missing after upload.", 502
                    )
            except (ServiceError, OSError) as error:
                raise ServiceError(
                    f"The copy to {path} is unconfirmed and may have completed. "
                    "Inspect that path before starting another copy. "
                    + str(upload_error or error),
                    getattr(error, "status", 503),
                ) from error
        return result

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
