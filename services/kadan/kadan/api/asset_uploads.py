"""Bounded, retryable uploads from Kadan's browser into Tsuchi."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import UploadFile
from pydantic import ValidationError

from .config import ServiceError
from .pydantic_models.assets import Asset

if TYPE_CHECKING:
    from .services import LocalServices

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_SECONDS = 300


def upload_filename(filename: str | None) -> str:
    name = (filename or "attachment").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(character for character in name if character.isprintable())[
        :255
    ].strip()
    # Avoid multipart's percent-encoding of quotes changing the saved name.
    name = name.replace('"', "_")
    return name if name not in {"", ".", ".."} else "attachment"


async def upload_asset(
    services: LocalServices,
    key: str,
    asset_id: UUID,
    source: UploadFile,
    title: str | None,
    folder_id: UUID | None = None,
) -> Asset:
    if source.size is not None and source.size > MAX_UPLOAD_BYTES:
        raise ServiceError("Files cannot exceed 100 MiB.", 413, "file_too_large")
    size = 0
    digest = hashlib.sha256()
    try:
        await source.seek(0)
        while chunk := await source.read(64 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ServiceError(
                    "Files cannot exceed 100 MiB.", 413, "file_too_large"
                )
            digest.update(chunk)
        await source.seek(0)
    except OSError as error:
        raise ServiceError("Kadan could not read the uploaded file.", 503) from error
    filename = upload_filename(source.filename)
    data = {"asset_id": str(asset_id), "title": title or filename}
    if folder_id is not None:
        data["folder_id"] = str(folder_id)
    try:
        async with asyncio.timeout(UPLOAD_SECONDS):
            value = await services.json_request(
                "Tsuchi",
                "POST",
                f"/api/agents/{key}/assets",
                timeout=UPLOAD_SECONDS,
                files={"file": (filename, source.file, "application/octet-stream")},
                data=data,
            )
        asset = Asset.model_validate(value)
    except ServiceError as error:
        if (
            folder_id is not None
            and error.status == 400
            and "Folder must" in str(error)
        ):
            raise ServiceError(
                "The folder you chose no longer exists.", 404, "not_found"
            ) from error
        raise
    except (TimeoutError, OSError, ValidationError) as error:
        raise ServiceError(
            "The upload could not be confirmed. Check its asset ID before retrying.",
            502,
            "unconfirmed_upload",
        ) from error
    if (
        asset.id != asset_id
        or asset.agent_key != key
        or asset.deleted_at is not None
        or asset.size_bytes != size
        or asset.sha256 != digest.hexdigest()
        or asset.filename != filename
    ):
        raise ServiceError(
            "Tsuchi did not confirm the uploaded file. Check its asset ID before retrying.",
            502,
            "unconfirmed_upload",
        )
    return asset
