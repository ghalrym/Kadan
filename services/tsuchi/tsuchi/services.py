import hashlib
import json
import mimetypes
import os
import tempfile
import warnings
import wave
import zipfile
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from PIL import Image, UnidentifiedImageError

from .config import StorageError
from .models import Asset, Metadata, utc_now
from .repository import Repository

RASTER_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


def classify_media(path: Path, filename: str) -> tuple[str, bool]:
    """Only successfully parsed raster images become image-library entries."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                media_type = RASTER_MEDIA_TYPES.get(image.format or "")
                if media_type:
                    image.verify()
                    return media_type, True
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        pass
    guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    # A filename cannot promote an unverified raster or WAV file to inline media.
    if guessed.startswith("image/") and guessed != "image/svg+xml":
        guessed = "application/octet-stream"
    if guessed in {"audio/x-wav", "audio/wav", "audio/vnd.wave"}:
        try:
            with wave.open(str(path), "rb") as audio:
                if audio.getnchannels() > 0 and audio.getframerate() > 0:
                    audio.readframes(1)
                    return "audio/wav", False
        except (wave.Error, EOFError, OSError):
            pass
        guessed = "application/octet-stream"
    return guessed, False


def safe_filename(filename: str | None) -> str:
    # Retain the basename as display metadata only; storage never uses it.
    name = (filename or "attachment").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(character for character in name if character.isprintable())[
        :255
    ].strip()
    return name if name not in {"", ".", ".."} else "attachment"


def inline_media(asset: Asset) -> bool:
    return asset.is_image or asset.media_type == "audio/wav"


class FileStorage:
    def __init__(self, repository: Repository, max_upload_bytes: int):
        self.repository = repository
        self.max_upload_bytes = max_upload_bytes
        self.blob_dir = repository.data_dir / "files"
        self.temp_dir = repository.data_dir / "temporary"

    def initialize(self) -> None:
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def blob_path(self, asset_id: UUID) -> Path:
        return self.blob_dir / str(UUID(str(asset_id)))

    def upload(
        self,
        key: str,
        source: BinaryIO,
        filename: str | None,
        metadata: Metadata,
        asset_id: UUID | None = None,
    ) -> Asset:
        self.repository.get_agent(key)
        asset_id = asset_id or uuid4()
        target = self.blob_path(asset_id)
        digest, size = hashlib.sha256(), 0
        descriptor, name = tempfile.mkstemp(prefix="upload-", dir=self.temp_dir)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise StorageError(
                            f"Upload exceeds the {self.max_upload_bytes // (1024 * 1024)} MiB limit",
                            413,
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            filename = safe_filename(filename)
            media_type, is_image = classify_media(temporary, filename)
            now = utc_now()
            asset = Asset(
                **metadata.model_dump(),
                id=asset_id,
                agent_key=key,
                filename=filename,
                media_type=media_type,
                is_image=is_image,
                size_bytes=size,
                sha256=digest.hexdigest(),
                version=1,
                created_at=now,
                updated_at=now,
            )
            try:
                os.link(temporary, target)
            except FileExistsError:
                with target.open("rb") as existing:
                    if (
                        hashlib.file_digest(existing, "sha256").hexdigest()
                        != asset.sha256
                    ):
                        raise StorageError(
                            "This asset ID already contains different artwork", 409
                        ) from None
            directory_descriptor = os.open(self.blob_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            # Retain a published immutable blob if the database write fails. A
            # retry with the same ID/content can finish registration after a crash.
            return self.repository.create_asset(asset)
        finally:
            temporary.unlink(missing_ok=True)

    def export(self, key: str) -> Path:
        descriptor, filename = tempfile.mkstemp(
            prefix="export-", suffix=".zip", dir=self.temp_dir
        )
        os.close(descriptor)
        path = Path(filename)
        try:
            # The read transaction fixes metadata while immutable files are copied.
            with self.repository.connect() as db:
                manifest = self.repository.export_manifest(db, key)
                with zipfile.ZipFile(
                    path, "w", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    archive.writestr(
                        "manifest.json",
                        json.dumps(
                            manifest.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                    for asset in manifest.assets:
                        asset_id = asset.id
                        blob = self.blob_path(asset_id)
                        if not blob.is_file():
                            raise StorageError(
                                f"Stored file is missing for asset {asset_id}; export stopped",
                                500,
                            )
                        archive.write(blob, f"files/{asset_id}")
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise
