import io
import json
import wave
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from tsuchi.config import StorageError
from tsuchi.models import Agent, Asset, ExportManifest, Metadata, utc_now
from tsuchi.services import FileStorage, classify_media, safe_filename


def test_png_bytes_are_classified_independently_of_filename(tmp_path):
    path = tmp_path / "blob"
    Image.new("RGB", (2, 2)).save(path, format="PNG")
    assert classify_media(path, "wrong.html") == ("image/png", True)


@pytest.mark.parametrize(
    "filename", ["image.png", "image.jpg", "image.webp", "image.gif"]
)
def test_fake_image_extension_cannot_enter_the_image_library(tmp_path, filename):
    path = tmp_path / "blob"
    path.write_text("<html>Not an image</html>")
    assert classify_media(path, filename) == ("application/octet-stream", False)


def test_svg_stays_a_downloadable_attachment(tmp_path):
    path = tmp_path / "blob"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    assert classify_media(path, "image.svg") == ("image/svg+xml", False)


def test_wav_requires_a_parseable_audio_file(tmp_path):
    path = tmp_path / "blob"
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 8)
    assert classify_media(path, "recording.wav") == ("audio/wav", False)
    path.write_text("<html>Not audio</html>")
    assert classify_media(path, "recording.wav") == ("application/octet-stream", False)


@pytest.mark.parametrize(
    "filename", ["../../secret.txt", "C:\\somewhere\\secret.txt", "secret.txt\r\n"]
)
def test_untrusted_filename_cannot_become_a_path(filename):
    assert safe_filename(filename) == "secret.txt"


class UploadRepository:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.assets = []

    def get_agent(self, key):
        if key != "alpha":
            raise StorageError("Agent namespace not found", 404)

    def create_asset(self, asset):
        self.assets.append(asset)
        return asset


def test_upload_enforces_limit_and_removes_incomplete_blob(tmp_path):
    repository = UploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=3)
    files.initialize()
    with pytest.raises(StorageError) as error:
        files.upload("alpha", io.BytesIO(b"abcd"), "file.bin", Metadata(title="File"))
    assert error.value.status_code == 413
    assert list(files.blob_dir.iterdir()) == []
    assert repository.assets == []


def test_upload_records_exact_bytes_and_immutable_uuid_path(tmp_path):
    repository = UploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=10)
    files.initialize()
    asset = files.upload(
        "alpha", io.BytesIO(b"abc"), "../../file.txt", Metadata(title="File")
    )
    assert files.blob_path(asset.id).read_bytes() == b"abc"
    assert asset.size_bytes == 3
    assert asset.filename == "file.txt"
    assert (
        asset.sha256
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


class ExportRepository:
    def __init__(self, data_dir, manifest):
        self.data_dir, self.manifest = data_dir, manifest

    @contextmanager
    def connect(self):
        yield SimpleNamespace()

    def export_manifest(self, db, key) -> ExportManifest:
        assert key == "alpha"
        return self.manifest


def manifest_with_asset(asset_id):
    now = utc_now()
    return ExportManifest(
        exported_at=now,
        agent=Agent(key="alpha", name="Alpha", created_at=now),
        assets=[
            Asset(
                id=asset_id,
                agent_key="alpha",
                title="File",
                filename="file.bin",
                media_type="application/octet-stream",
                is_image=False,
                size_bytes=13,
                sha256="abc",
                version=1,
                created_at=now,
                updated_at=now,
            )
        ],
    )


def test_export_writes_manifest_and_exact_asset_bytes(tmp_path):
    asset_id = uuid4()
    manifest = manifest_with_asset(asset_id)
    files = FileStorage(ExportRepository(tmp_path, manifest), max_upload_bytes=100)
    files.initialize()
    files.blob_path(asset_id).write_bytes(b"durable bytes")
    path = files.export("alpha")
    try:
        with zipfile.ZipFile(path) as archive:
            assert json.loads(archive.read("manifest.json")) == manifest.model_dump(
                mode="json"
            )
            assert archive.read(f"files/{asset_id}") == b"durable bytes"
    finally:
        path.unlink()


def test_failed_export_removes_temporary_archive(tmp_path):
    manifest = manifest_with_asset(uuid4())
    files = FileStorage(ExportRepository(tmp_path, manifest), max_upload_bytes=100)
    files.initialize()
    with pytest.raises(StorageError, match="missing"):
        files.export("alpha")
    assert list(files.temp_dir.iterdir()) == []
