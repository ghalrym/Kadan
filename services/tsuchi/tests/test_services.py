import io
import json
import wave
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4, uuid5

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


class ImageUploadRepository(UploadRepository):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.job = SimpleNamespace(id=uuid4(), upload_status="pending")

    def get_image_job(self, key, job_id):
        assert key == "alpha" and job_id == self.job.id
        return self.job

    def fail_image_upload(self, key, job_id):
        job = self.get_image_job(key, job_id)
        if job.upload_status == "pending":
            job.upload_status = "failed"
        return job

    def create_asset(self, asset, image_job_id):
        job = self.get_image_job(asset.agent_key, image_job_id)
        assert job.upload_status in {"pending", "completed"}
        job.upload_status = "completed"
        return super().create_asset(asset)


def test_live_upload_is_pending_even_without_metadata_and_after_storage_recreation(
    tmp_path,
):
    repository = ImageUploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=1000)
    files.initialize()
    restarted = FileStorage(repository, max_upload_bytes=1000)
    job = repository.job
    raw = io.BytesIO()
    Image.new("RGB", (2, 2)).save(raw, format="PNG")

    class Source(io.BytesIO):
        def read(self, size):
            assert (
                restarted.resolve_image_upload("alpha", job.id).upload_status
                == "pending"
            )
            assert repository.assets == []
            return super().read(size)

    asset = files.upload(
        "alpha",
        Source(raw.getvalue()),
        "image.png",
        Metadata(title="Image"),
        uuid5(job.id, "image"),
        job.id,
    )
    assert restarted.resolve_image_upload("alpha", job.id).upload_status == "completed"
    assert files.blob_path(asset.id).read_bytes() == raw.getvalue()


def test_storage_restart_closes_abandoned_upload_and_rejects_delayed_request(tmp_path):
    repository = ImageUploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=1000)
    files.initialize()
    job = repository.job
    # A lock file left by a dead uploader is not evidence that it is still live.
    with files._image_upload_lock(job.id):
        pass
    restarted = FileStorage(repository, max_upload_bytes=1000)
    assert restarted.resolve_image_upload("alpha", job.id).upload_status == "failed"
    with pytest.raises(StorageError, match="closed"):
        files.upload(
            "alpha",
            io.BytesIO(b"late bytes"),
            "image.png",
            Metadata(title="Image"),
            uuid5(job.id, "image"),
            job.id,
        )
    assert repository.assets == []
    assert list(files.blob_dir.iterdir()) == []


def test_upload_error_confirms_failure_and_releases_storage_lock(tmp_path):
    repository = ImageUploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=3)
    files.initialize()
    job = repository.job
    with pytest.raises(StorageError, match="limit"):
        files.upload(
            "alpha",
            io.BytesIO(b"too large"),
            "image.png",
            Metadata(title="Image"),
            uuid5(job.id, "image"),
            job.id,
        )
    assert job.upload_status == "failed"
    with files._image_upload_lock(job.id, blocking=False) as acquired:
        assert acquired
    assert list(files.temp_dir.glob("upload-*")) == []


def test_replayed_upload_error_cannot_downgrade_confirmed_success(tmp_path):
    repository = ImageUploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=1000)
    files.initialize()
    job = repository.job
    asset_id = uuid5(job.id, "image")
    raw = io.BytesIO()
    Image.new("RGB", (2, 2)).save(raw, format="PNG")
    first = files.upload(
        "alpha",
        io.BytesIO(raw.getvalue()),
        "file.bin",
        Metadata(title="Image"),
        asset_id,
        job.id,
    )
    with pytest.raises(StorageError, match="different artwork"):
        files.upload(
            "alpha",
            io.BytesIO(b"different bytes"),
            "file.bin",
            Metadata(title="Image"),
            asset_id,
            job.id,
        )
    assert job.upload_status == "completed"
    assert files.blob_path(first.id).read_bytes() == raw.getvalue()
    assert repository.assets == [first]


def test_registration_failure_keeps_pending_if_storage_cannot_confirm_failure(
    tmp_path, monkeypatch
):
    repository = ImageUploadRepository(tmp_path)
    files = FileStorage(repository, max_upload_bytes=1000)
    files.initialize()
    job = repository.job
    raw = io.BytesIO()
    Image.new("RGB", (2, 2)).save(raw, format="PNG")

    def unavailable(*args):
        raise RuntimeError("Database unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "create_asset", unavailable)
        patch.setattr(repository, "fail_image_upload", unavailable)
        with pytest.raises(RuntimeError, match="Database unavailable"):
            files.upload(
                "alpha",
                io.BytesIO(raw.getvalue()),
                "image.png",
                Metadata(title="Image"),
                uuid5(job.id, "image"),
                job.id,
            )
    assert job.upload_status == "pending"
    assert files.blob_path(uuid5(job.id, "image")).read_bytes() == raw.getvalue()
    restarted = FileStorage(repository, max_upload_bytes=1000)
    assert restarted.resolve_image_upload("alpha", job.id).upload_status == "failed"
