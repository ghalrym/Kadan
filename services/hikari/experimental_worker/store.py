"""Pinned snapshots acquired only by the existing explicit download request."""

import hashlib
import fcntl
import json
import os
import urllib.request
from pathlib import Path, PurePosixPath


class ModelStore:
    def __init__(self, root: Path, metadata: dict, manifest: dict):
        self.metadata = metadata
        self.artifacts = manifest["artifacts"]
        self.root = root / "experimental" / metadata["id"] / metadata["revision"]
        self.marker = self.root / ".verified.json"
        self.fingerprint = hashlib.sha256(
            json.dumps(self.artifacts, sort_keys=True).encode()
        ).hexdigest()
        for artifact in self.artifacts:
            path = PurePosixPath(artifact["name"])
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Invalid checkpoint artifact path")
            expected = (
                f"https://huggingface.co/{metadata['repo_id']}/resolve/"
                f"{metadata['revision']}/{artifact['name']}"
            )
            if artifact["url"] != expected:
                raise ValueError("Checkpoint URL does not match its pinned revision")
            if artifact["algorithm"] not in ("sha256", "git-sha1"):
                raise ValueError("Unsupported checkpoint digest")

    def complete(self) -> bool:
        try:
            return (
                json.loads(self.marker.read_text()) == {"manifest": self.fingerprint}
                and all(
                    (self.root / item["name"]).is_file()
                    and (self.root / item["name"]).stat().st_size == item["size"]
                    for item in self.artifacts
                )
            )
        except (OSError, ValueError):
            return False

    def status(self) -> dict:
        metadata = self.metadata
        total = sum(item["size"] for item in self.artifacts)
        return {
            "models": [{
                "id": metadata["id"], "repo_id": metadata["repo_id"],
                "revision": metadata["revision"], "license": metadata["license"],
                "license_url": metadata["license_url"],
                "license_notice": "Experimental LFM3 trial; review the model license before downloading.",
                "display_name": metadata.get("display_name", metadata["id"]),
                "short_name": metadata.get("short_name"),
                "kind": metadata["kind"], "estimated_bytes": total,
                "status": "complete" if self.complete() else "not_downloaded",
                "downloaded_bytes": total if self.complete() else 0,
                "total_bytes": total, "error": None,
                "inference_available": True,
                "context_limit": metadata["context_limit"],
                "architecture_context_limit": metadata.get(
                    "architecture_context_limit", metadata["context_limit"]
                ),
            }],
            "tiers": [], "selected_tier": None, "selected_model_id": None,
            "selected_whisper": None, "whisper_formatting": True,
        }

    def download(self, acknowledged: bool, emit) -> dict:
        if not acknowledged:
            raise ValueError("Review and acknowledge the model license before downloading.")
        self.root.mkdir(parents=True, exist_ok=True)
        # Replicas may share the checkpoint volume; never publish overlapping
        # writers or mistake another worker's incomplete files for a snapshot.
        with (self.root / ".download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._download(emit)

    def _download(self, emit) -> dict:
        self.marker.unlink(missing_ok=True)
        downloaded = 0
        total = sum(item["size"] for item in self.artifacts)
        for artifact in self.artifacts:
            target = self.root / artifact["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            digest = hashlib.sha256() if artifact["algorithm"] == "sha256" else hashlib.sha1()
            if artifact["algorithm"] == "git-sha1":
                digest.update(f"blob {artifact['size']}\0".encode())
            received = 0
            request = urllib.request.Request(artifact["url"])
            # Public pinned snapshots need no new credentials or login flow.
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > artifact["size"]:
                        raise ValueError("Checkpoint artifact exceeds its pinned size")
                    output.write(chunk)
                    digest.update(chunk)
                    emit({"state": "progress", "phase": "download", "model": self.metadata["id"],
                          "value": downloaded + received, "total": total})
            if received != artifact["size"] or digest.hexdigest() != artifact["digest"]:
                partial.unlink(missing_ok=True)
                raise ValueError(f"Checkpoint verification failed: {artifact['name']}")
            os.replace(partial, target)
            downloaded += received
        marker = self.marker.with_suffix(".partial")
        marker.write_text(json.dumps({"manifest": self.fingerprint}))
        os.replace(marker, self.marker)
        return self.status()
