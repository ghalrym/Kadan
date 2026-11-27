"""Unit tests for Kadan Desktop local Work sidecar server."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[4]
for extra in (ROOT / "services" / "tsuchi", ROOT / "services" / "kadan"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from kadan.api.config import Settings as KadanSettings  # noqa: E402
from kadan.work.desktop_server import (  # noqa: E402
    create_desktop_work_app,
    derive_upstream_urls,
)
from tsuchi.config import Settings as TsuchiSettings  # noqa: E402
from tsuchi.main import create_app as create_tsuchi_app  # noqa: E402


def test_derive_upstream_urls():
    assert derive_upstream_urls("http://192.168.0.19:8000/") == (
        "http://192.168.0.19:8000/tsuchi",
        "http://192.168.0.19:8000/hikari",
    )
    assert derive_upstream_urls("192.168.0.19:8000") == (
        "http://192.168.0.19:8000/tsuchi",
        "http://192.168.0.19:8000/hikari",
    )


@pytest.mark.anyio
async def test_desktop_work_server_health_browse_and_upstream_retarget(tmp_path):
    tsuchi_app = create_tsuchi_app(TsuchiSettings(data_dir=tmp_path / "tsuchi"))
    sample_project = tmp_path / "my-local-repo"
    sample_project.mkdir()

    async with (
        tsuchi_app.router.lifespan_context(tsuchi_app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tsuchi_app), base_url="http://tsuchi"
        ) as tsuchi_client,
    ):
        desktop_app = create_desktop_work_app(
            KadanSettings(tsuchi_url="http://tsuchi", hikari_url="http://hikari"),
            start_runner=False,
            http_client=tsuchi_client,
        )
        async with (
            desktop_app.router.lifespan_context(desktop_app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=desktop_app),
                base_url="http://desktop-work",
            ) as api,
        ):
            health = (await api.get("/api/work/health")).json()
            assert health["status"] == "ok"
            assert health["mode"] == "desktop"
            assert health["tsuchi_url"] == "http://tsuchi"

            browsed = (
                await api.get(
                    "/api/work/filesystem/browse", params={"path": str(tmp_path)}
                )
            ).json()
            assert any(e["name"] == "my-local-repo" for e in browsed["entries"])

            retargeted = (
                await api.put(
                    "/api/work/upstream",
                    json={"active_host": "http://10.0.0.5:8630/"},
                )
            ).json()
            assert retargeted["tsuchi_url"] == "http://10.0.0.5:8630/tsuchi"
            assert retargeted["hikari_url"] == "http://10.0.0.5:8630/hikari"
