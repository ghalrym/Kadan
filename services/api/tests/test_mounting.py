import asyncio
from contextlib import asynccontextmanager

from combined.mounting import create_app
from fastapi import FastAPI
from fastapi.testclient import TestClient


def service(name: str, events: list[str]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        events.append(f"start {name}")
        app.state.started = True
        try:
            yield
        finally:
            events.append(f"stop {name}")

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/health")
    async def health():
        return {"service": name, "started": getattr(app.state, "started", False)}

    return app


def test_each_service_is_served_under_its_own_prefix():
    events: list[str] = []
    app = create_app(
        {name: service(name, events) for name in ("tsuchi", "hikari", "kadan")}
    )
    with TestClient(app) as client:
        for name in ("tsuchi", "hikari", "kadan"):
            assert client.get(f"/{name}/api/health").json() == {
                "service": name,
                "started": True,
            }
        assert client.get("/api/health").status_code == 404


def test_mounted_lifespans_start_in_order_and_stop_in_reverse():
    events: list[str] = []
    app = create_app(
        {name: service(name, events) for name in ("tsuchi", "hikari", "kadan")}
    )

    async def run():
        async with app.router.lifespan_context(app):
            assert events == ["start tsuchi", "start hikari", "start kadan"]

    asyncio.run(run())
    assert events[3:] == ["stop kadan", "stop hikari", "stop tsuchi"]


def test_a_failed_start_stops_services_already_started():
    events: list[str] = []

    @asynccontextmanager
    async def broken(_):
        raise RuntimeError("cannot start")
        yield

    apps = {"tsuchi": service("tsuchi", events), "hikari": FastAPI(lifespan=broken)}
    app = create_app(apps)

    async def run():
        async with app.router.lifespan_context(app):
            pass

    try:
        asyncio.run(run())
    except RuntimeError:
        pass
    assert events == ["start tsuchi", "stop tsuchi"]


def test_root_well_known_oauth_discovery_forwards_to_kadan_with_prefix(
    tmp_path, monkeypatch
):
    (tmp_path / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    monkeypatch.setenv("KADAN_FRONTEND_DIR", str(tmp_path))

    kadan = FastAPI()

    @kadan.get("/.well-known/oauth-protected-resource/{resource_path:path}")
    async def prm(resource_path: str):
        return {"resource_path": resource_path}

    @kadan.get("/.well-known/oauth-authorization-server")
    async def asm():
        return {"issuer": "/kadan/api/group-chat/oauth"}

    app = create_app({"kadan": kadan})
    with TestClient(app) as client:
        res_prm = client.get(
            "/.well-known/oauth-protected-resource/kadan/mcp/group-chat"
        )
        assert res_prm.status_code == 200
        assert res_prm.json() == {"resource_path": "kadan/mcp/group-chat"}

        res_asm = client.get("/.well-known/oauth-authorization-server")
        assert res_asm.status_code == 200
        assert res_asm.json() == {"issuer": "/kadan/api/group-chat/oauth"}

        # Regular browser route still serves the SPA.
        spa = client.get("/group-chat")
        assert spa.status_code == 200
        assert "<html>spa</html>" in spa.text
