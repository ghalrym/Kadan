"""An in-memory stand-in for Tsuchi's resource API used by chat and Space routes."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from kadan.api.config import Settings
from kadan.api.services import LocalServices

BASE = "/api/agents/alpha/resources"
START = datetime(2026, 11, 1, tzinfo=UTC)


class FakeTsuchi:
    def __init__(self):
        self.resources: dict[str, dict] = {}
        self.requests: list[tuple[str, str, dict, object]] = []
        self.clock = 0
        # Called with the resource ID before each PUT or DELETE, as a concurrent writer.
        self.before_write = None

    def tick(self) -> str:
        self.clock += 1
        return (START + timedelta(minutes=self.clock)).isoformat()

    def add(self, kind="conversation", title="New chat", **changes) -> dict:
        now = self.tick()
        record = {
            "id": str(uuid4()),
            "agent_key": "alpha",
            "kind": kind,
            "title": title,
            "markdown": "",
            "parent_id": None,
            "tags": [],
            "sources": [],
            "attachment_ids": [],
            "deck": None,
            "messages": [],
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "deleted_at": None,
            **changes,
        }
        self.resources[record["id"]] = record
        return record

    def message(self, record: dict, role: str, content: str) -> None:
        record["messages"].append(
            {
                "id": str(uuid4()),
                "role": role,
                "content": content,
                "created_at": START.isoformat(),
                "attachment_ids": [],
                "resource_ids": [],
                "sources": [],
            }
        )
        record["version"] += 1
        record["updated_at"] = self.tick()

    def writes(self, method: str) -> list[str]:
        return [path for verb, path, _, _ in self.requests if verb == method]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        self.requests.append((method, path, dict(request.url.params), body))
        if path == BASE and method == "GET":
            kinds = request.url.params.get_list("kinds")
            rows = [
                record
                for record in self.resources.values()
                if record["deleted_at"] is None and record["kind"] in kinds
            ]
            rows.sort(key=lambda record: record["id"])
            rows.sort(key=lambda record: record["updated_at"], reverse=True)
            offset = int(request.url.params["offset"])
            limit = int(request.url.params["limit"])
            return httpx.Response(200, json=rows[offset : offset + limit])
        if path == BASE and method == "POST":
            return httpx.Response(201, json=self.add(**body))
        record = self.resources.get(path.removeprefix(BASE + "/"))
        if record is None:
            return httpx.Response(
                404, json={"detail": "Resource not found in this namespace"}
            )
        if method == "GET":
            return httpx.Response(200, json=record)
        if self.before_write is not None:
            self.before_write(record["id"])
        expected = (
            body.pop("expected_version")
            if method == "PUT"
            else int(request.url.params["expected_version"])
        )
        if expected != record["version"]:
            return httpx.Response(
                409,
                json={"detail": "This item changed. Reload it before saving again."},
            )
        if method == "PUT":
            record.update(body)
        else:
            if any(
                child["parent_id"] == record["id"] and child["deleted_at"] is None
                for child in self.resources.values()
            ):
                return httpx.Response(
                    409,
                    json={"detail": "Move or trash this Space's active children first"},
                )
            record["deleted_at"] = self.tick()
        record["version"] += 1
        record["updated_at"] = self.tick()
        return httpx.Response(200, json=record)


@pytest.fixture
def tsuchi():
    return FakeTsuchi()


@pytest.fixture
def call(tsuchi):
    """Run a route coroutine against real Kadan services and the fake Tsuchi."""

    def run(route, *args):
        async def operation():
            transport = httpx.MockTransport(tsuchi)
            async with httpx.AsyncClient(transport=transport) as client:
                services = LocalServices(client, Settings(_env_file=None))
                request = SimpleNamespace(
                    app=SimpleNamespace(state=SimpleNamespace(services=services))
                )
                return await route(*args, request)

        return asyncio.run(operation())

    return run
