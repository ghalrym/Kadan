"""Asset metadata query units with a recording database boundary."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import Mock
from uuid import UUID

import pytest
from pydantic import TypeAdapter, ValidationError
from tsuchi.models import Asset
from tsuchi.repository import Repository
from tsuchi.routes import create_router


def asset(number, **changes):
    return Asset(
        **{
            "id": UUID(int=number),
            "agent_key": "alpha",
            "title": "Weekly report",
            "filename": "report.pdf",
            "tags": ["finance"],
            "sources": [{"label": "Source", "url": "https://example.com/archive"}],
            "media_type": "application/pdf",
            "size_bytes": 42,
            "sha256": "a" * 64,
            "is_image": False,
            "version": 1,
            "created_at": "2026-10-01T00:00:00Z",
            "updated_at": "2026-10-22T00:00:00Z",
            **changes,
        }
    )


def repository(monkeypatch, assets):
    storage = Repository(Path("/unused"))
    db = SimpleNamespace(
        execute=Mock(return_value=((item.model_dump_json(),) for item in assets))
    )
    monkeypatch.setattr(storage, "connect", lambda: nullcontext(db))
    monkeypatch.setattr(storage, "_agent", Mock())
    return storage, db


def test_asset_query_scopes_active_rows_and_orders_by_update_then_id(monkeypatch):
    storage, db = repository(monkeypatch, [asset(1)])
    result = storage.list_assets("alpha", limit=20, offset=0)
    storage._agent.assert_called_once_with(db, "alpha")
    sql, values = db.execute.call_args.args
    assert "WHERE agent_key = ? AND deleted = ?" in sql
    assert "ORDER BY updated_at DESC, id" in sql
    assert values == ("alpha", 0, 0)
    assert [item.id for item in result] == [UUID(int=1)]


@pytest.mark.parametrize(
    "query",
    [
        "WEEKLY REPORT",
        "report.PDF",
        "FINANCE",
        "source archive",
        "WEEK port finance archive",
    ],
)
def test_search_matches_all_casefolded_substrings_in_existing_metadata(
    monkeypatch, query
):
    storage, _ = repository(monkeypatch, [asset(1)])
    assert [item.id for item in storage.list_assets("alpha", query=query)] == [
        UUID(int=1)
    ]


@pytest.mark.parametrize(
    "query", ["report missing", "application/pdf", '"weekly"', "semantic summary"]
)
def test_search_does_not_add_type_operators_or_content_matching(monkeypatch, query):
    storage, _ = repository(monkeypatch, [asset(1)])
    assert storage.list_assets("alpha", query=query) == []


def test_pagination_counts_matches_after_filtering_and_stops_consuming_rows(
    monkeypatch,
):
    consumed = []

    def rows():
        for number in range(1, 10):
            consumed.append(number)
            yield asset(
                number, title="Match" if number % 2 else "Skip", filename="file.txt"
            )

    storage, _ = repository(monkeypatch, rows())
    result = storage.list_assets("alpha", query="match", limit=2, offset=1)
    assert [item.id for item in result] == [UUID(int=3), UUID(int=5)]
    assert consumed == [1, 2, 3, 4, 5]


def test_unpaged_calls_still_return_all_matches_in_storage_order(monkeypatch):
    storage, _ = repository(monkeypatch, [asset(number) for number in range(1, 251)])
    assert len(storage.list_assets("alpha")) == 250


def test_page_beyond_matches_is_empty(monkeypatch):
    storage, _ = repository(monkeypatch, [asset(1)])
    assert storage.list_assets("alpha", limit=2, offset=1) == []


def test_image_and_trash_filters_remain_database_parameters(monkeypatch):
    storage, db = repository(monkeypatch, [])
    assert storage.list_assets("beta", images=True, trash=True, limit=2) == []
    assert db.execute.call_args.args[1] == ("beta", 1, 1)


@pytest.mark.parametrize("paging", [{}, {"limit": 51, "offset": 7}])
def test_asset_route_forwards_optional_paging_to_existing_query(paging):
    storage = SimpleNamespace(list_assets=Mock(return_value=[]))
    route = next(
        route
        for route in create_router(storage, Mock()).routes
        if route.path == "/api/agents/{key}/assets" and "GET" in route.methods
    )
    assert route.endpoint("alpha", q="report", **paging) == []
    storage.list_assets.assert_called_once_with(
        "alpha", False, False, "report", paging.get("limit"), paging.get("offset", 0)
    )


@pytest.mark.parametrize(
    "parameter, value", [("limit", 0), ("limit", 201), ("offset", -1)]
)
def test_asset_route_schema_rejects_invalid_pages(parameter, value):
    route = next(
        route
        for route in create_router(Mock(), Mock()).routes
        if route.path == "/api/agents/{key}/assets" and "GET" in route.methods
    )
    annotation = get_type_hints(route.endpoint, include_extras=True)[parameter]
    with pytest.raises(ValidationError):
        TypeAdapter(annotation).validate_python(value)
