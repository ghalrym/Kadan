"""Development restart policy with fake runner, tracing and signal boundaries."""

import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from kadan.runner import __main__ as runner


@pytest.fixture
def boundary(monkeypatch):
    provider = object()
    client = AsyncMock()
    values = SimpleNamespace(
        configure=Mock(return_value=provider),
        shutdown=AsyncMock(),
        serve=AsyncMock(),
        client=client,
        provider=provider,
        signals={},
    )
    monkeypatch.setattr(runner, "configure", values.configure)
    monkeypatch.setattr(runner, "shutdown", values.shutdown)
    monkeypatch.setattr(runner, "serve", values.serve)
    monkeypatch.setattr(runner, "Settings", Mock())
    monkeypatch.setattr(runner, "RunnerSettings", Mock())
    monkeypatch.setattr(runner.httpx, "AsyncClient", Mock(return_value=client))
    return values


async def run_main(monkeypatch, boundary, **kwargs):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", boundary.signals.__setitem__)
    await runner.main(**kwargs)


def test_development_returns_after_runner_stops(monkeypatch, boundary):
    asyncio.run(run_main(monkeypatch, boundary, restart_on_failure=True))

    boundary.serve.assert_awaited_once()
    boundary.shutdown.assert_awaited_once_with(boundary.provider)


def test_development_keeps_tracing_alive_across_retries(monkeypatch, boundary):
    attempts = 0

    async def serve(*args):
        nonlocal attempts
        attempts += 1
        boundary.configure.assert_called_once_with("kadan-runner")
        boundary.shutdown.assert_not_awaited()
        if attempts == 1:
            raise OSError("runner failed")

    async def delay(waiter, timeout):
        assert timeout == 5
        waiter.close()
        raise TimeoutError

    boundary.serve.side_effect = serve
    monkeypatch.setattr(runner.asyncio, "wait_for", delay)

    asyncio.run(run_main(monkeypatch, boundary, restart_on_failure=True))

    assert attempts == 2
    assert boundary.client.__aexit__.await_count == 2
    boundary.shutdown.assert_awaited_once_with(boundary.provider)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_stop_during_retry_delay_prevents_another_attempt(
    monkeypatch, boundary, signum
):
    boundary.serve.side_effect = OSError("runner failed")

    async def stop_during_delay(waiter, timeout):
        assert timeout == 5
        boundary.signals[signum]()
        await waiter

    monkeypatch.setattr(runner.asyncio, "wait_for", stop_during_delay)

    asyncio.run(run_main(monkeypatch, boundary, restart_on_failure=True))

    boundary.serve.assert_awaited_once()
    boundary.shutdown.assert_awaited_once_with(boundary.provider)


def test_production_failure_is_not_retried(monkeypatch, boundary):
    boundary.serve.side_effect = OSError("runner failed")

    with pytest.raises(OSError, match="runner failed"):
        asyncio.run(run_main(monkeypatch, boundary))

    boundary.serve.assert_awaited_once()
    boundary.shutdown.assert_awaited_once_with(boundary.provider)


@pytest.mark.parametrize(
    "stop", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_development_does_not_restart_process_stops(monkeypatch, boundary, stop):
    boundary.serve.side_effect = stop

    with pytest.raises(stop):
        asyncio.run(run_main(monkeypatch, boundary, restart_on_failure=True))

    boundary.serve.assert_awaited_once()
    boundary.shutdown.assert_awaited_once_with(boundary.provider)


def test_watchfiles_entry_point_enables_development_retries(monkeypatch):
    main = AsyncMock()
    monkeypatch.setattr(runner, "main", main)

    runner.run_development()

    main.assert_awaited_once_with(restart_on_failure=True)
