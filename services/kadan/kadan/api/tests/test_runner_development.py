"""Development restart policy with a fake runner and no child processes."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from kadan.runner import __main__ as runner


def test_development_returns_after_runner_stops(monkeypatch):
    main = AsyncMock()
    sleep = Mock()
    monkeypatch.setattr(runner, "main", main)
    monkeypatch.setattr(runner.time, "sleep", sleep)

    runner.run_development()

    main.assert_awaited_once()
    sleep.assert_not_called()


def test_development_retries_runner_crashes_after_delay(monkeypatch):
    main = AsyncMock(side_effect=[OSError("runner failed"), None])
    sleep = Mock()
    monkeypatch.setattr(runner, "main", main)
    monkeypatch.setattr(runner.time, "sleep", sleep)

    runner.run_development()

    assert main.await_count == 2
    sleep.assert_called_once_with(5)


@pytest.mark.parametrize(
    "stop", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_development_does_not_restart_process_stops(monkeypatch, stop):
    main = AsyncMock(side_effect=stop)
    sleep = Mock()
    monkeypatch.setattr(runner, "main", main)
    monkeypatch.setattr(runner.time, "sleep", sleep)

    with pytest.raises(stop):
        runner.run_development()

    main.assert_awaited_once()
    sleep.assert_not_called()
