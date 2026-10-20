import asyncio
import logging
import signal

import httpx
from kadan_tracing import configure, shutdown

from ..api.config import Settings
from .config import RunnerSettings
from .service import serve

logger = logging.getLogger(__name__)


async def main() -> None:
    # OpenTelemetry's global provider can be installed only once per process.
    provider = configure("kadan-runner")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        while not stop.is_set():
            try:
                settings, runner_settings = Settings(), RunnerSettings()
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(120, connect=3),
                    follow_redirects=False,
                    trust_env=False,
                ) as client:
                    await serve(client, settings, runner_settings, stop)
                return
            except Exception:
                logger.exception("Runner crashed; restarting in five seconds.")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=5)
                except TimeoutError:
                    pass
    finally:
        await shutdown(provider)


def run() -> None:
    """Retry crashes until stopped or reloaded by watchfiles."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())


if __name__ == "__main__":
    run()
