import asyncio
import logging
import signal

import httpx
from kadan_tracing import configure, shutdown

from ..api.config import Settings
from .config import RunnerSettings
from .service import serve


async def main():
    settings, runner_settings = Settings(), RunnerSettings()
    provider = configure("kadan-runner")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=3),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            await serve(client, settings, runner_settings, stop)
    finally:
        await shutdown(provider)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
