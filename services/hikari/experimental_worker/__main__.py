import asyncio
import os
import signal

from experimental_worker.runtime import Runtime


async def main():
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    await Runtime(os.environ["HIKARI_EXPERIMENTAL_BACKEND"]).serve()


if __name__ == "__main__":
    asyncio.run(main())
