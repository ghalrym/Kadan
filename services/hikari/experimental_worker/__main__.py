import asyncio
import os

from experimental_worker.runtime import Runtime


if __name__ == "__main__":
    asyncio.run(Runtime(os.environ["HIKARI_EXPERIMENTAL_BACKEND"]).serve())
