"""Kadan-owned model lifecycle; no external inference server or engine process."""
import asyncio
from contextlib import suppress
import os
import threading

from api.inference.resources import ResourceManager, probe_memory


class RuntimeFailure(Exception):
    def __init__(self, detail: str, status_code: int = 503):
        super().__init__(detail)
        self.status_code = status_code


class RuntimeManager:
    def __init__(self, factory=None, resources=None):
        self.state = 'unloaded'
        self.model_id = None
        self.error = None
        self.adapter = None
        self.resources = resources
        self._factory = factory
        self._leased = False
        self._cancel = threading.Event()
        self._transition = asyncio.Lock()
        self._generation = asyncio.Lock()
        self.task = None
        self._worker = None

    def status(self):
        state = self.state
        if state == 'ready' and self.adapter is not None and not self.adapter.is_resident:
            state = 'offloaded'
        return dict(state=state, model_id=self.model_id, error=self.error,
                    memory=self.resources.snapshot() if self.resources else None)

    def _construct(self, entry, path, cancel):
        # Optional model dependencies are imported only on a requested load.
        factory = self._factory
        if factory is None:
            try:
                from api.inference.model_adapter import build_runtime
            except ImportError as exc:
                raise RuntimeFailure('Install the optional Kadan inference requirements before loading a model.') from exc
            factory = build_runtime
        if self.resources is None:
            available = probe_memory()
            self.resources = ResourceManager(int(available.host_bytes * .8),
                {index: int(size * .8) for index, size in available.device_bytes.items()}, probe=probe_memory)
        gpu = os.environ.get('KADAN_GPU', '0')
        if not gpu.isdecimal():
            raise RuntimeFailure('KADAN_GPU must be one nonnegative GPU index; VRAM is not pooled.')
        if self._factory is None and int(gpu) not in self.resources.capacity.device_bytes:
            raise RuntimeFailure('The selected CUDA GPU is unavailable. Install a compatible PyTorch CUDA build on the GPU host.')
        return factory(entry, path, self.resources, device=f'cuda:{gpu}', cancel_event=cancel)

    def _release(self):
        if self._leased:
            from api.services.model_downloads import model_manager
            model_manager.release_runtime_model()
            self._leased = False

    async def _dispose(self):
        adapter = self.adapter
        if adapter is not None:
            await asyncio.to_thread(adapter.close)
            self.adapter = None
        self._release()

    async def load(self):
        from api.services.model_downloads import model_manager
        async with self._transition:
            if (self.state in ('loading', 'ready', 'unloading') or self._generation.locked()
                    or (self.task is not None and not self.task.done())):
                raise RuntimeFailure('Unload the current model before loading another.', 409)
            try:
                entry, path = model_manager.acquire_runtime_model()
            except ValueError as exc:
                raise RuntimeFailure(str(exc), 409) from exc
            self._leased = True
            self._cancel = threading.Event()
            self.model_id, self.error, self.state = entry.id, None, 'loading'
            self.task = asyncio.create_task(self._load(entry, path, self._cancel))
            return self.status()

    async def _load(self, entry, path, cancel):
        worker = asyncio.create_task(asyncio.to_thread(self._construct, entry, path, cancel))
        try:
            # A timeout requests cooperative cancellation; it never abandons a
            # loader thread that may still own CPU/GPU allocations.
            try:
                self.adapter = await asyncio.wait_for(asyncio.shield(worker), 1800)
            except asyncio.TimeoutError:
                cancel.set()
                self.adapter = await asyncio.shield(worker)
                raise RuntimeFailure('Model load exceeded 30 minutes and was cancelled.')
            if cancel.is_set():
                await self._dispose()
                return
            self.state = 'ready'
        except asyncio.CancelledError:
            cancel.set()
            with suppress(Exception):
                self.adapter = await asyncio.shield(worker)
            await self._dispose()
            raise
        except Exception as exc:
            self.error = str(exc)
            self.state = 'error'
            await self._dispose()

    async def unload(self):
        async with self._transition:
            self.state = 'unloading'
            self._cancel.set()
            if self.task is not None:
                with suppress(Exception):
                    await asyncio.shield(self.task)
                self.task = None
            if self._worker is not None:
                with suppress(Exception):
                    await asyncio.shield(self._worker)
            await self._dispose()
            self.model_id, self.error, self.state = None, None, 'unloaded'
            return self.status()

    async def close(self):
        await self.unload()

    async def complete(self, messages, model: str | None):
        if self.state != 'ready' or self.adapter is None:
            raise RuntimeFailure('No model is ready. Download, select and load one in Settings.')
        if model is not None and model != self.model_id:
            raise RuntimeFailure('Requested model is not the loaded model.', 409)
        if self._generation.locked():
            raise RuntimeFailure('A chat request is already running. Retry when it finishes.', 429)
        async with self._generation:
            adapter = self.adapter
            self._cancel = threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(adapter.generate,
                [{'role': message.role, 'text': message.text} for message in messages],
                max_new_tokens=256, cancel_event=self._cancel))
            self._worker = worker
            try:
                try:
                    text = await asyncio.wait_for(asyncio.shield(worker), 300)
                except asyncio.TimeoutError:
                    self._cancel.set()
                    with suppress(Exception):
                        await asyncio.shield(worker)
                    raise RuntimeFailure('Generation timed out; cooperative cleanup completed.', 504)
                if self.adapter is not adapter or self.state != 'ready':
                    raise RuntimeFailure('Model unloaded during generation.', 409)
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeFailure('Model produced no answer text.', 502)
                return text
            except asyncio.CancelledError:
                self._cancel.set()
                with suppress(Exception):
                    await asyncio.shield(worker)
                await self.unload()
                raise
            except Exception as exc:
                if self.state != 'unloading':
                    async with self._transition:
                        await self._dispose()
                        self.state, self.error = 'error', str(exc)
                if isinstance(exc, RuntimeFailure):
                    raise
                raise RuntimeFailure(f'Inference failed: {exc}', 502) from exc
            finally:
                if self._worker is worker:
                    self._worker = None


runtime_manager = RuntimeManager()
