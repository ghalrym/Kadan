"""Kadan-owned model lifecycle; no external inference server or engine process."""
import asyncio
from contextlib import suppress
import os
import json
import threading

from api.inference.resources import ResourceManager, probe_memory
from api.inference.context import ContextLimitError, ContextMemoryError, resolve_context


_UNSET = object()

def read_context_settings(path, configured):
    """Read local checkpoint configuration and return configured, supported and effective limits;
    file or validation errors propagate.
    """
    supported, effective = resolve_context(json.loads((path / 'config.json').read_text()), configured)
    return dict(configured_context_limit=configured, supported_context_limit=supported,
                effective_context_limit=effective)


async def finish_cleanup(task):
    """Keep ownership until cleanup ends, then propagate any caller cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


class RuntimeFailure(Exception):
    def __init__(self, detail: str, status_code: int = 503):
        """Attach an HTTP status to a caller-visible lifecycle or inference failure."""
        super().__init__(detail)
        self.status_code = status_code


class RuntimeManager:
    def __init__(self, factory=None, resources=None):
        """Initialize one process-local controller with optional test factory and shared resource
        manager; allocate no model at construction.
        """
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
        self.context_settings = dict(configured_context_limit=None, supported_context_limit=None, effective_context_limit=None)

    def status(self):
        """Return lifecycle, context and accounting metadata, reporting ready but evicted adapters
        as offloaded without reloading them.
        """
        state = self.state
        if state == 'ready' and self.adapter is not None and not self.adapter.is_resident:
            state = 'offloaded'
        return dict(state=state, model_id=self.model_id, error=self.error, max_output_tokens=256, **self.context_settings,
                    memory=self.resources.snapshot() if self.resources else None)

    def _construct(self, entry, path, cancel):
        # Native model dependencies are imported only on a requested load.
        """Build on a worker thread using the selected single GPU and shared budgets, then apply
        context settings; close the adapter if configuration fails.
        """
        factory = self._factory
        if factory is None:
            try:
                from api.inference.model_adapter import build_runtime
            except ImportError as exc:
                detail = (f'Inference dependency is missing: {exc.name}.'
                          if isinstance(exc, ModuleNotFoundError) and exc.name
                          else f'Inference runtime import failed: {exc}')
                raise RuntimeFailure(detail) from exc
            factory = build_runtime
        if self.resources is None:
            available = probe_memory()
            self.resources = ResourceManager(int(available.host_bytes * .8),
                {index: int(size * .8) for index, size in available.device_bytes.items()}, probe=probe_memory)
        gpu = os.environ.get('KADAN_GPU', '0')
        if not gpu.isdecimal():
            raise RuntimeFailure('KADAN_GPU must be one nonnegative GPU index; VRAM is not pooled.')
        if self._factory is None and int(gpu) not in self.resources.capacity.device_bytes:
            raise RuntimeFailure('The selected CUDA GPU is unavailable. Check GPU access and driver compatibility.')
        adapter = factory(entry, path, self.resources, device=f'cuda:{gpu}', cancel_event=cancel)
        try:
            adapter.configure_context(self.context_settings['configured_context_limit'])
        except BaseException:
            adapter.close()
            raise
        return adapter

    def _release(self):
        """Release the model-selection lease once, allowing selection changes after cleanup."""
        if self._leased:
            from api.services.model_downloads import model_manager
            model_manager.release_runtime_model()
            self._leased = False

    async def _dispose(self):
        """Close the owned adapter on a worker thread before releasing selection; cleanup errors
        preserve its handle.
        """
        await finish_cleanup(asyncio.create_task(self._finish_dispose()))

    async def _finish_dispose(self):
        """Release the handle and selection only after native close has succeeded."""
        adapter = self.adapter
        if adapter is not None:
            await asyncio.to_thread(adapter.close)
            self.adapter = None
        self._release()

    def _start_load(self, model_manager):
        """Acquire selection and schedule construction; caller owns the transition lock."""
        try:
            entry, path = model_manager.acquire_runtime_model()
        except ValueError as exc:
            raise RuntimeFailure(str(exc), 409) from exc
        self._leased = True
        try:
            self.context_settings = read_context_settings(path, model_manager.configured_context(entry.id))
        except (ValueError, OSError) as exc:
            self._release()
            raise RuntimeFailure(f'Cannot load context configuration: {exc}', 422) from exc
        self._cancel = threading.Event()
        self.model_id, self.error, self.state = entry.id, None, 'loading'
        self.task = asyncio.create_task(self._load(entry, path, self._cancel))
        return self.status()

    async def load(self, model_id=None, context_limit=_UNSET):
        """Validate, configure and start one load transaction, or load the saved selection.

        Explicit requests may switch an idle ready model. Identical loading/ready
        requests are idempotent; conflicting construction or generation returns 409.
        A 202/loading result accepts construction, whose failure remains visible in status.
        """
        from api.services.model_downloads import model_manager, BusyError
        async with self._transition:
            if model_id is None:
                if (self.state in ('loading', 'ready', 'unloading') or self._generation.locked()
                        or (self.task is not None and not self.task.done())):
                    raise RuntimeFailure('Unload the current model before loading another.', 409)
                return self._start_load(model_manager)

            def validate():
                entry = model_manager._catalog_entry(model_id)
                if not model_manager._checkpoint_complete(entry):
                    raise RuntimeFailure('Download this model completely before loading it.', 409)
                configured = model_manager.configured_context(model_id) if context_limit is _UNSET else context_limit
                if configured is not None and (type(configured) is not int or not 1 <= configured <= 2**31 - 1):
                    raise ValueError('Context limit must be a positive integer or null')
                return configured, read_context_settings(model_manager._checkpoint_directory(entry), configured)

            try:
                # Never unload a usable model for an invalid or incomplete target.
                with model_manager._lock:
                    configured, settings = validate()
                same = self.model_id == model_id and self.context_settings == settings
                if same and self.state in ('loading', 'ready'):
                    return self.status()
                if (self.state in ('loading', 'unloading') or self._generation.locked()
                        or (self.task is not None and not self.task.done())):
                    raise RuntimeFailure('Model lifecycle or generation is busy; retry when it finishes.', 409)
                if self.adapter is not None or self._leased:
                    await self._unload_locked()
                # Selection/context endpoints use this same store lock. No await
                # occurs between persistence and acquiring the runtime selection lease.
                with model_manager._lock:
                    configured, _ = validate()
                    files = [model_manager.root / name for name in ('context.json', 'selection.json')]
                    previous = {path: path.read_bytes() if path.exists() else None for path in files}
                    try:
                        model_manager.set_context(model_id, configured)
                        model_manager.select(model_id)
                        return self._start_load(model_manager)
                    except Exception as failure:
                        try:
                            for path, data in previous.items():
                                if data is None:
                                    path.unlink(missing_ok=True)
                                else:
                                    temporary = path.with_suffix('.rollback.tmp')
                                    temporary.write_bytes(data)
                                    temporary.replace(path)
                        except OSError as rollback:
                            raise RuntimeFailure('Load failed and saved settings could not be restored; refresh Settings before retrying.', 503) from rollback
                        raise failure
            except BusyError as exc:
                raise RuntimeFailure(str(exc), 409) from exc
            except ValueError as exc:
                raise RuntimeFailure(f'Cannot load context configuration: {exc}', 422) from exc
            except OSError as exc:
                raise RuntimeFailure('Model storage is unavailable; load was not started. Refresh Settings before retrying.', 503) from exc

    async def _load(self, entry, path, cancel):
        """Await a shielded construction worker. Timeout or cancellation requests cooperative
        cleanup and waits for ownership to return rather than abandoning allocations.
        """
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
        """Serialize cancellation and cleanup, waiting for loader/generation workers before
        releasing the adapter and selection; return unloaded status.
        """
        async with self._transition:
            return await self._unload_locked()

    async def _unload_locked(self):
        """Retain transition ownership through cleanup, even on repeated caller cancellation."""
        self.state = 'unloading'
        return await finish_cleanup(asyncio.create_task(self._finish_unload()))

    async def _finish_unload(self):
        """Complete the owned unload transaction before its caller may release the lock."""
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
        self.context_settings = dict(configured_context_limit=None, supported_context_limit=None, effective_context_limit=None)
        return self.status()

    async def close(self):
        """Run the same cooperative unload path during application shutdown."""
        await self.unload()

    async def complete(self, messages, model: str | None):
        """Return text from one bounded-output generation, rejecting concurrent calls.
        Context/admission errors preserve the model; cancellation waits for cleanup and unloads,
        while other inference failures dispose it.
        """
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
            except (ContextLimitError, ContextMemoryError) as exc:
                # Admission failures are request errors, not a damaged model.
                raise RuntimeFailure(str(exc), 422 if isinstance(exc, ContextLimitError) else 503) from exc
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
