"""Kadan-owned model lifecycle, including an opt-in original native worker."""
import asyncio
from contextlib import suppress
import os
import json
import math
import threading

from api.inference.placement import select_device
from api.inference.resources import ResourceManager, ResourceExhausted, probe_memory
from api.inference.llm.context import ContextLimitError, ContextMemoryError, resolve_context
from api.services.model_downloads import BusyError, model_manager


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
    def __init__(self, factory=None, resources=None, *, generation_timeout=300):
        """Initialize one process-local controller with optional test factory and shared resource
        manager; allocate no model at construction.
        """
        if not isinstance(generation_timeout, (int, float)) or not math.isfinite(generation_timeout) or generation_timeout <= 0:
            raise ValueError('Generation timeout must be finite and positive')
        self.generation_timeout = generation_timeout
        self.state = 'unloaded'
        self.model_id = None
        self.error = None
        self.adapter = None
        self.resources = resources
        self._resources_lock = threading.Lock()
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

    def ensure_resources(self):
        """Initialize shared admission from currently free host and per-device memory.

        Host admission defaults to 80% of available RAM. Device admission leaves
        20% free, capped at 1 GiB, so larger cards do not lose a disproportionate
        part of their capacity. KADAN_GPU_HEADROOM_BYTES sets a fixed per-device
        reserve; KADAN_HOST_BUDGET_BYTES and KADAN_GPU_BUDGET_BYTES override total
        admission. Explicit budgets must fit current physical free memory and
        take precedence over default headroom. Every later reservation also
        checks physical availability; these estimates are not allocator caps.
        """
        with self._resources_lock:
            if self.resources is None:
                available = probe_memory()
                headroom = os.environ.get('KADAN_GPU_HEADROOM_BYTES')
                host = os.environ.get('KADAN_HOST_BUDGET_BYTES')
                for name, raw in [('KADAN_GPU_HEADROOM_BYTES', headroom), ('KADAN_HOST_BUDGET_BYTES', host)]:
                    if raw is not None and (not raw.isascii() or not raw.isdecimal()
                                            or (name == 'KADAN_HOST_BUDGET_BYTES' and int(raw) <= 0)):
                        raise RuntimeFailure(f'{name} must be an integer byte count')
                host_bytes = int(available.host_bytes * .8) if host is None else int(host)
                if host_bytes > available.host_bytes:
                    raise RuntimeFailure('KADAN_HOST_BUDGET_BYTES exceeds currently available host memory')
                budgets = {index: max(0, size - (min(size - int(size * .8), 1024**3)
                    if headroom is None else int(headroom))) for index, size in available.device_bytes.items()}
                override = os.environ.get('KADAN_GPU_BUDGET_BYTES')
                if override is not None:
                    try:
                        values = json.loads(override)
                        if not isinstance(values, dict) or not values:
                            raise ValueError()
                        for index, size in values.items():
                            if (not index.isascii() or not index.isdecimal() or str(int(index)) != index
                                    or type(size) is not int or size <= 0
                                    or size > available.device_bytes.get(int(index), 0)):
                                raise ValueError()
                            budgets[int(index)] = size
                    except (ValueError, TypeError) as exc:
                        raise RuntimeFailure('KADAN_GPU_BUDGET_BYTES must be a nonempty JSON object of GPU indices '
                                             'to positive byte budgets within currently free device memory.') from exc
                self.resources = ResourceManager(host_bytes,
                    budgets, probe=probe_memory)
            return self.resources

    async def start(self):
        """Restore the saved selection in the background, exposing failures through status."""
        selected = model_manager._read_selected_model_id()
        if selected is None:
            return
        try:
            await self.load()
        except (RuntimeFailure, OSError, ValueError) as exc:
            self.model_id, self.state, self.error = selected, 'error', str(exc)

    def _construct(self, entry, path, cancel):
        """Build on a worker thread using the selected single GPU and shared budgets, then apply
        context settings; close the adapter if configuration fails.
        """
        factory = self._factory
        native = False
        if factory is None:
            backend = os.environ.get('KADAN_LLM_BACKEND', 'python')
            if backend not in ('python', 'native', 'native-resident'):
                raise RuntimeFailure('KADAN_LLM_BACKEND must be python, native or native-resident.')
            if backend in ('native', 'native-resident'):
                # Lazy optional backend import preserves startup/default behavior.
                from api.inference.llm.native import build_native
                factory, native = build_native, True
                if backend == 'native-resident':
                    # Optional protocol-v2 bridge; no Torch model engine import.
                    from api.inference.llm.qwen_residency import build_resident_qwen
                    factory = build_resident_qwen
        if factory is None:
            # Keep the API available when native dependencies are broken so Settings
            # can report the import failure instead of preventing server startup.
            try:
                from api.inference.llm.model_adapter import build_runtime
            except ImportError as exc:
                detail = (f'Inference dependency is missing: {exc.name}.'
                          if isinstance(exc, ModuleNotFoundError) and exc.name
                          else f'Inference runtime import failed: {exc}')
                raise RuntimeFailure(detail) from exc
            factory = build_runtime
        self.ensure_resources()
        gpu = os.environ.get('KADAN_GPU', 'auto')
        if gpu != 'auto' and not gpu.isdecimal():
            raise RuntimeFailure('KADAN_GPU must be auto or one primary GPU index.')
        if native:
            # Native planning admits its exact arena in configure_context; the
            # Python adapter's 2 GiB placement floor is not a native budget.
            device = 'auto' if gpu == 'auto' else f'cuda:{gpu}'
        elif self._factory is not None:
            device = f'cuda:{gpu if gpu != "auto" else 0}'
        else:
            # Prefer whole-checkpoint residency when possible. Packed adapters
            # distribute expert/projection entries if no one card fits the bank.
            floor = 2 * 1024**3
            required = getattr(entry, 'estimated_bytes', 0) + floor
            try:
                device = select_device(self.resources, required, 'auto' if gpu == 'auto' else f'cuda:{gpu}')
            except ResourceExhausted:
                device = select_device(self.resources, floor, 'auto' if gpu == 'auto' else f'cuda:{gpu}')
        adapter = factory(entry, path, self.resources, device=device, cancel_event=cancel)
        try:
            adapter.configure_context(self.context_settings['configured_context_limit'])
        except BaseException:
            try:
                adapter.close()
            except BaseException:
                # The load task waits for this construction thread before
                # disposing/unloading. Transfer uncertain ownership so it keeps
                # the selection lease and can retry cleanup deterministically.
                self.adapter = adapter
                raise
            raise
        return adapter

    def _release(self):
        """Release the model-selection lease once, allowing selection changes after cleanup."""
        if self._leased:
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
        async with self._transition:
            if model_id is None:
                if (self.state in ('loading', 'ready', 'unloading') or self._generation.locked()
                        or (self.task is not None and not self.task.done())):
                    raise RuntimeFailure('Unload the current model before loading another.', 409)
                return self._start_load(model_manager)

            def validate():
                entry = model_manager._language_entry(model_id)
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

    async def complete(self, messages, model: str | None, on_event=None, conversation_id=None):
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
            budget = getattr(adapter, 'completion_timeout', None)
            timeout = self.generation_timeout if budget is None else budget(self.generation_timeout)
            if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < self.generation_timeout:
                raise RuntimeFailure('Adapter completion timeout is invalid.')
            self._cancel = threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(adapter.generate,
                [{'role': message.role, 'text': message.text} for message in messages],
                max_new_tokens=256, cancel_event=self._cancel,
                **({"on_event": on_event} if on_event else {}),
                **({"conversation_id": conversation_id} if conversation_id else {})))
            self._worker = worker
            try:
                try:
                    text = await asyncio.wait_for(asyncio.shield(worker), timeout)
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
