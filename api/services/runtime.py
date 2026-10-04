"""One owned FreeToken process; host-backed experts, bounded GPU/request usage."""
import asyncio
import os
from pathlib import Path
import signal
import socket
import time

import httpx


class RuntimeFailure(Exception):
    def __init__(self, detail: str, status_code: int = 503):
        super().__init__(detail)
        self.status_code = status_code


class RuntimeManager:
    def __init__(self):
        self.state = 'unloaded'
        self.model_id: str | None = None
        self.error: str | None = None
        self.process = None
        self.task = None
        self.port = None
        self._leased = False
        self._transition = asyncio.Lock()
        self._generation = asyncio.Lock()

    def status(self):
        return dict(state=self.state, model_id=self.model_id, error=self.error)

    def command(self, path: Path, model_id: str, port: int):
        command = [os.environ.get('KADAN_FT_EXECUTABLE', 'ft'), 'serve',
                   '--model', str(path), '--served-model-name', model_id,
                   '--host', '127.0.0.1', '--port', str(port),
                   '--gpu', os.environ.get('KADAN_FT_GPU', '0'),
                   '--moe-strategy', 'offload', '--moe-cache-auto',
                   '--text-model-only',
                   '--memory-ratio', '0.8', '--max-running-requests', '1',
                   '--max-output-tokens', '1024', '--max-seq-len-override', '4096',
                   '--kv-reserve-tokens', '4096']
        if model_id in ('small', 'large'):
            command.extend(['--quant-backend', 'moe.nvfp4=triton'])
        return command

    async def load(self):
        from api.services.model_downloads import model_manager
        async with self._transition:
            if (self.state in ('loading', 'ready') or self._generation.locked()
                    or (self.task is not None and not self.task.done())):
                raise RuntimeFailure('Unload the current model before loading another.', 409)
            try:
                entry, path = model_manager.acquire_runtime_model()
            except ValueError as exc:
                raise RuntimeFailure(str(exc), 409) from exc
            self._leased = True
            self.model_id, self.error, self.state = entry.id, None, 'loading'
            self.task = asyncio.create_task(self._run(path))
            return self.status()

    async def _run(self, path):
        try:
            # Reserve a currently free loopback port. Readiness checks verify the model
            # name and owned process liveness, not just whether anything answers here.
            for _ in range(32):
                with socket.socket() as sock, socket.socket() as adjacent:
                    sock.bind(('127.0.0.1', 0))
                    candidate = sock.getsockname()[1]
                    if candidate == 65535:
                        continue
                    try:
                        # FreeToken also binds its scheduler transport on port + 1.
                        adjacent.bind(('127.0.0.1', candidate + 1))
                    except OSError:
                        continue
                    self.port = candidate
                    break
            else:
                raise RuntimeFailure('No free loopback port pair found for FreeToken.')
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            spawn = asyncio.create_task(asyncio.create_subprocess_exec(
                *self.command(path, self.model_id, self.port), env=env,
                start_new_session=True,
                # Inherit service logs: no pipe deadlocks or unbounded log buffers.
            ))
            try:
                self.process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                self.process = await spawn
                raise
            deadline = time.monotonic() + 1800
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{self.port}',
                                         timeout=2, trust_env=False) as client:
                while self.process.returncode is None:
                    if self.state == 'loading' and time.monotonic() > deadline:
                        raise RuntimeFailure('FreeToken startup timed out after 30 minutes; inspect API service logs.')
                    try:
                        health = await client.get('/health')
                        doc = health.json()
                        if doc.get('status') == 'error':
                            raise RuntimeFailure('FreeToken reported an engine error; inspect API service logs.')
                        health.raise_for_status()
                        if doc.get('status') == 'ok':
                            models = await client.get('/v1/models')
                            models.raise_for_status()
                            if self.model_id not in [item.get('id') for item in models.json().get('data', [])]:
                                raise RuntimeFailure('FreeToken readiness returned an unexpected model.')
                            self.state = 'ready'
                        elif self.state == 'ready':
                            raise RuntimeFailure('FreeToken is no longer ready.')
                    except httpx.HTTPError:
                        if self.state == 'ready':
                            raise RuntimeFailure('FreeToken stopped responding; unload and retry.')
                    await asyncio.sleep(1)
                raise RuntimeFailure(f'FreeToken exited with code {self.process.returncode}; inspect API service logs.')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = str(exc)
            self.state = 'error'
        finally:
            await self._stop_process()
            self._release()

    def _release(self):
        if self._leased:
            from api.services.model_downloads import model_manager
            model_manager.release_runtime_model()
            self._leased = False

    async def _stop_process(self):
        process = self.process
        if process is not None:
            # Kill the session even when the launcher exited: workers may survive it.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            self.process = None

    async def unload(self):
        async with self._transition:
            self.state = 'unloaded'
            if self.task is not None:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
                self.task = None
            # A cancelled task may never have entered its try/finally.
            await self._stop_process()
            self._release()
            self.model_id, self.error, self.port = None, None, None
            return self.status()

    async def close(self):
        await self.unload()

    async def complete(self, messages, model: str | None):
        if self.state != 'ready' or self.process is None or self.process.returncode is not None:
            raise RuntimeFailure('No model is ready. Download, select and load one in Settings.')
        if model is not None and model != self.model_id:
            raise RuntimeFailure('Requested model is not the loaded model.', 409)
        if self._generation.locked():
            raise RuntimeFailure('A chat request is already running. Retry when it finishes.', 429)
        async with self._generation:
            active_process = self.process
            try:
                async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                    response = await client.post(f'http://127.0.0.1:{self.port}/v1/chat/completions',
                        json={'model': self.model_id, 'messages': [
                            {'role': message.role, 'content': message.text} for message in messages],
                            'stream': False, 'max_tokens': 1024})
                    response.raise_for_status()
                    if self.process is not active_process or self.state != 'ready':
                        raise RuntimeFailure('Model unloaded during generation.', 409)
                    text = response.json()['choices'][0]['message']['content']
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('No text returned')
                    return text
            except asyncio.CancelledError:
                # Closing HTTP alone doesn't guarantee GPU work cancellation.
                await self.unload()
                raise
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                await self.unload()
                self.state, self.error = 'error', 'Generation failed; runtime unloaded. Inspect API service logs and reload.'
                raise RuntimeFailure(self.error, 502) from exc


runtime_manager = RuntimeManager()
