"""Explicit real-leaf acceptance harness; run only in the isolated GPU1 container."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import diffusers
import torch

from api.inference.image.model import NativeImage, CONTEXT_BYTES
from api.inference.image.profiling import profile_pipeline
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.utils import compile_times
from api.inference.llm.native_resident import ResidentAdapter
from api.inference.resources import ResourceManager, probe_memory
from api.memory_manager import MemoryManager
from api.memory_manager.queue import InferenceQueue
from api.services.images import ImageManager
from api.services.model_downloads import model_manager
from api.services.runtime import RuntimeManager, RuntimeFailure

ROOT = Path('/evidence')
MANIFEST = json.loads(Path('/harness/manifest.json').read_text())
START = time.monotonic()
LOG_LOCK = threading.Lock()
PHASE = 'startup'
RESOURCES = None
RUNTIME = None
IMAGE = None
FIRST_PID = None
FIRST_LOAD_IO = None
PARK_IO = None
PARK_CACHE = None
IMAGE_START = None
STEP_LAST = None
COMPILE_EVENTS = []
INDUCTOR = lookup_backend("inductor")

def measured_inductor(graph, inputs, **kwargs):
    started = time.monotonic()
    try:
        return INDUCTOR(graph, inputs, **kwargs)
    finally:
        item = {"seconds": time.monotonic()-started, "nodes": len(list(graph.graph.nodes)), "phase": PHASE}
        COMPILE_EVENTS.append(item)
        emit("compile.backend", **item)


def proc(pid):
    result = {'pid': pid}
    for name in ('status', 'io'):
        try:
            lines = Path(f'/proc/{pid}/{name}').read_text().splitlines()
            for line in lines:
                key, _, value = line.partition(':')
                if name == 'io' or key in ('VmRSS', 'RssAnon', 'VmHWM'):
                    result[key] = value.strip()
        except FileNotFoundError:
            result['exited'] = True
    return result


def snapshot():
    result = {'python': proc(os.getpid())}
    if RESOURCES:
        result['admission'] = RESOURCES.snapshot()
    if RUNTIME and RUNTIME.adapter:
        result['cache'] = RUNTIME.adapter.cache_statistics
    if RUNTIME and RUNTIME.adapter and RUNTIME.adapter.worker:
        child = RUNTIME.adapter.worker.process
        if child:
            result['native'] = proc(child.pid)
    for name in ('memory.current', 'memory.peak', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result[name] = path.read_text().strip()
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info(0)
        result['cuda'] = {'free': free, 'total': total,
            'python_allocated': torch.cuda.memory_allocated(0),
            'python_reserved': torch.cuda.memory_reserved(0)}
    return result


def emit(event, **fields):
    row = {'event': event, 'phase': PHASE, 'elapsed_s': round(time.monotonic()-START, 3),
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}
    with LOG_LOCK:
        with (ROOT/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, default=str)+'\n')
        print(json.dumps(row, default=str), flush=True)


class ObservedResident(ResidentAdapter):
    def configure_context(self, configured):
        global FIRST_PID, FIRST_LOAD_IO
        emit('text.load.begin', capacity=configured, snapshot=snapshot())
        super().configure_context(configured)
        FIRST_PID = self.worker.process.pid
        FIRST_LOAD_IO = proc(FIRST_PID)
        emit('text.load.end', native=FIRST_LOAD_IO, session=self.session, snapshot=snapshot())

    def generate(self, messages, **kwargs):
        kwargs['max_new_tokens'] = MANIFEST['text_output_tokens']
        return super().generate(messages, **kwargs)

    def _begin_request(self, cancel):
        emit('text.start.begin', snapshot=snapshot())
        super()._begin_request(cancel)
        assert self.worker.process.pid == FIRST_PID, 'Native child was reconstructed'
        emit('text.start.ack', request_id=self.request_id, snapshot=snapshot())

    def _end_request(self, cancel):
        ident = self.request_id
        super()._end_request(cancel)
        emit('text.end.ack', request_id=ident, snapshot=snapshot())

    def offload_to_ram(self):
        global PARK_IO, PARK_CACHE
        occupied = self.reservation is not None
        if occupied:
            emit('text.park.begin', snapshot=snapshot())
        super().offload_to_ram()
        if occupied:
            PARK_IO = proc(self.worker.process.pid)
            PARK_CACHE = dict(self.cache_statistics)
            emit('text.park.ack', native=PARK_IO, snapshot=snapshot())


class PipelineProxy:
    def __init__(self, actual):
        self.actual = actual
        self.compiled = False

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def to(self, device):
        emit('image.transfer.begin', device=str(device), snapshot=snapshot())
        result = self.actual.to(device)
        emit('image.transfer.end', device=str(device), snapshot=snapshot())
        return result

    def remove_all_hooks(self):
        emit('image.hooks.remove.begin')
        result = self.actual.remove_all_hooks()
        emit('image.hooks.remove.end', snapshot=snapshot())
        return result

    def __call__(self, **kwargs):
        global IMAGE_START, STEP_LAST
        IMAGE_START = STEP_LAST = time.monotonic()
        compile_before = len(COMPILE_EVENTS)
        if PHASE in ('D', 'F') and not self.compiled:
            assert self.actual.transformer._repeated_blocks == ['QwenImage21TransformerBlock']
            emit('compile.install.begin', mode='default', fullgraph=True, snapshot=snapshot())
            self.actual.transformer.compile_repeated_blocks(backend=measured_inductor, mode='default', fullgraph=True)
            self.compiled = True
            emit('compile.install.end', snapshot=snapshot())
        original = kwargs['callback_on_step_end']
        def step(pipeline, index, timestep, values):
            global STEP_LAST
            now = time.monotonic()
            emit('image.step', index=index+1, step_s=round(now-STEP_LAST,3),
                 image_s=round(now-IMAGE_START,3), snapshot=snapshot())
            STEP_LAST = now
            return original(pipeline, index, timestep, values)
        kwargs['callback_on_step_end'] = step
        emit('image.forward.begin', width=kwargs['width'], height=kwargs['height'],
             steps=kwargs['num_inference_steps'], snapshot=snapshot())
        with profile_pipeline(self.actual, torch, IMAGE.native.device, emit,
                ROOT/f'transformer-{PHASE}-step10.json', trace_transformer_index=10, record_shapes=True):
            result = self.actual(**kwargs)
        emit('image.forward.end', image_s=round(time.monotonic()-IMAGE_START,3), compiled=self.compiled, new_compilations=len(COMPILE_EVENTS)-compile_before, compile_metrics=compile_times(), snapshot=snapshot())
        if PHASE == 'F':
            assert len(COMPILE_EVENTS) == compile_before, 'Warm request recompiled after park/unpark'
        return result


class PipelineFactory:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        emit('image.load.begin', snapshot=snapshot())
        actual = diffusers.QwenImage21Pipeline.from_pretrained(*args, **kwargs)
        components = {}
        for name in ('text_encoder', 'transformer', 'vae'):
            counts = {}
            for parameter in getattr(actual, name).parameters():
                key = str(parameter.device) + '/' + str(parameter.dtype)
                counts[key] = counts.get(key, 0) + parameter.numel() * parameter.element_size()
            components[name] = counts
        emit('image.load.end', components=components, snapshot=snapshot())
        return PipelineProxy(actual)


def native_factory(entry, path, resources, device, cancel_event):
    return ObservedResident(entry, path, resources, device=device, cancel_event=cancel_event,
        worker_path='/native/bin/run-model-worker', load_timeout=MANIFEST['native_load_deadline_s'],
        step_timeout=MANIFEST['native_step_deadline_s'])


async def run():
    global RESOURCES, RUNTIME, IMAGE, PHASE
    assert MANIFEST['source_sha'] == Path('/harness/source-revision.txt').read_text().strip()
    for relative, digest in MANIFEST['native_sha256'].items():
        assert hashlib.sha256((Path('/native')/relative).read_bytes()).hexdigest() == digest
    assert torch.cuda.device_count() == 1, 'Only GPU1 may be exposed'
    assert torch.cuda.get_device_capability(0) == (8, 6)
    available = probe_memory()
    assert available.device_bytes[0] >= MANIFEST['gpu_budget_bytes'], available
    assert available.host_bytes >= MANIFEST['startup_host_min_bytes'], available
    RESOURCES = ResourceManager(MANIFEST['host_budget_bytes'], {0: MANIFEST['gpu_budget_bytes']}, probe=probe_memory)
    RESOURCES.reserve("benchmark:compiler", "image", host_bytes=MANIFEST["compiler_host_reservation_bytes"], offload_on_handoff=False, allow_eviction=False)
    RESOURCES.framework_context(0, CONTEXT_BYTES)
    RUNTIME = RuntimeManager(factory=native_factory, resources=RESOURCES)
    assert model_manager._read_selected_model_id() == 'small'
    assert model_manager.configured_context('small') == MANIFEST['text_context']
    image_entry, image_path = model_manager.get_checkpoint('qwen-image-2.1')
    assert image_entry.revision == MANIFEST['image_revision']
    IMAGE = ImageManager(ROOT/'media', downloads=model_manager, runtime=RUNTIME)
    IMAGE.native = NativeImage(image_path, RESOURCES, device='cuda:0',
        modules=lambda: (torch, SimpleNamespace(QwenImage21Pipeline=PipelineFactory)),
        offload_mode=MANIFEST['image_offload_mode'])
    manager = MemoryManager(runtime=RUNTIME, images=IMAGE)
    await manager.queue.redis.aclose()
    queue = InferenceQueue(manager._execute, url=os.environ['KADAN_REDIS_URL'],
        prefix=MANIFEST['redis_prefix'], lock_path=ROOT/'inference.lock')
    manager.queue = queue
    abort = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, abort.set)
    loop.add_signal_handler(signal.SIGINT, abort.set)
    result = {'manifest': MANIFEST, 'status': 'running'}
    ids = []
    async def execute(job):
        global PHASE
        PHASE = job.payload.get('prompt', 'text') if job.feature == 'image' else job.payload['messages'][0]['text']
        label = dict(zip(ids, 'ABCDEFG')).get(job.id, '?')
        PHASE = label
        emit('fifo.begin', label=label, job_id=job.id, feature=job.feature, snapshot=snapshot())
        try:
            value = await manager._execute(job)
            emit('fifo.end', label=label, job_id=job.id, snapshot=snapshot())
            if job.feature == 'image':
                assert IMAGE.native.gpu is None
                assert torch.cuda.memory_allocated(0) <= CONTEXT_BYTES, 'Compiled state exceeded retained framework context allowance'
            return value
        except BaseException as exc:
            emit('fifo.failure', label=label, error=repr(exc), snapshot=snapshot())
            raise
    queue.execute = execute
    async def watch():
        while not abort.is_set():
            if (ROOT/'ABORT').exists():
                emit('abort.request', reason=(ROOT/'ABORT').read_text())
                abort.set()
                break
            if time.monotonic()-START > MANIFEST['run_deadline_s']:
                emit('abort.request', reason='overall deadline')
                abort.set()
                break
            if IMAGE_START and PHASE in ('B','D','F') and time.monotonic()-IMAGE_START > MANIFEST['image_deadline_s']:
                emit('abort.request', reason='image execution deadline')
                abort.set()
                break
            emit('heartbeat', snapshot=snapshot())
            await asyncio.sleep(15)
        for ident in ids:
            await queue.cancel(ident)
    watcher = None
    try:
        PHASE = 'load-text'
        emit('preflight.accepted', available=available.__dict__, snapshot=snapshot())
        await RUNTIME.load()  # Read saved selection/context; never persist changes.
        await asyncio.wait_for(asyncio.shield(RUNTIME.task), MANIFEST['native_load_deadline_s']+10)
        assert RUNTIME.state == 'ready', RUNTIME.error
        emit('loaded', snapshot=snapshot())
        await queue.start()
        # Pause consumer dispatch until all three accepted IDs have been recorded.
        gate = asyncio.Event()
        real_execute = queue.execute
        async def gated(job):
            await gate.wait()
            return await real_execute(job)
        queue.execute = gated
        ids.append(await queue.submit('llm','completion',{'messages':[{'role':'user','text':MANIFEST['text_a']}]},'small'))
        ids.append(await queue.submit('image','generate',{'prompt':MANIFEST['image_prompt'],'aspect':'1:1','count':1,'seed':42},'qwen-image-2.1'))
        ids.append(await queue.submit('llm','completion',{'messages':[{'role':'user','text':MANIFEST['text_c']}]},'small'))
        for _ in range(2):
            ids.append(await queue.submit('image','generate',{'prompt':MANIFEST['image_prompt'],'aspect':'1:1','count':1,'seed':42},'qwen-image-2.1'))
            ids.append(await queue.submit('llm','completion',{'messages':[{'role':'user','text':MANIFEST['text_c']}]},'small'))
        emit('fifo.accepted', ids=ids, snapshot=snapshot())
        gate.set()
        watcher = asyncio.create_task(watch())
        answers = []
        result['answers'] = answers
        for index, ident in enumerate(ids):
            answer = await queue.wait(ident)
            answers.append(answer)
            (ROOT/f'answer-{index}.json').write_text(json.dumps(answer,indent=2,default=str))
        assert not abort.is_set()
        assert RUNTIME.adapter.worker.process.pid == FIRST_PID
        assert IMAGE.native.pipeline is not None and IMAGE.native.gpu is None
        after = proc(FIRST_PID)
        cache = RUNTIME.adapter.cache_stats()
        assert PARK_CACHE['retained_bytes'] == RUNTIME.adapter.packed_weight_bytes
        assert cache['retained_bytes'] == RUNTIME.adapter.packed_weight_bytes
        assert cache['source_bytes'] == PARK_CACHE['source_bytes'] == RUNTIME.adapter.packed_weight_bytes
        assert cache['hits'] > PARK_CACHE['hits'] and cache['hit_bytes'] > PARK_CACHE['hit_bytes']
        assert cache['evictions'] == 0
        result['cache_park'] = PARK_CACHE
        result['cache_final'] = cache
        result['compile_events'] = COMPILE_EVENTS
        result.update(status='succeeded', answers=answers, native_initial_io=FIRST_LOAD_IO,
            native_park_io=PARK_IO, native_final_io=after, snapshot=snapshot())
        if FIRST_LOAD_IO and PARK_IO:
            result['native_initial_rchar'] = int(FIRST_LOAD_IO['rchar'])
            result['native_reload_rchar'] = int(after['rchar'])-int(PARK_IO['rchar'])
        emit('acceptance.complete', result=result)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc), snapshot=snapshot())
        emit('acceptance.failed', error=repr(exc), snapshot=snapshot())
    finally:
        PHASE = 'cleanup'
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        emit('cleanup.begin', snapshot=snapshot())
        try:
            await asyncio.wait_for(manager.close(), MANIFEST['cleanup_deadline_s'])
            rows = RESOURCES.snapshot()['reservations']
            assert set(rows) == {'framework-context:0', 'benchmark:compiler'}, rows
            result['cleanup'] = {'status':'model_allocations_released_context_and_compiler_reservation_until_exit','snapshot':snapshot()}
            emit('cleanup.confirmed', snapshot=snapshot())
        except BaseException as exc:
            result['cleanup'] = {'status':'failed','error':repr(exc),'snapshot':snapshot()}
            result['status'] = 'failed'
            emit('cleanup.failed', error=repr(exc), snapshot=snapshot())
        result['elapsed_s'] = time.monotonic()-START
        (ROOT/'result.json').write_text(json.dumps(result,indent=2,default=str))
    return result['status']=='succeeded'


if __name__ == '__main__':
    raise SystemExit(0 if asyncio.run(run()) else 1)
