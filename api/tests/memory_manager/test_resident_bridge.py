"""Real Redis + native IPC + image lifecycle; synthetic numerical leaves, no DB."""
import asyncio
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from PIL import Image

from api.inference.image.qwen_image_pipeline import QwenImagePipeline, REVISION, GIB
from api.inference.resources import MemoryCapacity, ResourceManager
from api.memory_manager import MemoryManager
from api.memory_manager.queue import InferenceQueue
from api.services.image_jobs import ImageJobs
from api.services.runtime import RuntimeManager, RuntimeFailure
from api.tests.inference.llm.test_qwen_residency import fixture


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'), 'Dedicated Redis URL required')
class ResidentBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.events=[];self.active=0;self.label=None
        self.started=threading.Event();self.release=threading.Event()
        self.image_started=threading.Event();self.image_release=threading.Event()
        self.fail_image=False
        self.resources=ResourceManager(24*GIB,{0:10*GIB})
        self.adapter=fixture(self.root,self.resources)
        generate=self.adapter.generate;begin=self.adapter._begin_request;end=self.adapter._end_request;park=self.adapter.offload_to_ram
        def generate_text(messages, **kwargs):
            self.label=messages[0]['text']
            try:
                return generate(messages, **kwargs)
            except BaseException:
                if self.active:
                    self.leave('text:'+self.label)
                raise
        def begin_text(cancel):
            begin(cancel);self.enter('text:'+self.label)
            if self.label=='A':
                self.started.set()
                if not self.release.wait(5):raise TimeoutError('test barrier')
        def end_text(cancel):
            end(cancel);self.leave('text:'+self.label)
        def park_text():
            occupied=self.adapter.reservation is not None
            park()
            if occupied:self.events.append('text:park')
        self.adapter.generate=generate_text;self.adapter._begin_request=begin_text
        self.adapter._end_request=end_text;self.adapter.offload_to_ram=park_text
        self.adapter.configure_context(512)
        self.runtime=RuntimeManager(resources=self.resources)
        self.runtime.adapter=self.adapter;self.runtime.model_id='small';self.runtime.state='ready'
        image_path=self.root/'image';image_path.mkdir();(image_path/'weights.safetensors').write_bytes(b'fixture')
        self.pipeline=Mock()
        def forward(**kwargs):
            self.enter('image:B');self.image_started.set()
            try:
                if not self.image_release.wait(5):raise TimeoutError('test image barrier')
                kwargs['callback_on_step_end'](self.pipeline,0,0,{})
                if self.fail_image:raise RuntimeError('synthetic image failure')
                return SimpleNamespace(images=[Image.new('RGB',(2,2))])
            finally:self.leave('image:B')
        self.pipeline.side_effect=forward
        def placement(device):
            if device=='cpu':self.events.append('image:park')
            return self.pipeline
        self.pipeline.to.side_effect=placement
        self.image_factory=Mock(return_value=self.pipeline)
        torch=SimpleNamespace(float32='fp32',bfloat16='bf16',Generator=Mock(),
            cuda=SimpleNamespace(device=lambda _:nullcontext(),empty_cache=Mock(),synchronize=Mock()))
        self.image_pipeline=QwenImagePipeline(image_path,self.resources,modules=lambda:(torch,
            SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(from_pretrained=self.image_factory))))
        image_generate=self.image_pipeline.generate
        def generate_image(prompt, aspect, seeds, cancel, image=None):
            self.image_cancel=cancel
            return image_generate(prompt, aspect, seeds, cancel, image=image)
        self.image_pipeline.generate=generate_image
        downloads=SimpleNamespace(get_checkpoint=lambda _:(SimpleNamespace(revision=REVISION),image_path))
        self.images=ImageJobs(self.root,downloads,self.runtime);self.images.generator=self.image_pipeline
        self.manager=MemoryManager(runtime=self.runtime,images=self.images)
        await self.manager.queue.redis.aclose()
        self.prefix='kadan:test:'+uuid4().hex+':'
        self.manager.queue=InferenceQueue(self.manager._execute,url=os.environ['KADAN_TEST_REDIS_URL'],
            prefix=self.prefix,lock_path=self.root/'consumer.lock')
        self.assertTrue(await self.manager.start())

    def enter(self, label):
        self.assertEqual(self.active,0,'model executions overlapped');self.active+=1
        rows=self.resources.snapshot()['reservations']
        if label.startswith('image'):
            self.assertTrue(self.adapter.parked or not self.adapter.worker_alive)
            self.assertNotIn(self.adapter.owner+':device',rows)
            if self.adapter.worker_alive:self.assertIn(self.adapter.owner+':context',rows)
        else:
            self.assertFalse(any(row['workload']=='image' and row['offload_on_handoff'] and row['device_bytes'] for row in rows.values()))
        self.events.append(label+':start')

    def leave(self, label):
        self.events.append(label+':end');self.active-=1

    async def text(self, label):
        return await self.manager.queue.submit('llm','completion',{'messages':[{'role':'user','text':label}]},'small')

    async def image(self):
        return await self.manager.queue.submit('image','generate',{'prompt':'tree','count':1},'qwen-image-2.1')

    async def wait_started(self, event):
        self.assertTrue(await asyncio.to_thread(event.wait,3))

    async def asyncTearDown(self):
        self.release.set();self.image_release.set()
        redis=self.manager.queue.redis
        await self.manager.queue.close()
        keys=[key async for key in redis.scan_iter(match=self.prefix+'*')]
        if keys:await redis.delete(*keys)
        await self.manager.close();self.temp.cleanup()

    async def test_fifo_handoff_and_arrivals_keep_backing_and_single_execution(self):
        worker=self.adapter.worker;host=self.adapter.host
        a=await self.text('A');await self.wait_started(self.started)
        b=await self.image();c=await self.text('C');self.release.set()
        await self.wait_started(self.image_started)
        d=await self.text('D')  # Cache hit arrives while older image owns execution.
        self.assertEqual(self.events,['text:A:start','text:A:end','text:park','image:B:start'])
        self.image_release.set()
        for job in (a,b,c,d):await self.manager.queue.wait(job)
        self.assertEqual(self.events,['text:A:start','text:A:end','text:park','image:B:start','image:B:end',
            'image:park','text:C:start','text:C:end','text:D:start','text:D:end'])
        self.assertIs(self.adapter.worker,worker);self.assertIs(self.adapter.host,host)
        self.image_factory.assert_called_once();self.assertEqual(len(self.images.history()),1)
        self.assertEqual(self.active,0)

    async def test_queued_image_cancel_does_not_park_or_bypass_text(self):
        a=await self.text('A');await self.wait_started(self.started)
        b=await self.image();c=await self.text('C');await self.manager.queue.cancel(b);self.release.set()
        await self.manager.queue.wait(a)
        with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(b)
        await self.manager.queue.wait(c)
        self.assertEqual(self.events,['text:A:start','text:A:end','text:C:start','text:C:end'])
        self.image_factory.assert_not_called()

    async def test_image_failure_cleans_before_next_text_and_publishes_nothing(self):
        self.release.set();self.image_release.set();self.fail_image=True
        a=await self.text('A');b=await self.image();c=await self.text('C')
        await self.manager.queue.wait(a)
        with self.assertRaisesRegex(RuntimeFailure,'synthetic image failure'):await self.manager.queue.wait(b)
        await self.manager.queue.wait(c)
        self.assertLess(self.events.index('image:B:end'),self.events.index('text:C:start'))
        self.assertIsNone(self.image_pipeline.gpu);self.assertIsNone(self.image_pipeline.host)
        self.assertEqual(self.images.history(),[]);self.assertEqual(self.active,0)

    async def test_uncertain_park_prevents_image_execution_and_text_reuse(self):
        a=await self.text('A');await self.wait_started(self.started)
        b=await self.image();c=await self.text('C')
        (self.root/'mode').write_text('park-failure');self.release.set()
        await self.manager.queue.wait(a)
        with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(b)
        with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(c)
        self.image_factory.assert_not_called()
        self.assertEqual(self.events,['text:A:start','text:A:end'])
        self.assertFalse(self.adapter.worker_alive)  # Runtime reconciles by reaping.

    async def test_active_image_cancel_waits_for_callback_cleanup_before_text(self):
        self.release.set()
        a=await self.text('A');b=await self.image();c=await self.text('C')
        await self.manager.queue.wait(a);await self.wait_started(self.image_started)
        await self.manager.queue.cancel(b)
        await self.wait_started(self.image_cancel)
        # The consumer signals cancellation while the synthetic pipeline is
        # blocked; the callback observes it only after releasing this barrier.
        self.image_release.set()
        with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(b)
        await self.manager.queue.wait(c)
        self.assertLess(self.events.index('image:B:end'),self.events.index('text:C:start'))
        self.assertEqual(self.images.history(),[]);self.assertEqual(self.active,0)

    async def test_active_native_cancel_reaps_before_image_execution(self):
        self.release.set();self.image_release.set()
        (self.root/'mode').write_text('hang')
        step_entered=threading.Event();step=self.adapter._step
        def blocked_step(*args):
            step_entered.set()
            return step(*args)
        self.adapter._step=blocked_step
        # Allow time for the queue cancellation poll rather than the IPC deadline.
        self.adapter.step_timeout=5
        a=await self.text('A');await self.wait_started(step_entered)
        child=self.adapter.worker.process
        b=await self.image();await self.manager.queue.cancel(a)
        with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(a)
        await self.manager.queue.wait(b)
        self.assertIsNotNone(child.poll())
        self.assertLess(self.events.index('text:A:end'),self.events.index('image:B:start'))
        self.assertNotIn(self.adapter.owner+':context',self.resources.snapshot()['reservations'])
        self.assertEqual(self.active,0)

    async def test_invalid_edit_never_parks_text_and_validation_scratch_is_released(self):
        self.release.set()
        a=await self.text('A');await self.manager.queue.wait(a)
        before=self.resources.snapshot()['reservations']
        b=await self.manager.queue.submit('image','edit',
            {'prompt':'edit','count':1,'image':'data:image/png;base64,aW52YWxpZA=='},'qwen-image-2.1')
        c=await self.text('C')
        with self.assertRaises(RuntimeFailure) as caught:await self.manager.queue.wait(b)
        self.assertEqual(caught.exception.status_code,422)
        await self.manager.queue.wait(c)
        self.assertEqual(self.events,['text:A:start','text:A:end','text:C:start','text:C:end'])
        self.assertEqual(self.resources.snapshot()['reservations'],before)
        self.image_factory.assert_not_called()

    async def test_invalid_device_and_impossible_gpu_budget_do_not_park_text(self):
        self.release.set()
        a=await self.text('A');await self.manager.queue.wait(a)
        for configured in ('cuda:nope','cuda:9'):
            with patch.dict(os.environ,{'KADAN_IMAGE_DEVICE':configured}):
                b=await self.image()
                with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(b)
        # No configuration can fit even the sequential workspace on this budget.
        capacity=self.resources.capacity
        self.resources.capacity=MemoryCapacity(capacity.host_bytes,{0:GIB})
        try:
            b=await self.image()
            with self.assertRaises(RuntimeFailure):await self.manager.queue.wait(b)
        finally:self.resources.capacity=capacity
        self.assertEqual(self.events,['text:A:start','text:A:end'])
        self.image_factory.assert_not_called()
