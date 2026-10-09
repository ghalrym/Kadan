"""Real Redis FIFO and shared ledger; image/text arithmetic is synthetic."""
import asyncio
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from PIL import Image

from api.inference.image.two_rank_generation import TwoRankQwenImage
from api.inference.image.image_requests import ImageRequests
from api.inference.image.qwen_image_pipeline import GIB, REVISION
from api.inference.resources import ResourceManager, ResourceCancelled
from api.memory_manager import MemoryManager
from api.memory_manager.queue import InferenceQueue, Job
from api.services.image_jobs import ImageJobs
from api.services.runtime import RuntimeFailure


class Text:
    operations=('completion',)
    def __init__(self,resources,events):
        self.resources,self.events=resources,events;self.gpu=None
        self.started=asyncio.Event();self.release=asyncio.Event()
    def validate(self,payload,operation):return SimpleNamespace(**payload)
    async def offload_to_ram(self):
        if self.gpu:
            self.gpu.release();self.gpu=None;self.events.append('text:park')
    async def __call__(self,request,**kwargs):
        self.assert_parked()
        self.gpu=self.gpu or self.resources.reserve('text','llm',device_bytes={0:20*GIB},evict=None)
        with self.gpu.lease():
            self.events.append('text:'+request.label)
            if request.label=='A':self.started.set();await self.release.wait()
        return {'text':request.label}
    def assert_parked(self):
        assert not any(row['workload']=='image' and row['offload_on_handoff'] and row['device_bytes']
            for row in self.resources.snapshot()['reservations'].values())


class Ranks:
    def __init__(self,root,events):
        self.root,self.events=root,events;self.starts=0;self.started=threading.Event();self.release=threading.Event()
    def start(self,session,devices,deadline,cancel):self.starts+=1;self.devices=devices
    def exchange(self,command,deadline,cancel):
        op=command['operation']
        if op=='execute':
            assert not any(e=='text:overlap' for e in self.events)
            self.events.append('image:'+command['payload']['prompt']);self.started.set()
            while not self.release.wait(.01):
                if cancel.is_set():raise ResourceCancelled('image cancelled')
            Image.new('RGBA',(2048,2048),(10,20,30,255)).save(self.output())
        if op=='park':self.events.append('image:park')
        return [dict(command,rank=r,device=self.devices[r],status='ok',resident_bytes=0,request_seconds=.01) for r in (0,1)]
    def stop(self,deadline):self.events.append('image:reaped');return True
    def output(self):return self.root/'rank.png'


@unittest.skipUnless(os.getenv('KADAN_TEST_REDIS_URL'),'Dedicated Redis test URL required')
class DualQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.env=patch.dict(os.environ,{'KADAN_IMAGE_BACKEND':'dual','KADAN_IMAGE_DEVICES':'[0,1]'});self.env.start()
        self.events=[];self.resources=ResourceManager(300*GIB,{0:30*GIB,1:30*GIB})
        self.text=Text(self.resources,self.events);self.ranks=Ranks(self.root,self.events)
        model=self.root/'model';model.mkdir();(model/'weights.safetensors').write_bytes(b'fixture')
        downloads=SimpleNamespace(get_checkpoint=lambda _:(SimpleNamespace(revision=REVISION),model))
        runtime=SimpleNamespace(ensure_resources=lambda:self.resources)
        self.images=ImageJobs(self.root,downloads,runtime)
        self.images.generator=TwoRankQwenImage(model,self.resources,transport_factory=lambda *_:self.ranks)
        self.manager=object.__new__(MemoryManager)
        self.manager.features={'llm':self.text,'image':ImageRequests(self.images)}
        self.prefix='kadan:test:dual:'+uuid4().hex+':'
        self.queue=InferenceQueue(self.manager._execute,url=os.environ['KADAN_TEST_REDIS_URL'],prefix=self.prefix,lock_path=self.root/'queue.lock')
        self.manager.queue=self.queue;await self.queue.start()
    async def asyncTearDown(self):
        self.text.release.set();self.ranks.release.set();await self.queue.close()
        keys=[k async for k in self.queue.redis.scan_iter(match=self.prefix+'*')]
        if keys:await self.queue.redis.delete(*keys)
        await self.queue.redis.aclose();self.images.close();await self.text.offload_to_ram()
        self.env.stop();self.temp.cleanup()
    async def submit_text(self,label):return await self.queue.submit('llm','completion',{'label':label},'small')
    async def submit_image(self,prompt='B',**kwargs):return await self.queue.submit('image','generate',{'prompt':prompt,'count':1,**kwargs},'qwen-image-2.1')
    async def test_fifo_text_image_image_text_parks_both_and_reuses_ranks(self):
        a=await self.submit_text('A');await self.text.started.wait()
        b=await self.submit_image('B');c=await self.submit_image('C');d=await self.submit_text('D')
        self.text.release.set();self.assertTrue(await asyncio.to_thread(self.ranks.started.wait,3))
        self.assertEqual(self.events,['text:A','text:park','image:B'])
        self.ranks.release.set()
        for job in (a,b,c,d):await self.queue.wait(job)
        self.assertEqual(self.events,['text:A','text:park','image:B','image:C','image:park','text:D'])
        self.assertEqual(self.ranks.starts,1)
        self.assertEqual(self.images.generator.session.state,'parked')
        self.assertEqual(len(self.images.history()),2)
    async def test_queued_cancel_and_invalid_shape_never_park_text(self):
        a=await self.submit_text('A');await self.text.started.wait()
        b=await self.submit_image();await self.queue.cancel(b)
        invalid=await self.submit_image(aspect='16:9');c=await self.submit_text('C');self.text.release.set()
        await self.queue.wait(a)
        for job in (b,invalid):
            with self.assertRaises(RuntimeFailure):await self.queue.wait(job)
        await self.queue.wait(c)
        self.assertEqual(self.events,['text:A','text:C']);self.assertEqual(self.ranks.starts,0)
    async def test_active_cancel_reaps_before_next_text(self):
        self.text.release.set();b=await self.submit_image();self.assertTrue(await asyncio.to_thread(self.ranks.started.wait,3))
        c=await self.submit_text('C');await self.queue.cancel(b)
        with self.assertRaises(RuntimeFailure):await self.queue.wait(b)
        await self.queue.wait(c)
        self.assertLess(self.events.index('image:reaped'),self.events.index('text:C'))
        self.assertEqual(self.images.history(),[])


class DualPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_affinity_preserves_resident_text_before_fifo_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'weights.safetensors').write_bytes(b'fixture')
            resources = ResourceManager(100 * GIB, {0: 24 * GIB, 1: 24 * GIB})
            events = []
            text = Text(resources, events)
            await text(SimpleNamespace(label='resident'))
            downloads = SimpleNamespace(get_checkpoint=lambda _: (SimpleNamespace(revision=REVISION), root))
            images = ImageJobs(root / 'outputs', downloads,
                SimpleNamespace(ensure_resources=lambda: resources))
            manager = object.__new__(MemoryManager)
            manager.features = {'llm': text, 'image': ImageRequests(images)}
            manager.queue = SimpleNamespace(streams={})
            job = Job(id='a' * 32, feature='image', operation='generate', model='qwen-image-2.1',
                payload={'prompt': 'image', 'count': 1})
            resident = text.gpu
            before = resources.snapshot()
            unavailable = max(os.sched_getaffinity(0)) + 1
            try:
                with patch.dict('os.environ', {'KADAN_IMAGE_BACKEND': 'dual',
                        'KADAN_IMAGE_CPUS': f'[{unavailable}]'}, clear=True), patch(
                        'api.inference.image.rank_processes.ImageRankProcesses.start') as start:
                    with self.assertRaises(RuntimeFailure) as caught:
                        await manager._execute(job)
                self.assertEqual(caught.exception.status_code, 422)
                self.assertEqual(events, ['text:resident'])
                self.assertIs(text.gpu, resident)
                self.assertEqual(resources.snapshot(), before)
                self.assertIsNone(images.generator)
                start.assert_not_called()
            finally:
                images.close()
                await text.offload_to_ram()
