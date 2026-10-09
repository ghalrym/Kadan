from types import SimpleNamespace
import unittest

from PIL import Image
import torch

from api.inference.image.split_pipeline import SplitPipeline
from api.inference.resources import ResourceCancelled


class Scheduler:
    config = {'fixed':True}
    def __init__(self):self.steps=0
    @classmethod
    def from_config(cls,config):return cls()


class Adapter:
    def __init__(self,transformer,rank,control,job):
        self.transformer=transformer;self.job=job;self.prefix={};self.closed=False
    def install(self):self.transformer.adapter=self
    def gather(self,output):return output
    def close(self):self.prefix.clear();self.closed=True;self.transformer.adapter=None


class Pipeline:
    def __init__(self):
        self.scheduler=Scheduler();self.fail=False
        self.transformer=SimpleNamespace(forward=self.forward,adapter=None)
        self.image_processor=SimpleNamespace(postprocess=lambda image,**kwargs:image)
    def forward(self,**kwargs):
        adapter=self.transformer.adapter
        if not adapter.prefix:adapter.prefix['conditioning']=self.condition
        self.scheduler.steps+=1
        if self.fail and self.scheduler.steps==7:raise ResourceCancelled('synthetic cancel')
        return (SimpleNamespace(ndim=3,shape=(1,16384,64)),)
    def __call__(self,**kwargs):
        self.condition=(sum(kwargs['prompt'].encode())+kwargs['generator'].initial_seed())%256
        for i in range(40):self.transformer.forward(kv_cache_mode='extract' if i==0 else 'cached')
        assert self.scheduler.steps==40
        value=self.transformer.adapter.prefix['conditioning']
        self.image_processor.postprocess(torch.ones(1),output_type='pil')
        return SimpleNamespace(images=[Image.new('RGBA',(2048,2048),(value,0,0,255))])


class SplitRequestTests(unittest.TestCase):
    def test_a_b_a_matches_isolated_baselines_with_new_prefix_scheduler_generator(self):
        pipeline=Pipeline();adapters=[]
        def factory(*args):
            item=Adapter(*args);adapters.append(item);return item
        engine=SplitPipeline(pipeline,0,None,factory)
        originals=(pipeline.transformer.forward,pipeline.image_processor.postprocess)
        for index,(prompt,seed) in enumerate([('alpha',3),('beta',9),('alpha',3)]):
            actual,_=engine.generate(str(index)*32,prompt,seed)
            baseline,_=SplitPipeline(Pipeline(),0,None,Adapter).generate('f'*32,prompt,seed)
            self.assertEqual(actual.tobytes(),baseline.tobytes())
            self.assertEqual(originals,(pipeline.transformer.forward,pipeline.image_processor.postprocess))
        self.assertEqual(len({id(a) for a in adapters}),3)
        self.assertEqual([a.job for a in adapters],['0'*32,'1'*32,'2'*32])
        self.assertTrue(all(a.closed and not a.prefix for a in adapters))

    def test_cancel_discards_prefix_and_restores_hooks(self):
        pipeline=Pipeline();adapters=[]
        def factory(*args):
            item=Adapter(*args);adapters.append(item);return item
        engine=SplitPipeline(pipeline,1,None,factory)
        original=pipeline.transformer.forward;pipeline.fail=True
        with self.assertRaises(ResourceCancelled):engine.generate('a'*32,'alpha',7)
        self.assertEqual(pipeline.transformer.forward,original)
        self.assertTrue(adapters[0].closed);self.assertFalse(adapters[0].prefix)
        pipeline.fail=False
        actual,_=engine.generate('b'*32,'beta',5)
        baseline,_=SplitPipeline(Pipeline(),1,None,Adapter).generate('c'*32,'beta',5)
        self.assertEqual(actual.tobytes(),baseline.tobytes())
