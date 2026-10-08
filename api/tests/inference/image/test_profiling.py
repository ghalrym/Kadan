from types import SimpleNamespace
import unittest

from api.inference.image.profiling import profile_pipeline


class Operation:
    def forward(self, value):
        return value + 1

    def decode(self, value):
        return self.decoder.forward(value)

    def encode_prompt(self, value):
        return value

    def postprocess(self, value):
        return value


class ProfilingTests(unittest.TestCase):
    def test_stage_counts_results_and_methods_restored_on_failure(self):
        pipeline=Operation()
        pipeline.transformer=Operation();pipeline.vae=Operation()
        pipeline.vae.decoder=Operation();pipeline.image_processor=Operation()
        rows=[]
        def emit(event, **values):rows.append((event,values))
        with self.assertRaisesRegex(ValueError,'intentional'):
            with profile_pipeline(pipeline,SimpleNamespace(),'cpu',emit):
                self.assertEqual(pipeline.encode_prompt(2),2)
                self.assertEqual(pipeline.transformer.forward(2),3)
                self.assertEqual(pipeline.vae.decode(2),3)
                self.assertEqual(pipeline.image_processor.postprocess(2),2)
                raise ValueError('intentional')
        self.assertNotIn('encode_prompt',vars(pipeline))
        self.assertNotIn('forward',vars(pipeline.transformer))
        self.assertEqual(rows[-1][1]['counts'],dict(prompt_encoding=1,transformer=1,vae_decode=1,vae_tile=1,postprocess=1))
        ends=[v for event,v in rows if event=='image.phase.end']
        self.assertEqual(len(ends),5)
        self.assertTrue(all(v['seconds']>=0 for v in ends))

    def test_throwing_stage_is_labeled_failed_and_original_hook_is_restored(self):
        pipeline=Operation();pipeline.transformer=Operation();pipeline.vae=Operation()
        pipeline.vae.decoder=Operation();pipeline.image_processor=Operation()
        def installed_hook(value):raise ValueError('stage failed')
        pipeline.transformer.forward=installed_hook
        rows=[]
        with self.assertRaisesRegex(ValueError,'stage failed'):
            with profile_pipeline(pipeline,SimpleNamespace(),'cpu',lambda event,**values:rows.append((event,values))):
                pipeline.transformer.forward(1)
        self.assertIs(pipeline.transformer.forward,installed_hook)
        ends=[v for event,v in rows if event=='image.phase.end']
        self.assertEqual(ends[0]['status'],'failed')
        # A subsequent hook installation also survives the next profile cycle.
        pipeline.transformer.forward=lambda value:value+2
        replacement=pipeline.transformer.forward
        with profile_pipeline(pipeline,SimpleNamespace(),'cpu',lambda *a,**k:None):
            self.assertEqual(pipeline.transformer.forward(1),3)
        self.assertIs(pipeline.transformer.forward,replacement)
