"""Synthetic subprocess faults; no Torch, CUDA, checkpoint payload or service."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.inference.llm.context import ContextLimitError
from api.inference.llm.native import NativeAdapter, NativeProtocolError, NATIVE_HOST_BYTES
from api.inference.resources import ResourceExhausted, ResourceManager

WORKER = r'''
import os, sys, time
from pathlib import Path
root=Path(sys.argv[2]); mode=(root/'mode').read_text()
if sys.argv[1]=='--plan':
 if mode=='planhang':time.sleep(30)
 capacity=int(sys.argv[3]); print(f'plan 1 {258*1024**2} 1024 16 {capacity} 64',flush=True);sys.exit(0)
capacity=int(sys.argv[4])
if mode=='readyhang':time.sleep(30)
if mode=='badready':print('ready 1 17 512 1024 1',flush=True)
else:print(f'ready 1 16 {capacity} 1024 {258*1024**2}',flush=True)
progress=0
for line in sys.stdin:
 with (root/'commands').open('a') as log:log.write(line)
 args=line.split()
 if args==['close']:print('closed 0',flush=True);sys.exit(0)
 if args==['reset']:progress=0;print('ok reset',flush=True);continue
 if mode=='hang':time.sleep(30)
 if mode=='crash':sys.exit(3)
 if mode=='oversize':print('x'*5000,flush=True);continue
 if mode=='stderr':os.write(2,b'x'*100000)
 progress+=1; token=int(args[1]); selected={7:3,3:4,4:15}.get(token,1);eos=int(selected==15)
 if mode=='badprogress':progress+=1
 if mode=='badeos':eos=2
 print(f'token {selected} {eos} {progress}',flush=True)
'''


class Tokenizer:
    tokens = [2,7]
    def apply_chat_template(self, chat, **kwargs):
        return list(self.tokens)
    def decode(self, tokens, **kwargs):
        return ''.join({3:'A',4:'B'}.get(token,'?') for token in tokens)


class Stream:
    def __init__(self, tokenizer, callback):self.tokenizer, self.callback = tokenizer, callback
    def put(self, token):self.callback({'content': self.tokenizer.decode([token])})
    def end(self):pass


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root/'config.json').write_text(json.dumps({'model_type':'qwen3_5_moe','text_config':{'max_position_embeddings':512}}))
        self.mode('normal')
        self.binary = self.root/'worker'
        self.binary.write_text('#!' + sys.executable + '\n' + WORKER)
        self.binary.chmod(0o700)
        self.resources = ResourceManager(2*1024**3,{0:1024**3})
        self.tokenizer = Tokenizer()
        self.adapter = NativeAdapter(SimpleNamespace(id='small'), self.root, self.resources,
            tokenizer_factory=lambda root:self.tokenizer, streamer_factory=Stream,
            worker_path=str(self.binary), load_timeout=.5, step_timeout=.3)
        self.addCleanup(self.cleanup)

    def mode(self, name):(self.root/'mode').write_text(name)
    def cleanup(self):
        try:self.adapter.close()
        finally:self.assertFalse(self.resources.snapshot()['reservations'])
    def loaded(self):self.adapter.configure_context(512)
    def generate(self, **kwargs):return self.adapter.generate([{'role':'user','text':'Hello'}], **kwargs)

    def test_prefill_decode_eos_streaming_and_repeat_reset(self):
        self.loaded();events=[]
        self.assertEqual(self.generate(on_event=events.append, conversation_id='x'), 'AB')
        self.assertEqual([e['content'] for e in events if 'content' in e], ['A','B'])
        self.assertEqual(events[-1], {'finish_reason':'stop'})
        timing=next(e['timing'] for e in events if 'timing' in e)
        self.assertEqual((timing['output_tokens'], timing['prefill_tokens']), (2,2))
        cache=next(e['cache'] for e in events if 'cache' in e)
        self.assertFalse(cache['hit']);self.assertEqual(cache['stored_tokens'],0)
        self.assertEqual((self.root/'commands').read_text().splitlines(),
                         ['reset','step 2 0','step 7 0','step 3 1','step 4 1'])
        self.assertEqual(self.generate(), 'AB')
        self.assertTrue(self.adapter.is_resident)

    def test_output_limit_does_not_feed_unneeded_last_token(self):
        self.loaded();events=[]
        self.assertEqual(self.generate(max_new_tokens=1,on_event=events.append),'A')
        self.assertEqual(events[-1], {'finish_reason':'length'})
        self.assertNotIn('step 3 1', (self.root/'commands').read_text())

    def test_early_prompt_prediction_of_eos_does_not_end_prefill(self):
        self.loaded();self.tokenizer.tokens=[4,7]
        self.assertEqual(self.generate(), 'AB')
        self.assertIn('step 7 0', (self.root/'commands').read_text())

    def test_final_prefill_eos_emits_stop_without_decode(self):
        self.loaded();self.tokenizer.tokens=[2,4];events=[]
        self.assertEqual(self.generate(on_event=events.append), '')
        self.assertEqual(events[-1], {'finish_reason':'stop'})
        self.assertEqual((self.root/'commands').read_text().splitlines(), ['reset','step 2 0','step 4 0'])

    def test_context_error_does_not_mutate_or_truncate_sequence(self):
        self.loaded();self.tokenizer.tokens=[2]*300
        with self.assertRaises(ContextLimitError):self.generate()
        self.assertFalse((self.root/'commands').exists());self.assertTrue(self.adapter.is_resident)

    def test_eviction_restores_but_explicit_close_never_reloads(self):
        self.loaded();self.adapter._evict();self.assertFalse(self.adapter.is_resident)
        self.assertEqual(self.generate(), 'AB')
        self.adapter.close()
        with self.assertRaises(RuntimeError):self.generate()
        self.assertFalse(self.adapter.is_resident)

    def test_bad_ready_and_load_timeouts_release_reservations(self):
        for mode in ('badready','planhang','readyhang'):
            self.mode(mode)
            with self.subTest(mode=mode),self.assertRaises((NativeProtocolError,TimeoutError)):
                self.loaded()
            self.assertFalse(self.resources.snapshot()['reservations'])

    def test_malformed_steps_crash_and_timeout_poison_child(self):
        for mode in ('badprogress','badeos','crash','oversize','hang'):
            self.mode(mode);self.loaded()
            with self.subTest(mode=mode),self.assertRaises((NativeProtocolError,TimeoutError)):
                self.generate()
            self.assertFalse(self.adapter.is_resident)
            self.adapter._close_locked()  # Runtime disposes failed generations after leases exit.
            self.assertFalse(self.resources.snapshot()['reservations'])

    def test_stderr_backpressure_is_drained_and_bounded(self):
        self.mode('stderr');self.loaded()
        self.assertEqual(self.generate(),'AB')
        self.assertLessEqual(len(self.adapter.worker.diagnostics),8192)

    def test_cooperative_parent_cancellation_reaps_worker(self):
        self.mode('hang');self.loaded();event=threading.Event()
        timer=threading.Timer(.05,event.set);timer.start()
        try:
            with self.assertRaises(InterruptedError):self.generate(cancel_event=event)
        finally:timer.join()
        self.assertFalse(self.adapter.is_resident)

    def test_unconfirmed_exit_retains_accounting_until_retry(self):
        self.loaded()
        with patch.object(self.adapter.worker, 'stop', side_effect=subprocess.TimeoutExpired('worker',5)):
            with self.assertRaises(subprocess.TimeoutExpired):self.adapter._evict()
        self.assertTrue(self.resources.snapshot()['reservations'])
        self.adapter._evict();self.assertFalse(self.resources.snapshot()['reservations'])

    def test_exact_plan_budget_is_admitted_before_serve(self):
        self.resources = ResourceManager(2*1024**3,{0:1024})
        self.adapter.resources=self.resources
        with self.assertRaises(ResourceExhausted):self.loaded()
        self.assertFalse((self.root/'commands').exists())
        self.assertFalse(self.resources.snapshot()['reservations'])

    def test_invalid_tokenizer_ids_preserve_idle_worker(self):
        self.loaded()
        for tokens in ([],[-1],[16],[True],[[2,7]]):
            self.tokenizer.tokens=tokens
            with self.subTest(tokens=tokens),self.assertRaises(ContextLimitError):self.generate()
        self.assertFalse((self.root/'commands').exists())


if __name__=='__main__':unittest.main()
