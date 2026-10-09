"""Synthetic subprocess faults; no Torch, CUDA, checkpoint payload or service."""
from collections import UserDict
import json
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.inference.llm.context import ContextLimitError, ContextMemoryError
from api.inference.llm.qwen_subprocess import QwenSubprocessAdapter, QWEN_HOST_BYTES
from api.inference.line_protocol import LineProtocolError
from api.inference.resources import ResourceBusy, ResourceExhausted, ResourceManager

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


class QwenSubprocessAdapterTests(unittest.TestCase):
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
        self.adapter = QwenSubprocessAdapter(SimpleNamespace(id='small'), self.root, self.resources,
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

    def test_mapping_tokenizer_result_reaches_worker(self):
        self.loaded()
        with patch.object(self.tokenizer, 'apply_chat_template',
                          return_value=UserDict({'input_ids': [2, 7], 'attention_mask': [1, 1]})):
            self.assertEqual(self.generate(), 'AB')
        self.assertIn('step 7 0', (self.root/'commands').read_text())

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

    def test_evicted_reload_budget_pressure_is_retryable_after_cleanup(self):
        self.loaded();self.adapter._evict()
        busy = self.resources.reserve('other-workload', 'image', device_bytes={0: 1024**3})
        try:
            with self.assertRaises(ContextMemoryError):self.generate()
            self.assertFalse(self.adapter._closed)
            self.assertIsNone(self.adapter.worker)
            self.assertIsNone(self.adapter.host)
            self.assertIsNone(self.adapter.reservation)
        finally:busy.release()
        self.assertEqual(self.generate(), 'AB')

    def test_reload_does_not_translate_admission_error_with_uncertain_ownership(self):
        self.loaded();self.adapter._evict()
        retained = self.resources.reserve('uncertain-native-host', 'llm', host_bytes=1)
        self.adapter.host = retained
        with patch.object(self.adapter, '_configure_context_locked', side_effect=ResourceBusy('cleanup uncertain')):
            with self.assertRaises(ResourceBusy):self.generate()
        self.assertIs(self.adapter.host, retained)
        self.assertTrue(self.resources.snapshot()['reservations'])

    def test_eviction_before_generation_gate_reloads_instead_of_reporting_unloaded(self):
        self.loaded()
        entering, proceed = threading.Event(), threading.Event()
        original_lock = self.adapter._lock
        class Gate:
            def __enter__(self):
                entering.set()
                if not proceed.wait(2):raise TimeoutError('test gate timed out')
                original_lock.acquire()
            def __exit__(self, *args):original_lock.release()
            def acquire(self, **kwargs):return original_lock.acquire(**kwargs)
            def release(self):original_lock.release()
        self.adapter._lock = Gate()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.generate)
            try:
                self.assertTrue(entering.wait(2))
                self.adapter._evict()
                self.assertFalse(self.adapter.is_resident)
            finally:proceed.set()
            self.assertEqual(future.result(timeout=3), 'AB')
        self.assertTrue(self.adapter.is_resident)

    def test_eviction_cannot_interleave_between_reload_and_generation(self):
        self.loaded();self.adapter._evict()
        loaded, proceed = threading.Event(), threading.Event()
        configure = self.adapter._configure_context_locked
        def pause_after_reload(configured):
            configure(configured)
            loaded.set()
            if not proceed.wait(2):raise TimeoutError('test reload gate timed out')
        with patch.object(self.adapter, '_configure_context_locked', side_effect=pause_after_reload), \
                ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.generate)
            try:
                self.assertTrue(loaded.wait(2))
                with self.assertRaises(ResourceBusy):self.adapter._evict()
            finally:proceed.set()
            self.assertEqual(future.result(timeout=3), 'AB')
        self.assertTrue(self.adapter.is_resident)

    def test_bad_ready_and_load_timeouts_release_reservations(self):
        for mode in ('badready','planhang','readyhang'):
            self.mode(mode)
            with self.subTest(mode=mode),self.assertRaises((LineProtocolError,TimeoutError)):
                self.loaded()
            self.assertFalse(self.resources.snapshot()['reservations'])

    def test_malformed_steps_crash_and_timeout_poison_child(self):
        for mode in ('badprogress','badeos','crash','oversize','hang'):
            self.mode(mode);self.loaded()
            with self.subTest(mode=mode),self.assertRaises((LineProtocolError,TimeoutError)):
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

    def test_post_spawn_setup_failures_keep_child_and_budgets_until_confirmed_exit(self):
        self.mode('readyhang')
        original_selector = selectors.DefaultSelector
        original_blocking = os.set_blocking
        for fault in ('selector', 'blocking', 'register_first', 'register_second'):
            for cleanup in ('success', 'terminate', 'kill', 'wait'):
                with self.subTest(fault=fault, cleanup=cleanup):
                    children = []
                    with ExitStack() as patches:
                        def setup_selector():
                            worker = self.adapter.worker
                            self.assertIsNotNone(worker)  # Ownership precedes all setup.
                            if '--serve' not in worker.process.args:
                                return original_selector()
                            children.append(worker.process)
                            if cleanup == 'terminate':
                                patches.enter_context(patch('os.killpg', side_effect=PermissionError('terminate failed')))
                            elif cleanup in ('kill', 'wait'):
                                def kill(pid, sig):
                                    if cleanup == 'kill' and sig == signal.SIGKILL:
                                        raise PermissionError('kill failed')
                                    # Simulate signals that have not yet caused exit.
                                patches.enter_context(patch('os.killpg', side_effect=kill))
                                patches.enter_context(patch.object(worker.process, 'wait',
                                    side_effect=subprocess.TimeoutExpired('worker', 2)))
                            if fault == 'selector':
                                raise RuntimeError('selector creation failed')
                            selector = original_selector()
                            if fault.startswith('register'):
                                original_register = selector.register
                                calls = 0
                                def register(*args, **kwargs):
                                    nonlocal calls
                                    calls += 1
                                    if calls == (1 if fault == 'register_first' else 2):
                                        raise RuntimeError('registration failed')
                                    return original_register(*args, **kwargs)
                                patches.enter_context(patch.object(selector, 'register', side_effect=register))
                            return selector
                        def blocking(fd, value):
                            if fault == 'blocking' and '--serve' in self.adapter.worker.process.args:
                                raise RuntimeError('set_blocking failed')
                            return original_blocking(fd, value)
                        patches.enter_context(patch('selectors.DefaultSelector', side_effect=setup_selector))
                        patches.enter_context(patch('os.set_blocking', side_effect=blocking))
                        with self.assertRaises((RuntimeError, PermissionError, subprocess.TimeoutExpired)):
                            self.loaded()
                        self.assertEqual(len(children), 1)
                        self.assertFalse(self.adapter.is_resident)
                        if cleanup == 'success':
                            self.assertIsNotNone(children[0].poll())
                            self.assertIsNone(self.adapter.worker)
                            self.assertFalse(self.resources.snapshot()['reservations'])
                        else:
                            self.assertIs(self.adapter.worker.process, children[0])
                            self.assertIsNone(children[0].poll())
                            self.assertEqual(len(self.resources.snapshot()['reservations']), 2)
                    # With injected faults removed, retry must reap the same
                    # real synthetic child before returning either reservation.
                    self.adapter._close_locked()
                    self.assertIsNotNone(children[0].poll())
                    self.assertFalse(self.resources.snapshot()['reservations'])


if __name__=='__main__':unittest.main()
