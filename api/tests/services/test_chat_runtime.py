import asyncio
import builtins
from pathlib import Path
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from api.inference.resources import MemoryCapacity, ResourceManager
from api.inference.llm.context import ContextLimitError, ContextMemoryError
from api.pydantic_models.chat import ChatMessage
from api.inference.errors import InferenceFailure
from api.services.chat_runtime import ChatRuntime
from api.tests.inference.llm.test_qwen_residency import fixture
from api.inference.llm.qwen_subprocess import HEADROOM_BYTES


class Adapter:
    def __init__(self):
        self.is_resident = True
        self.closed = False
        self.calls = []

    def configure_context(self, value):
        self.configured_context_limit = value

    def generate(self, messages, max_new_tokens, cancel_event):
        self.calls.append(messages)
        self.is_resident = True
        return 'Controlled adapter response'

    def close(self):
        self.closed = True
        self.is_resident = False


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = Adapter()
        self.factory = Mock(return_value=self.adapter)
        self.manager = ChatRuntime(self.factory, ResourceManager(1000, {0: 1000}))
        context_patch = patch('api.services.chat_runtime.read_context_settings', return_value=dict(
            configured_context_limit=None, effective_context_limit=131072, supported_context_limit=131072))
        context_patch.start()
        self.addCleanup(context_patch.stop)
        self.models = Mock()
        self.models.configured_context.return_value = None
        self.models.acquire_runtime_model.return_value = (SimpleNamespace(id='medium'), Path('/models/pinned'))
        patched = patch('api.services.chat_runtime.model_manager', self.models)
        patched.start()
        self.addCleanup(patched.stop)

    async def ready(self):
        await self.manager.load()
        await self.manager.task
        self.assertEqual(self.manager.state, 'ready')

    async def test_inference_backend_selects_exact_planner_before_device_admission(self):
        self.manager._factory = None
        for backend, module in [('native', 'qwen_subprocess.build_qwen_subprocess'),
                                ('native-resident', 'qwen_residency.build_resident_qwen')]:
            for gpu, device in (('auto', 'auto'), ('1', 'cuda:1')):
                with patch.dict('os.environ', {'KADAN_LLM_BACKEND': backend, 'KADAN_GPU': gpu}), \
                        patch('api.inference.llm.' + module, return_value=self.adapter) as factory, \
                        patch('api.services.chat_runtime.select_device', side_effect=AssertionError('Python placement called')):
                    await self.ready()
                    self.assertEqual(factory.call_args.kwargs['device'], device)
                    self.assertIs(factory.call_args.args[2], self.manager.resources)
                    self.assertIsNone(self.adapter.configured_context_limit)
                    await self.manager.unload()

    async def test_unknown_backend_fails_before_allocations(self):
        self.manager._factory = None
        with patch.dict('os.environ', {'KADAN_LLM_BACKEND': 'typo'}):
            await self.manager.load()
            await self.manager.task
        self.assertEqual(self.manager.state, 'error')
        self.assertIn('KADAN_LLM_BACKEND', self.manager.error)
        self.assertFalse(self.manager.resources.snapshot()['reservations'])

    async def test_failed_construction_cleanup_retains_selection_until_unload_retry(self):
        self.adapter.configure_context = Mock(side_effect=RuntimeError('configuration cleanup failed'))
        self.adapter.close = Mock(side_effect=[RuntimeError('close failed'), RuntimeError('close failed again'), None])
        await self.manager.load()
        with self.assertRaisesRegex(RuntimeError, 'close failed again'):
            await self.manager.task
        self.assertIs(self.manager.adapter, self.adapter)
        self.assertEqual(self.manager.state, 'error')
        self.models.release_runtime_model.assert_not_called()
        await self.manager.unload()
        self.assertEqual(self.adapter.close.call_count, 3)
        self.assertIsNone(self.manager.adapter)
        self.assertEqual(self.manager.state, 'unloaded')
        self.models.release_runtime_model.assert_called_once()

    async def test_context_errors_preserve_ready_model_and_selection(self):
        await self.ready()
        for error, status in ((ContextLimitError('too many tokens'), 422), (ContextMemoryError('does not fit'), 503)):
            self.adapter.generate = Mock(side_effect=error)
            with self.assertRaises(InferenceFailure) as caught:
                await self.manager.complete([], None)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(self.manager.state, 'ready')
            self.assertFalse(self.adapter.closed)
            self.models.release_runtime_model.assert_not_called()
        self.assertEqual(self.manager.status()['effective_context_limit'], 131072)
        await self.manager.close()

    async def test_direct_adapter_receives_owned_resources_and_selected_path(self):
        await self.ready()
        args, kwargs = self.factory.call_args
        self.assertEqual(args[0].id, 'medium')
        self.assertEqual(args[1], Path('/models/pinned'))
        self.assertIs(args[2], self.manager.resources)
        self.assertEqual(kwargs['device'], 'cuda:0')
        answer = await self.manager.complete([ChatMessage(role='user', text='Hello')], None)
        self.assertEqual(answer, 'Controlled adapter response')
        self.assertEqual(self.adapter.calls, [[{'role': 'user', 'text': 'Hello'}]])
        await self.manager.unload()
        self.assertTrue(self.adapter.closed)
        self.models.release_runtime_model.assert_called_once()

    async def test_unloaded_and_wrong_model_fail(self):
        with self.assertRaises(InferenceFailure):
            await self.manager.complete([], None)
        await self.ready()
        with self.assertRaises(InferenceFailure) as error:
            await self.manager.complete([], 'small')
        self.assertEqual(error.exception.status_code, 409)
        await self.manager.close()

    async def test_load_failure_never_reports_ready_and_releases_selection(self):
        self.factory.side_effect = ValueError('Unsupported checkpoint tensor layout')
        await self.manager.load()
        await self.manager.task
        self.assertEqual(self.manager.state, 'error')
        self.assertIn('Unsupported', self.manager.error)
        self.models.release_runtime_model.assert_called_once()

    async def test_load_cancel_waits_for_loader_before_closing(self):
        entered = threading.Event()
        def build(*args, cancel_event, **kwargs):
            entered.set()
            cancel_event.wait(2)
            return self.adapter
        self.factory.side_effect = build
        await self.manager.load()
        await asyncio.to_thread(entered.wait, 1)
        await self.manager.unload()
        self.assertTrue(self.adapter.closed)
        self.assertEqual(self.manager.state, 'unloaded')

    async def test_concurrent_generation_rejected(self):
        await self.ready()
        async with self.manager._generation:
            with self.assertRaises(InferenceFailure) as error:
                await self.manager.complete([], None)
        self.assertEqual(error.exception.status_code, 429)
        await self.manager.close()

    async def test_offloaded_model_restores_on_next_generation(self):
        await self.ready()
        self.adapter.is_resident = False
        self.assertEqual(self.manager.status()['state'], 'offloaded')
        await self.manager.complete([], None)
        self.assertEqual(self.manager.status()['state'], 'ready')
        await self.manager.close()

    async def test_offloaded_admission_failure_preserves_selection_for_retry(self):
        await self.ready()
        self.adapter.is_resident = False
        self.adapter.generate = Mock(side_effect=ContextMemoryError('temporary budget pressure'))
        with self.assertRaises(InferenceFailure) as caught:
            await self.manager.complete([], None)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.manager.status()['state'], 'offloaded')
        self.assertFalse(self.adapter.closed)
        self.models.release_runtime_model.assert_not_called()
        def restored(*args, **kwargs):
            self.adapter.is_resident = True
            return 'Retry succeeded'
        self.adapter.generate.side_effect = restored
        self.assertEqual(await self.manager.complete([], None), 'Retry succeeded')
        self.assertEqual(self.manager.status()['state'], 'ready')
        await self.manager.close()

    async def test_cancelled_generation_finishes_before_close(self):
        entered, finished = threading.Event(), threading.Event()
        def generate(*args, cancel_event, **kwargs):
            entered.set()
            cancel_event.wait(2)
            finished.set()
            return 'cancelled result must not escape'
        self.adapter.generate = generate
        self.adapter.close = Mock(side_effect=lambda: self.assertTrue(finished.is_set()))
        await self.ready()
        task = asyncio.create_task(self.manager.complete([], None))
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.adapter.close.assert_called_once()
        self.assertEqual(self.manager.state, 'unloaded')

    async def test_bad_response_is_error_and_actual_adapter_is_closed(self):
        self.adapter.generate = Mock(return_value='')
        await self.ready()
        with self.assertRaises(InferenceFailure):
            await self.manager.complete([], None)
        self.assertEqual(self.manager.state, 'error')
        self.assertTrue(self.adapter.closed)

    async def test_cleanup_failure_retains_adapter_for_retry(self):
        await self.ready()
        self.adapter.close = Mock(side_effect=[ValueError('cleanup failed'), None])
        with self.assertRaises(ValueError):
            await self.manager.unload()
        self.assertIs(self.manager.adapter, self.adapter)
        self.models.release_runtime_model.assert_not_called()
        await self.manager.unload()
        self.assertIsNone(self.manager.adapter)

    async def test_multiple_gpu_selection_rejected(self):
        with patch.dict('os.environ', {'KADAN_GPU': '0,1'}):
            await self.manager.load()
            await self.manager.task
        self.assertEqual(self.manager.state, 'error')
        self.factory.assert_not_called()

    async def test_inference_import_errors_identify_missing_modules_and_preserve_cause(self):
        original_import = builtins.__import__
        self.manager._factory = None
        failures = (
            (ModuleNotFoundError("No module named 'torch'", name='torch'),
             'Inference dependency is missing: torch.'),
            (ModuleNotFoundError("No module named 'safetensors'", name='safetensors'),
             'Inference dependency is missing: safetensors.'),
            (ImportError("cannot import name 'ChangedAPI'", name='transformers'),
             "Inference runtime import failed: cannot import name 'ChangedAPI'"),
            (ModuleNotFoundError('loader failed without a module name'),
             'Inference runtime import failed: loader failed without a module name'),
        )
        for failure, expected in failures:
            def controlled_import(name, *args, **kwargs):
                if name == 'api.inference.llm.qwen_subprocess':
                    raise failure
                return original_import(name, *args, **kwargs)
            with self.subTest(error=expected), patch('builtins.__import__', side_effect=controlled_import):
                with self.assertRaises(InferenceFailure) as caught:
                    self.manager._construct(SimpleNamespace(id='medium'), Path('/models/pinned'), threading.Event())
                self.assertEqual(str(caught.exception), expected)
                self.assertIs(caught.exception.__cause__, failure)
                await self.manager.load()
                await self.manager.task
            self.assertEqual(self.manager.state, 'error')
            self.assertEqual(self.manager.status()['error'], expected)
            self.assertIsNone(self.manager.adapter)
        self.assertEqual(self.models.release_runtime_model.call_count, len(failures))


class ResourceBudgetTests(unittest.TestCase):
    def budgets(self, override=None):
        environment = {} if override is None else {'KADAN_GPU_BUDGET_BYTES': override}
        manager = ChatRuntime()
        with patch.dict('os.environ', environment, clear=True), \
                patch('api.services.chat_runtime.probe_memory', return_value=MemoryCapacity(2000, {0: 1000, 1: 1500})):
            return manager.ensure_resources().capacity

    def test_default_budgets_unchanged(self):
        self.assertEqual(self.budgets(), MemoryCapacity(1600, {0: 800, 1: 1200}))

    def test_explicit_budget_changes_only_named_gpu(self):
        self.assertEqual(self.budgets('{"0":950}'), MemoryCapacity(1600, {0: 950, 1: 1200}))

    def test_invalid_or_overcommitted_budgets_fail_closed(self):
        for value in ('', '{}', '[]', '{"0":1001}', '{"2":1}', '{"0":true}',
                      '{"0":0}', '{"0":-1}', '{"0":1.5}', '{"00":1}', '{"-1":1}'):
            with self.subTest(value=value), self.assertRaises(InferenceFailure):
                self.budgets(value)


class ResidentRuntimeDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.resources=ResourceManager(2*1024**3,{0:HEADROOM_BYTES+4096})
        self.adapter=fixture(self.root,self.resources)
        await asyncio.to_thread(self.adapter.configure_context,512)
        self.manager=ChatRuntime(resources=self.resources,generation_timeout=.1)
        self.manager.adapter=self.adapter;self.manager.state='ready';self.manager.model_id='small'

    async def asyncTearDown(self):
        await self.manager.close()

    async def test_restore_gets_load_budget_in_addition_to_generation(self):
        await asyncio.to_thread(self.adapter.offload_to_ram)
        self.adapter.load_timeout=.8
        (self.root/'mode').write_text('slowstart')  # .5s, beyond .1s generation allowance.
        self.assertAlmostEqual(self.adapter.completion_timeout(.1),2.5)
        answer=await self.manager.complete([ChatMessage(role='user',text='Hi')],None)
        self.assertEqual(answer,'AB');self.assertEqual(self.manager.state,'ready')
        self.assertTrue(self.adapter.worker_alive)

    async def test_outer_timeout_cancels_and_reaps_before_releasing_budgets(self):
        self.adapter.load_timeout=.02
        self.adapter.step_timeout=2
        process=self.adapter.worker.process
        (self.root/'mode').write_text('hang')
        with self.assertRaises(InferenceFailure) as caught:
            await self.manager.complete([ChatMessage(role='user',text='Hi')],None)
        self.assertEqual(caught.exception.status_code,504)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.manager.adapter)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
        self.assertIsNone(self.manager._worker)

    async def test_restore_command_timeout_also_reconciles_runtime_ownership(self):
        await asyncio.to_thread(self.adapter.offload_to_ram)
        process=self.adapter.worker.process;self.adapter.load_timeout=.03
        (self.root/'mode').write_text('starthang')
        with self.assertRaises(InferenceFailure) as caught:
            await self.manager.complete([ChatMessage(role='user',text='Hi')],None)
        self.assertEqual(caught.exception.status_code,504)
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.manager.adapter)
        self.assertEqual(self.resources.snapshot()['reservations'],{})
