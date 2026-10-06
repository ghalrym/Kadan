import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from api.inference.resources import ResourceManager, ResourceExhausted
from api.routes.v1.decisions import DecisionRequest
from api.services.decisions import DecisionManager, parse_answer, translate_questions
from api.services.runtime import RuntimeFailure, RuntimeManager


def questions():
    return DecisionRequest(state='Service failed', questions=[
        dict(key='cause', type='Choice', instructions='Cause?', options=[dict(key='load', description='Capacity')]),
        dict(key='severity', type='Score', instructions='Severity?', levels=['Low', 'High']),
        dict(key='urgent', type='Noul', instructions='Urgent?', trueWhen='Immediate action', falseWhen='Can wait'),
    ]).questions


class ContractTests(unittest.TestCase):
    def test_translation_and_native_fractional_outputs(self):
        qs = questions()
        definitions = translate_questions(qs)
        self.assertEqual(definitions['cause']['criteria'], {'load': 'Capacity'})
        self.assertEqual(definitions['severity']['criteria'], ['Low', 'High'])
        self.assertEqual(definitions['urgent']['criteria'], {'true': 'Immediate action', 'false': 'Can wait'})
        answer = parse_answer(qs[1], dict(type='score', score=.75, confidence=.1,
                                         answer_confidence=.8, probabilities={'0': .25, '1': .75}))
        self.assertEqual(answer.value, .75)
        self.assertEqual(answer.confidence, .8)
        self.assertEqual(parse_answer(qs[2], dict(type='noul', noul=.85)).value, .85)
        self.assertIsNone(parse_answer(qs[0], dict(type='choice', choice='load', confidence=.9)).confidence)

    def test_invalid_values_and_probabilities_fail_closed(self):
        for question, answer in [
            (questions()[0], dict(type='choice', choice='unknown')),
            (questions()[1], dict(type='score', score=True)),
            (questions()[1], dict(type='score', score=1.1)),
            (questions()[2], dict(type='noul', noul=True)),
            (questions()[2], dict(type='noul', noul=float('nan'))),
            (questions()[2], dict(type='noul', noul=.5, answer_confidence=float('inf'))),
            (questions()[1], dict(type='score', score=.5, probabilities={'0': .2, '1': .2})),
        ]:
            with self.subTest(answer=answer), self.assertRaises(RuntimeFailure) as caught:
                parse_answer(question, answer)
            self.assertEqual(caught.exception.status_code, 502)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.resources = ResourceManager(100, {})
        self.agent = Mock()
        self.agent.predict.return_value = dict(answers={'urgent': dict(type='noul', noul=.7)}, usage={})
        self.loader = Mock(return_value=self.agent)
        self.manager = DecisionManager(self.loader, self.resources, check=Mock(), ram_bytes=60)
        self.qs = [questions()[2]]

    async def asyncTearDown(self):
        await self.manager.close()

    async def test_lazy_resident_reuse_pressure_eviction_and_reload(self):
        await self.manager.evaluate('state', self.qs)
        await self.manager.evaluate('state', self.qs)
        self.loader.assert_called_once()
        state = next(iter(self.resources.snapshot()['reservations'].values()))
        self.assertEqual(state, dict(workload='decision', host_bytes=60, device_bytes={}, active_leases=0, evicting=False, offload_on_handoff=True))
        other = self.resources.reserve('chat', 'llm', host_bytes=60)
        self.assertIsNone(self.manager.agent)
        other.release()
        await self.manager.evaluate('state', self.qs)
        self.assertEqual(self.loader.call_count, 2)

    async def test_admission_failure_precedes_model_allocation(self):
        other = self.resources.reserve('active-chat', 'llm', host_bytes=60)
        with self.assertRaises(RuntimeFailure) as caught:
            await self.manager.evaluate('state', self.qs)
        self.assertEqual(caught.exception.status_code, 503)
        self.loader.assert_not_called()
        other.release()

    async def test_load_failure_releases_budget(self):
        self.loader.side_effect = ValueError('broken checkpoint')
        with self.assertRaises(RuntimeFailure):
            await self.manager.evaluate('state', self.qs)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    async def test_failed_loader_traceback_does_not_keep_allocation_alive(self):
        import weakref
        refs = []
        class Allocation:
            pass
        def allocate_then_fail():
            payload = Allocation()
            refs.append(weakref.ref(payload))
            raise ValueError('failed after allocation')
        def failing_loader():
            try:
                allocate_then_fail()
            except ValueError as error:
                raise RuntimeError('wrapped loader failure') from error
        self.manager.loader = failing_loader
        try:
            await self.manager.evaluate('state', self.qs)
        except RuntimeFailure as retained_error:
            self.assertIsNotNone(retained_error.__cause__)
            self.assertIsNone(refs[0]())
            self.assertEqual(self.resources.snapshot()['reservations'], {})
        else:
            self.fail('expected load failure')

    async def test_no_chat_load_required_and_shared_manager_initialized_once(self):
        runtime = RuntimeManager(resources=self.resources)
        self.manager.resources = None
        with patch('api.services.decisions.runtime_manager', runtime):
            await self.manager.evaluate('state', self.qs)
        self.assertEqual(runtime.state, 'unloaded')
        self.assertIs(self.manager.resources, runtime.ensure_resources())

    async def test_truncation_and_preflight_fail_without_successful_answer(self):
        self.manager.check.side_effect = RuntimeFailure('too long', 422)
        with self.assertRaises(RuntimeFailure):
            await self.manager.evaluate('state', self.qs)
        self.agent.predict.assert_not_called()
        self.manager.check.side_effect = None
        self.agent.predict.return_value['usage'] = {'truncated': True}
        with self.assertRaises(RuntimeFailure) as caught:
            await self.manager.evaluate('state', self.qs)
        self.assertEqual(caught.exception.status_code, 422)

    async def test_cancellation_keeps_active_memory_until_worker_returns(self):
        entered, finish = threading.Event(), threading.Event()
        def predict(*args):
            entered.set()
            finish.wait(5)
            return dict(answers={'urgent': dict(type='noul', noul=.7)}, usage={})
        self.agent.predict.side_effect = predict
        task = asyncio.create_task(self.manager.evaluate('state', self.qs))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(task.done())
        self.assertTrue(self.manager._generation.locked())
        with self.assertRaises(RuntimeFailure) as busy:
            await self.manager.evaluate('state', self.qs)
        self.assertEqual(busy.exception.status_code, 429)
        with self.assertRaises(ResourceExhausted):
            self.resources.reserve('other', 'llm', host_bytes=60)
        finish.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(next(iter(self.resources.snapshot()['reservations'].values()))['active_leases'], 0)
