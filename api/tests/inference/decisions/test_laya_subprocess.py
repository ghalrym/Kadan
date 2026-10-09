"""Synthetic IPC tests: these validate adapter ownership, not model numerics."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from api.inference.decisions.laya_subprocess import LayaSubprocessEvaluator, DEFAULT_HOST_BUDGET_BYTES, resolve_laya_command
from api.inference.decisions.laya_backend import create_laya_evaluator
from api.inference.decisions.decision_requests import DecisionRequests
from api.inference.resources import ResourceManager
from api.routes.v1.decisions import DecisionRequest
from api.inference.errors import InferenceFailure

SCRIPT = '''import json, sys, time
for line in sys.stdin:
    q = json.loads(line)
    state = q['state']
    if state == 'hang': time.sleep(60)
    if len(state.encode('utf-8')) > 16384:
        print('{"error":"decision_empty_or_long_text"}', flush=True); continue
    if state == 'eof': break
    if state == 'large':
        print('x'*70000, flush=True); continue
    if state == 'duplicate':
        print('{"answers":[],"answers":[]}', flush=True); continue
    if state == 'reject':
        print('{"error":"decision_state_token_budget"}', flush=True); continue
    result=[]
    for question in q['questions']:
        kind=question['type']
        value=question['options'][0]['key'] if kind=='Choice' else .5
        probabilities={x['key']: 1/len(question['options']) for x in question['options']} if kind=='Choice' else ({'0': .5, '1': .5} if kind=='Score' else None)
        if kind=='Noul' and question.get('trueWhen') != 'yes': sys.exit(4)
        result.append(dict(key=question['key'],type=kind,value=value,confidence=.5,probabilities=probabilities))
    if state=='wrong': result.reverse()
    print(json.dumps(dict(answers=result)), flush=True)
'''


def body(state='hello café'):
    return DecisionRequest(state=state, questions=[
        dict(key='c', type='Choice', instructions='color?', options=[dict(key='red', description='Red'), dict(key='blue', description='Blue')]),
        dict(key='s', type='Score', instructions='score?', levels=['low', 'high']),
        dict(key='n', type='Noul', instructions='true?', trueWhen='yes', falseWhen='no')])


class DecisionWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.script = self.root / 'worker.py'
        self.script.write_text(SCRIPT)
        self.resources = ResourceManager(DEFAULT_HOST_BUDGET_BYTES * 2, {})
        self.manager = LayaSubprocessEvaluator(self.resources, resolve=lambda: [sys.executable, str(self.script)])

    async def asyncTearDown(self):
        await self.manager.close()
        self.temp.cleanup()

    async def evaluate(self, state='hello café'):
        request = body(state)
        return await self.manager.evaluate(request.state, request.questions)

    async def test_reuse_and_pressure_eviction_confirm_exit_before_release(self):
        result = await self.evaluate()
        worker = self.manager.agent
        self.assertEqual([x.type for x in result], ['Choice', 'Score', 'Noul'])
        self.assertEqual(result[2].probabilities, None)
        self.assertEqual(await self.evaluate(), result)
        self.assertIs(worker, self.manager.agent)
        self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), DEFAULT_HOST_BUDGET_BYTES)
        reservation = self.resources.reserve('pressure', 'image', host_bytes=DEFAULT_HOST_BUDGET_BYTES + 1)
        self.assertIsNotNone(worker.process.poll())
        self.assertIsNone(self.manager.agent)
        reservation.release()
        await self.evaluate()
        self.assertIsNot(self.manager.agent, worker)
        await self.manager.close()
        self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), 0)

    async def test_cancel_reaps_and_releases_before_return(self):
        task = asyncio.create_task(self.evaluate('hang'))
        for _ in range(100):
            if self.manager.agent and self.manager.agent.io_ready:
                break
            await asyncio.sleep(.01)
        worker = self.manager.agent
        self.assertIsNotNone(worker)
        with self.assertRaises(InferenceFailure) as caught:
            await self.evaluate()
        self.assertEqual(caught.exception.status_code, 429)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        self.assertIsNotNone(worker.process.poll())
        self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), 0)
        await self.evaluate()

    async def test_protocol_errors_cleanup_without_fallback(self):
        for state, status in [('large',502), ('duplicate',502), ('wrong',502), ('eof',502), ('reject',422)]:
            with self.subTest(state=state):
                with self.assertRaises(InferenceFailure) as caught:
                    await self.evaluate(state)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), 0)
                self.assertIsNone(self.manager.agent)

    async def test_timeout_is_bounded_and_reaped(self):
        with patch.dict(os.environ, {'KADAN_NATIVE_DECISION_TIMEOUT_SECONDS': '.05'}):
            with self.assertRaises(InferenceFailure) as caught:
                await self.evaluate('hang')
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), 0)

    async def test_uncertain_cleanup_quarantines_budget(self):
        await self.evaluate()
        worker = self.manager.agent
        with patch.object(worker, 'stop', side_effect=RuntimeError('reap failed')):
            with self.assertRaises(RuntimeError):
                await self.manager.close()
            self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), DEFAULT_HOST_BUDGET_BYTES)
            with self.assertRaises(InferenceFailure):
                await self.evaluate()
        await self.manager.close()
        self.assertEqual(sum(row['host_bytes'] for row in self.resources.snapshot()['reservations'].values()), 0)

    async def test_admission_precedes_spawn(self):
        self.manager.resources = ResourceManager(DEFAULT_HOST_BUDGET_BYTES - 1, {})
        with self.assertRaises(InferenceFailure) as caught:
            await self.evaluate()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertIsNone(self.manager.agent)

    def test_explicit_backend_and_preflight_no_fallback(self):
        with patch.dict(os.environ, {'KADAN_DECISION_BACKEND': 'native'}):
            self.assertIsInstance(create_laya_evaluator(), LayaSubprocessEvaluator)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(create_laya_evaluator(), LayaSubprocessEvaluator)
        for backend in ('python', 'invalid', ''):
            with patch.dict(os.environ, {'KADAN_DECISION_BACKEND': backend}):
                with self.assertRaises(InferenceFailure):
                    create_laya_evaluator()
        with patch.dict(os.environ, {'KADAN_NATIVE_DECISION_WORKER': '/missing/worker'}):
            with self.assertRaises(InferenceFailure):
                resolve_laya_command()

    async def test_request_executor_load_park_call_and_close_owns_one_worker(self):
        executor = DecisionRequests(self.manager)
        self.assertIsNone(await executor.load())
        self.assertIsNone(executor.adapter)
        result = await executor(body())
        worker = executor.adapter
        await executor.offload_to_ram()
        self.assertEqual([item['type'] for item in result], ['Choice', 'Score', 'Noul'])
        self.assertIs(executor.adapter, worker)
        await executor.unload()
        self.assertIsNotNone(worker.process.poll())
        self.assertIsNone(executor.adapter)
        self.assertEqual(self.resources.snapshot()['reservations'], {})
