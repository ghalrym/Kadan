import unittest

from pydantic import ValidationError

from api.pydantic_models.decisions import ScoreQuestion
from api.routes.v1.decisions import DecisionRequest


class DecisionValidationTests(unittest.TestCase):
    def test_decision_keys_must_be_unique(self):
        with self.assertRaises(ValidationError) as caught:
            DecisionRequest(state="Incident", questions=[ScoreQuestion(
                key="severity", instructions="Rate severity", type="Score", levels=["Low", "High"],
            )] * 2)
        error = caught.exception.errors()[0]
        self.assertEqual(error['type'], 'duplicate_question_key')
        self.assertEqual(error['ctx'], {'key': 'severity'})
        self.assertEqual(error['msg'], 'Question key "severity" is used more than once; each question must have a unique key')

    def test_duplicate_option_error_identifies_question_and_option(self):
        with self.assertRaises(ValidationError) as caught:
            DecisionRequest(state='Incident', questions=[{
                'key': 'cause', 'type': 'Choice', 'instructions': 'Pick cause',
                'options': [{'key': 'capacity', 'description': 'Capacity'}] * 2,
            }])
        error = caught.exception.errors()[0]
        self.assertEqual(error['type'], 'duplicate_option_key')
        self.assertEqual(error['ctx'], {'key': 'cause', 'option': 'capacity'})

    def test_question_type_requires_its_own_fields(self):
        with self.assertRaises(ValidationError):
            DecisionRequest(state="Incident", questions=[{
                "key": "severity", "instructions": "Rate severity", "type": "Score",
                "options": [{"key": "high", "description": "High"}],
            }])


class DecisionRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.v1.decisions import router
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.body = {'state': 'The service is down', 'questions': [
            {'key': 'cause', 'type': 'Choice', 'instructions': 'Pick cause', 'options': [{'key': 'capacity', 'description': 'Capacity'}]},
            {'key': 'severity', 'type': 'Score', 'instructions': 'Rate severity', 'levels': ['Low', 'High']},
            {'key': 'urgent', 'type': 'Noul', 'instructions': 'Is urgent?'},
        ]}

    def test_initial_state_is_empty(self):
        self.assertEqual(self.client.get('/v1/decisions').json(), {'state': '', 'questions': [], 'answers': []})

    def test_valid_model_answers_preserve_question_order_and_values(self):
        import json
        from unittest.mock import AsyncMock, patch
        answers = [{'key': 'urgent', 'type': 'Noul', 'value': .85},
                   {'key': 'severity', 'type': 'Score', 'value': .6},
                   {'key': 'cause', 'type': 'Choice', 'value': 'capacity'}]
        with patch('api.routes.v1.decisions.memory_manager.submit', AsyncMock(return_value=list(reversed(answers)))) as model:
            response = self.client.post('/v1/decisions', json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([answer['key'] for answer in response.json()['answers']], ['cause', 'severity', 'urgent'])
        self.assertEqual(response.json()['answers'][2]['value'], .85)
        self.assertEqual(model.call_args.args[0].state, self.body['state'])

    def test_service_errors_preserve_typed_failure_status(self):
        from unittest.mock import AsyncMock, patch
        from api.inference.errors import InferenceFailure
        for status in (422, 502):
            with patch('api.routes.v1.decisions.memory_manager.submit', AsyncMock(side_effect=InferenceFailure('invalid', status))):
                self.assertEqual(self.client.post('/v1/decisions', json=self.body).status_code, status)

    def test_unloaded_and_busy_runtime_status_are_preserved(self):
        from unittest.mock import AsyncMock, patch
        from api.inference.errors import InferenceFailure
        for status in (503, 429):
            with patch('api.routes.v1.decisions.memory_manager.submit', AsyncMock(side_effect=InferenceFailure('not ready', status))):
                self.assertEqual(self.client.post('/v1/decisions', json=self.body).status_code, status)

    def test_duplicate_option_keys_rejected_before_inference(self):
        from unittest.mock import AsyncMock, patch
        self.body['questions'][0]['options'] *= 2
        with patch('api.routes.v1.decisions.memory_manager.submit', AsyncMock()) as model:
            self.assertEqual(self.client.post('/v1/decisions', json=self.body).status_code, 422)
        model.assert_not_called()


class DecisionCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_cancels_and_awaits_runtime_cleanup(self):
        import asyncio
        from unittest.mock import patch
        from fastapi import HTTPException
        from starlette.requests import Request
        from api.routes.v1.decisions import evaluate_decisions
        started, cleaned = asyncio.Event(), asyncio.Event()
        async def complete(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()
        async def receive():
            await started.wait()
            return {'type': 'http.disconnect'}
        request = Request({'type': 'http'}, receive=receive)
        body = DecisionRequest(state='State', questions=[{'key': 'urgent', 'type': 'Noul', 'instructions': 'Urgent?'}])
        with patch('api.routes.v1.decisions.memory_manager.submit', complete):
            with self.assertRaises(HTTPException) as caught:
                await evaluate_decisions(body, request)
        self.assertEqual(caught.exception.status_code, 499)
        self.assertTrue(cleaned.is_set())
