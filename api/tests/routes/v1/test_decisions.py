import unittest

from pydantic import ValidationError

from api.pydantic_models.decisions import ScoreQuestion
from api.routes.v1.decisions import DecisionRequest


class DecisionValidationTests(unittest.TestCase):
    def test_decision_keys_must_be_unique(self):
        with self.assertRaises(ValidationError):
            DecisionRequest(state="Incident", questions=[ScoreQuestion(
                key="severity", instructions="Rate severity", type="Score", levels=["Low", "High"],
            )] * 2)

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
        answers = [{'key': 'urgent', 'type': 'Noul', 'value': True},
                   {'key': 'severity', 'type': 'Score', 'value': 1},
                   {'key': 'cause', 'type': 'Choice', 'value': 'capacity'}]
        with patch('api.routes.v1.decisions.runtime_manager.complete', AsyncMock(return_value=json.dumps({'answers': answers}))) as model:
            response = self.client.post('/v1/decisions', json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual([answer['key'] for answer in response.json()['answers']], ['cause', 'severity', 'urgent'])
        self.assertIs(response.json()['answers'][2]['value'], True)
        self.assertIsNone(model.call_args.args[1])

    def test_invalid_output_returns_502_instead_of_empty_success(self):
        import json
        from unittest.mock import AsyncMock, patch
        for value in ('not json', '{"answers":[]}', '{"answers":[],"answers":[]}',
                      json.dumps({'answers': [{'key': 'urgent', 'type': 'Noul', 'value': 'true'},
                         {'key': 'severity', 'type': 'Score', 'value': True},
                         {'key': 'cause', 'type': 'Choice', 'value': 'unknown'}]})):
            with self.subTest(value=value), patch('api.routes.v1.decisions.runtime_manager.complete', AsyncMock(return_value=value)):
                self.assertEqual(self.client.post('/v1/decisions', json=self.body).status_code, 502)

    def test_unloaded_and_busy_runtime_status_are_preserved(self):
        from unittest.mock import AsyncMock, patch
        from api.services.runtime import RuntimeFailure
        for status in (503, 429):
            with patch('api.routes.v1.decisions.runtime_manager.complete', AsyncMock(side_effect=RuntimeFailure('not ready', status))):
                self.assertEqual(self.client.post('/v1/decisions', json=self.body).status_code, status)

    def test_duplicate_option_keys_rejected_before_inference(self):
        from unittest.mock import AsyncMock, patch
        self.body['questions'][0]['options'] *= 2
        with patch('api.routes.v1.decisions.runtime_manager.complete', AsyncMock()) as model:
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
        async def complete(*args):
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
        with patch('api.routes.v1.decisions.runtime_manager.complete', complete):
            with self.assertRaises(HTTPException) as caught:
                await evaluate_decisions(body, request)
        self.assertEqual(caught.exception.status_code, 499)
        self.assertTrue(cleaned.is_set())


class DecisionAnswerValidationTests(unittest.TestCase):
    def test_each_type_and_key_constraint_rejects_invalid_output(self):
        import json
        from api.routes.v1.decisions import validate_output
        questions = DecisionRequest(state='State', questions=[
            {'key': 'q', 'type': 'Score', 'instructions': 'Rate', 'levels': ['Low', 'High']}
        ]).questions
        for answer in ({'key': 'q', 'type': 'Score', 'value': True},
                       {'key': 'q', 'type': 'Score', 'value': 1.5},
                       {'key': 'q', 'type': 'Score', 'value': -1},
                       {'key': 'q', 'type': 'Score', 'value': 2},
                       {'key': 'missing', 'type': 'Score', 'value': 1},
                       {'key': 'q', 'type': 'Noul', 'value': True},
                       {'key': 'q', 'type': 'Score', 'value': 1, 'confidence': .9}):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                validate_output(json.dumps({'answers': [answer]}), questions)
