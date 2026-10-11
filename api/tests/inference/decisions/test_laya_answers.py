import unittest

from api.routes.v1.decisions import DecisionRequest
from api.inference.decisions.laya_answers import parse_answer
from api.inference.errors import InferenceFailure


def questions():
    return DecisionRequest(state='Service failed', questions=[
        dict(key='cause', type='Choice', instructions='Cause?', options=[dict(key='load', description='Capacity')]),
        dict(key='severity', type='Score', instructions='Severity?', levels=['Low', 'High']),
        dict(key='urgent', type='Noul', instructions='Urgent?', trueWhen='Immediate action', falseWhen='Can wait'),
    ]).questions


class ContractTests(unittest.TestCase):
    def test_worker_fractional_outputs(self):
        qs = questions()
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
            with self.subTest(answer=answer), self.assertRaises(InferenceFailure) as caught:
                parse_answer(question, answer)
            self.assertEqual(caught.exception.status_code, 502)
