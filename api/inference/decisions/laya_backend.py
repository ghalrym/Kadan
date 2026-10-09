"""Explicit backend selection. Python remains the production default."""
import os

from api.inference.decisions.laya_python import python_laya_evaluator
from api.inference.decisions.laya_subprocess import LayaSubprocessEvaluator
from api.inference.errors import InferenceFailure


def create_laya_evaluator():
    backend = os.getenv('KADAN_DECISION_BACKEND', 'python')
    if backend == 'python':
        return python_laya_evaluator
    if backend == 'native':
        return LayaSubprocessEvaluator()
    raise InferenceFailure('KADAN_DECISION_BACKEND must be python or native.')


laya_evaluator = create_laya_evaluator()
