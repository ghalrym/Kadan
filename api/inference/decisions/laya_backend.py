"""Decision inference is owned exclusively by the C++ Laya worker."""
import os

from api.inference.decisions.laya_subprocess import LayaSubprocessEvaluator
from api.inference.errors import InferenceFailure


def create_laya_evaluator():
    if os.getenv('KADAN_DECISION_BACKEND', 'native') != 'native':
        raise InferenceFailure('Decision inference requires the native Laya worker; remove the legacy backend selection.')
    return LayaSubprocessEvaluator()


laya_evaluator = create_laya_evaluator()
