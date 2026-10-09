"""Explicit backend selection. Python remains the production default."""
import os

from api.inference.decisions.model import decision_manager as python_manager
from api.inference.decisions.native import NativeDecisionManager
from api.services.runtime import RuntimeFailure


def create_manager():
    backend = os.getenv('KADAN_DECISION_BACKEND', 'python')
    if backend == 'python':
        return python_manager
    if backend == 'native':
        return NativeDecisionManager()
    raise RuntimeFailure('KADAN_DECISION_BACKEND must be python or native.')


decision_manager = create_manager()
