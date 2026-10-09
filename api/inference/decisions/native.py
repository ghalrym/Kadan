"""Compatibility imports for the decision worker adapter. New code uses worker."""
from api.inference.decisions.worker import (
    DEFAULT_HOST_BUDGET_BYTES as RAM_BYTES,
    MAX_FRAME_BYTES as FRAME_BYTES,
    DecisionWorkerManager as NativeDecisionManager,
    DecisionWorkerProcess as DecisionProcess,
    decode_response_frame as decode,
    parse_worker_answers as answers_for,
    resolve_worker_command as command,
)

__all__ = [
    'RAM_BYTES', 'FRAME_BYTES', 'NativeDecisionManager', 'DecisionProcess',
    'decode', 'answers_for', 'command',
]
