"""Validate typed Laya worker answers at the API boundary."""
import math

from api.pydantic_models.decisions import ChoiceAnswer, ScoreAnswer, NoulAnswer
from api.inference.errors import InferenceFailure


def parse_answer(question, answer):
    """Validate typed decision output without boolean or integer coercion."""
    kind = question.type.lower()
    if not isinstance(answer, dict) or answer.get('type') != kind:
        raise InferenceFailure('Laya returned an invalid answer type.', 502)
    value = answer.get(kind)
    numeric = lambda number: type(number) in (int, float) and math.isfinite(number)
    if question.type == 'Choice':
        keys = {option.key for option in question.options}
        valid = isinstance(value, str) and value in keys
        cls = ChoiceAnswer
    elif question.type == 'Score':
        keys = {str(index) for index in range(len(question.levels))}
        valid = numeric(value) and 0 <= value <= len(question.levels) - 1
        cls = ScoreAnswer
    else:
        keys = None
        valid = numeric(value) and 0 <= value <= 1
        cls = NoulAnswer
    confidence = answer.get('answer_confidence')
    probabilities = answer.get('probabilities')
    if confidence is not None and (not numeric(confidence) or not 0 <= confidence <= 1):
        valid = False
    if probabilities is not None:
        if (not isinstance(probabilities, dict) or (keys is not None and set(probabilities) != keys)
                or not probabilities or any(not numeric(p) or not 0 <= p <= 1 for p in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > .01):
            valid = False
    if not valid:
        raise InferenceFailure('Laya returned an invalid typed value or probability.', 502)
    return cls(key=question.key, type=question.type, value=value,
               confidence=confidence, probabilities=probabilities)
