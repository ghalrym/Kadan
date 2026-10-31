"""Explicit developer adapter for an already-loaded d1 model, outside worker routing."""

import json
import math
from collections.abc import Callable
from copy import deepcopy
from typing import Protocol


class DecisionModel(Protocol):
    def system_one(
        self, state: str, questions: dict, *, score_readout: str
    ) -> dict: ...


def _text(value, name: str, *, blank: bool = False) -> str:
    if not isinstance(value, str) or (not blank and not value.strip()):
        raise ValueError(f"d1 requires {name} to be a valid string.")
    return value


def _questions(task: dict) -> tuple[str, dict]:
    if not isinstance(task, dict):
        raise TypeError("d1 requires a task mapping.")
    state = _text(task.get("state"), "state")
    questions = task.get("questions")
    if len(state) > 8000:
        raise ValueError("d1 state must fit 8000 characters.")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 8:
        raise ValueError("d1 requires one to eight questions.")
    named = {}
    serialized = []
    for question in questions:
        if not isinstance(question, dict):
            raise TypeError("d1 questions must be mappings.")
        key = _text(question.get("key"), "question key")
        if key in named:
            raise ValueError("d1 question keys must be unique.")
        instructions = _text(question.get("instructions"), "instructions")
        kind = question.get("type")
        row = {"instructions": instructions}
        wire = {"key": key, "instructions": instructions, "type": kind}
        if kind == "Choice":
            options = question.get("options")
            if not isinstance(options, list) or not options:
                raise ValueError("d1 Choice requires options.")
            criteria = {}
            for option in options:
                if not isinstance(option, dict):
                    raise TypeError("d1 options must be mappings.")
                option_key = _text(option.get("key"), "option key")
                if option_key in criteria:
                    raise ValueError("d1 option keys must be unique.")
                criteria[option_key] = _text(
                    option.get("description"), "option description", blank=True
                )
            row.update(type="choice", criteria=criteria)
            wire["options"] = [
                {"key": option_key, "description": description}
                for option_key, description in criteria.items()
            ]
        elif kind == "Score":
            levels = question.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise ValueError(
                    "The d1 trial supports Score rubrics of 2 to 10 levels."
                )
            levels = [_text(level, "Score level") for level in levels]
            row.update(type="score", criteria=levels)
            wire["levels"] = levels
        elif kind == "Noul":
            true_when = _text(
                question.get("trueWhen", question.get("true_when", "")),
                "true guidance",
                blank=True,
            )
            false_when = _text(
                question.get("falseWhen", question.get("false_when", "")),
                "false guidance",
                blank=True,
            )
            row["type"] = "noul"
            if true_when or false_when:
                row["criteria"] = {"true": true_when, "false": false_when}
            wire.update(true_when=true_when, false_when=false_when)
        else:
            raise ValueError("d1 supports Choice, Score and Noul questions only.")
        named[key] = row
        serialized.append(wire)
    payload = json.dumps(
        {"state": state, "questions": serialized},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(payload) > 16000:
        raise ValueError("The d1 request must fit 16000 characters.")
    return state, named


def _number(value, name: str, maximum: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"d1 returned an invalid {name}.")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"d1 returned an invalid {name}.") from exc
    if not math.isfinite(number) or not 0 <= number <= maximum:
        raise ValueError(f"d1 returned an invalid {name}.")
    return number


def _probabilities(value, keys: list[str]) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("d1 returned invalid distribution keys.")
    probabilities = {key: _number(value[key], "probability") for key in keys}
    if not math.isclose(sum(probabilities.values()), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("d1 probabilities must sum to one.")
    return probabilities


def evaluate(
    task: dict, model: DecisionModel, check_cancelled: Callable[[], None]
) -> list[dict]:
    """Translate and validate one explicit trial; cancellation checks may raise.

    This does not load models, reserve memory or interrupt a blocking model call.
    """
    check_cancelled()
    state, questions = _questions(task)
    result = model.system_one(state, deepcopy(questions), score_readout="mean")
    check_cancelled()
    answers = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("d1 must answer every requested question exactly once.")
    rows = []
    for key, question in questions.items():
        answer = answers[key]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("d1 returned an incorrect answer type.")
        row = {"key": key, "type": kind.capitalize()}
        if kind == "choice":
            value = answer.get("choice")
            if not isinstance(value, str) or value not in question["criteria"]:
                raise ValueError("d1 returned an unknown choice.")
            row["value"] = value
            row["probabilities"] = _probabilities(
                answer.get("probabilities"), list(question["criteria"])
            )
        elif kind == "score":
            count = len(question["criteria"])
            probabilities = _probabilities(
                answer.get("probabilities"), [str(index) for index in range(count)]
            )
            value = _number(answer.get("score"), "score", count - 1)
            mean = sum(int(index) * p for index, p in probabilities.items())
            if not math.isclose(value, mean, rel_tol=0, abs_tol=1e-6):
                raise ValueError("d1 Score must be the expected zero-based level.")
            row.update(value=value, probabilities=probabilities)
        else:
            value = _number(answer.get("noul"), "Noul probability")
            row["value"] = value
            if "probabilities" in answer:
                probabilities = _probabilities(
                    answer["probabilities"], ["true", "false"]
                )
                if not math.isclose(
                    value, probabilities["true"], rel_tol=0, abs_tol=1e-6
                ):
                    raise ValueError("d1 Noul must match its true probability.")
                row["probabilities"] = probabilities
        if "confidence" in answer:
            row["confidence"] = _number(answer["confidence"], "confidence")
        rows.append(row)
    return rows
