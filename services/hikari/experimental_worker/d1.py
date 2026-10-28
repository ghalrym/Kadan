"""Opt-in text decisions through the pinned d1 System One API, never generation."""

import json
import math
from collections.abc import Callable
from pathlib import Path

MODEL = json.loads(Path(__file__).with_name("catalog_d1.json").read_text())
# Provisional grants, not measured fit; the owner must validate target hardware.
REQUIREMENT = (8 * 1024**3, 16 * 1024**3)
MAX_CONTEXT = 128000


class InputError(ValueError):
    status_code = 422


def prepare_questions(task: dict) -> dict:
    """Map the existing text decision contract without dropping unsupported fields."""
    if set(task) - {"feature", "model", "state", "questions"}:
        raise InputError("d1 supports text Choice, Score and Noul decisions only.")
    if task.get("feature", "decisions") != "decisions":
        raise InputError("d1 supports the decisions feature only.")
    if task.get("model") != MODEL["id"]:
        raise InputError("The explicit pinned d1 model ID is required.")
    state, questions = task.get("state"), task.get("questions")
    if not isinstance(state, str) or not state.strip():
        raise InputError("Decision state must contain nonblank text.")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 8:
        raise InputError("A decision request needs 1 to 8 questions.")
    named = {}
    for question in questions:
        if not isinstance(question, dict):
            raise InputError("A decision question must be an object.")
        key, kind = question.get("key"), question.get("type")
        instructions = question.get("instructions")
        if not isinstance(key, str) or not key.strip() or key in named:
            raise InputError("Decision question keys must be unique and nonblank.")
        if not isinstance(instructions, str) or not instructions.strip():
            raise InputError(f'Instructions for question "{key}" must not be blank.')
        allowed = {"key", "type", "instructions"}
        result = {"instructions": instructions}
        if kind == "Choice":
            allowed.add("options")
            options = question.get("options")
            if not isinstance(options, list) or not 1 <= len(options) <= 64:
                raise InputError(f'd1 question "{key}" needs 1 to 64 Choice options.')
            criteria = {}
            for option in options:
                if not isinstance(option, dict) or set(option) != {
                    "key",
                    "description",
                }:
                    raise InputError(
                        "A Choice option needs only its key and description."
                    )
                label, description = option["key"], option["description"]
                if not isinstance(label, str) or not label.strip() or label in criteria:
                    raise InputError("Choice option keys must be unique and nonblank.")
                if not isinstance(description, str):
                    raise InputError("A Choice description must be text.")
                criteria[label] = description or label
            result.update(type="choice", criteria=criteria)
        elif kind == "Score":
            allowed.add("levels")
            levels = question.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise InputError(
                    f'd1 Score question "{key}" requires 2 to 10 ordered levels.'
                )
            if any(not isinstance(level, str) or not level.strip() for level in levels):
                raise InputError("Every Score level needs a nonblank description.")
            result.update(type="score", criteria=levels)
        elif kind == "Noul":
            allowed.update({"true_when", "false_when", "trueWhen", "falseWhen"})
            guidance = {}
            for label, field, alias, default in (
                ("true", "true_when", "trueWhen", "yes, the statement holds"),
                ("false", "false_when", "falseWhen", "no, the statement does not hold"),
            ):
                if field in question and alias in question:
                    raise InputError(
                        "Use only one spelling of each Noul guidance field."
                    )
                text = question.get(field, question.get(alias, ""))
                if not isinstance(text, str):
                    raise InputError("Noul guidance must be text.")
                guidance[label] = text or default
            result.update(type="noul", criteria=guidance)
        else:
            raise InputError("d1 supports text Choice, Score and Noul decisions only.")
        if set(question) - allowed:
            raise InputError(f'd1 question "{key}" has unsupported fields.')
        named[key] = result
    return named


def probability(value: object) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("d1 returned an invalid probability.")
    return float(value)


def answers(task: dict, result: dict) -> list[dict]:
    """Validate named results and restore the caller's question and option keys."""
    named = prepare_questions(task)
    rows = result.get("answers") if isinstance(result, dict) else None
    if not isinstance(rows, dict) or set(rows) != set(named):
        raise ValueError("d1 returned answers that do not match the questions.")
    output = []
    for key, question in named.items():
        row = rows[key]
        kind = question["type"]
        if not isinstance(row, dict) or row.get("type") != kind:
            raise ValueError("d1 returned an unexpected answer type.")
        if kind == "noul":
            value = probability(row.get("noul"))
            confidence, probabilities = None, None
        else:
            labels = (
                list(question["criteria"])
                if kind == "choice"
                else [str(index) for index in range(len(question["criteria"]))]
            )
            distribution = row.get("probabilities")
            if not isinstance(distribution, dict) or set(distribution) != set(labels):
                raise ValueError("d1 returned probabilities for unexpected labels.")
            probabilities = {
                label: probability(distribution[label]) for label in labels
            }
            if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
                raise ValueError("d1 probabilities do not sum to one.")
            confidence = probability(row.get("confidence"))
            if kind == "choice":
                value = row.get("choice")
                if not isinstance(value, str) or value not in probabilities:
                    raise ValueError("d1 returned an unknown Choice option.")
                expected_confidence = probabilities[value]
                if expected_confidence < max(probabilities.values()) - 1e-4:
                    raise ValueError(
                        "d1 did not select its highest-probability Choice option."
                    )
            else:
                value = row.get("score")
                expected = sum(
                    index * probabilities[label] for index, label in enumerate(labels)
                )
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not math.isclose(value, expected, abs_tol=1e-4)
                ):
                    raise ValueError(
                        "d1 must return the expected zero-based Score, without rounding."
                    )
                expected_confidence = max(probabilities.values())
            if not math.isclose(confidence, expected_confidence, abs_tol=1e-4):
                raise ValueError("d1 confidence does not match its probabilities.")
        output.append(
            {
                "key": key,
                "type": kind.capitalize(),
                "value": value,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )
    return output


def evaluate(model, task: dict) -> list[dict]:
    """Count every complete prompt and use mean scores with no input truncation."""
    named = prepare_questions(task)
    questions = [model.engine.ask(question) for question in named.values()]
    # set_size=1 below reads each question with the full state; tokens() measures
    # the longest of exactly those prompts, including its question and options.
    tokens = model.engine.tokens(task["state"], questions)
    if type(tokens) is not int or not 0 < tokens <= MAX_CONTEXT:
        raise InputError(
            "d1 state and questions exceed the 128000-token context; no input was truncated."
        )
    return answers(
        task, model.system_one(task["state"], named, score_readout="mean", set_size=1)
    )


def execute(
    task: dict, operation: str, model_path: str, emit: Callable[[dict], None]
) -> dict:
    if operation != "generate":
        raise InputError(
            "d1 supports decision evaluation only; it does not generate text."
        )
    prepare_questions(task)
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError(
            "This d1 BF16 worker requires a CUDA device with BF16 support."
        )
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    emit({"state": "running", "model": MODEL["id"]})
    with torch.inference_mode():
        return {"model": MODEL["id"], "answers": evaluate(model, task)}
