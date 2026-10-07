"""Lazy, process-local Laya CPU residency under Kadan's shared RAM admission.

KADAN_LAYA_MODEL and KADAN_LAYA_REVISION select an immutable Hub checkpoint;
custom Hub IDs require an explicit commit SHA. A local model directory may also
be supplied. This default is the public general checkpoint, not a claim about
which checkpoint an existing local Laya server uses. KADAN_LAYA_RAM_BYTES is a
conservative admission budget (default 4 GiB), not a measured RSS or hard cap.
It stays reserved while idle, covering FP32 weights, load transients, tokenizer
and a single question's workspace. Idle models may be evicted and lazy-reloaded.
"""
import asyncio
from contextlib import suppress
import gc
import math
import json
import logging
import sys
import traceback
import os
from pathlib import Path
import re
import threading

from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted
from api.pydantic_models.decisions import ChoiceAnswer, ScoreAnswer, NoulAnswer
from api.services.runtime import RuntimeFailure, runtime_manager
from api.services.model_downloads import model_manager

log = logging.getLogger(__name__)

DEFAULT_MODEL = 'convaiinnovations/laya'
DEFAULT_REVISION = '7b928d828b7b0e022f929d9bd2e44165aa270148'
DEFAULT_RAM_BYTES = 4 * 1024 ** 3


def clear_failure_frames(error):
    """Release failed allocations retained anywhere in a chained loader exception."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))
        pending.extend(getattr(current, 'exceptions', ()))


def load_laya():
    """Resolve pinned files and load CPU eager inference without an external server."""
    try:
        import laya
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise RuntimeFailure(f'Decision runtime import failed: {exc}') from exc
    model = os.environ.get('KADAN_LAYA_MODEL', DEFAULT_MODEL)
    revision = os.environ.get('KADAN_LAYA_REVISION', DEFAULT_REVISION if model == DEFAULT_MODEL else '')
    path = Path(model).expanduser()
    if not path.is_dir():
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise RuntimeFailure('KADAN_LAYA_REVISION must pin the selected Hub model to a commit SHA.')
        try:
            path = Path(snapshot_download(model, revision=revision, cache_dir=model_manager.root / 'hub',
                allow_patterns=['rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*']))
        except LocalEntryNotFoundError as exc:
            raise RuntimeFailure('Laya checkpoint is unavailable in the persistent model cache. '
                'Download the pinned checkpoint or configure KADAN_LAYA_MODEL with its complete local directory.') from exc
    # Laya otherwise falls back to the encoder named in its training config.
    for name in ('rl_agent_config.json', 'model.safetensors', 'tokenizer/tokenizer.json', 'encoder/config.json'):
        if not (path / name).is_file():
            raise RuntimeFailure(f'Laya checkpoint is incomplete: missing {name}.')
    validate_checkpoint_budget(path)
    try:
        return laya.load(str(path.resolve()), device='cpu', fast=False, compile=False)
    except Exception:
        clear_tokenizer_cache(directory=path / 'tokenizer')
        raise


def clear_tokenizer_cache(tokenizer=None, directory=None):
    """Release this checkpoint's upstream strong tokenizer cache on eviction/failure."""
    module = sys.modules.get('laya.agent')
    if module is not None:
        with module._TOKENIZERS_LOCK:
            for key, value in list(module._TOKENIZERS.items()):
                if value is tokenizer or (directory is not None and key[0] == str(directory.resolve())):
                    del module._TOKENIZERS[key]


def validate_checkpoint_budget(path):
    """Reject unsupported dimensions and insufficient peak RAM before tensor allocation.

    Support ModernBERT checkpoints no larger than the published large encoder.
    The estimate includes FP32 model storage, checkpoint mapping and 1 GiB for
    tokenizer/runtime/single-question workspace; it is not a measured peak.
    """
    from safetensors import safe_open
    config = json.loads((path / 'rl_agent_config.json').read_text())
    encoder = json.loads((path / 'encoder/config.json').read_text())
    limits = dict(hidden_size=1024, num_hidden_layers=28, intermediate_size=4096,
                  vocab_size=65536, max_position_embeddings=8192, num_attention_heads=32)
    if (encoder.get('model_type') != 'modernbert'
            or any(type(encoder.get(key)) is not int or not 0 < encoder[key] <= limit for key, limit in limits.items())
            or type(config.get('max_len', 512)) is not int or not 0 < config.get('max_len', 512) <= 1024
            or type(config.get('head_max_len', 192)) is not int or not 0 < config.get('head_max_len', 192) <= 1024
            or type(config.get('head_layers', 2)) is not int or not 0 <= config.get('head_layers', 2) <= 2
            or len(config.get('act_costs', {})) > 8):
        raise RuntimeFailure('Unsupported Laya checkpoint dimensions; use a compatible ModernBERT checkpoint.')
    with safe_open(path / 'model.safetensors', framework='pt', device='cpu') as weights:
        parameter_bytes = sum(math.prod(weights.get_slice(key).get_shape()) * 4 for key in weights.keys())
    budget = int(os.environ.get('KADAN_LAYA_RAM_BYTES', DEFAULT_RAM_BYTES))
    if parameter_bytes > 500_000_000 * 4 or parameter_bytes + (path / 'model.safetensors').stat().st_size + 1024 ** 3 > budget:
        raise RuntimeFailure('Laya checkpoint exceeds its configured RAM admission budget.')


def translate_questions(questions):
    """Keep Choice labels, ordered Score criteria and Noul truth criteria intact."""
    result = {}
    for question in questions:
        if question.type == 'Choice':
            criteria = {option.key: option.description for option in question.options}
        elif question.type == 'Score':
            criteria = question.levels
        else:
            criteria = {'true': question.true_when, 'false': question.false_when}
        result[question.key] = dict(type=question.type.lower(), instructions=question.instructions, criteria=criteria)
    return result


def preflight(agent, state, questions):
    """Reject any state, instruction or option truncation using the pinned tokenizer.

    Laya's usage flags cover state truncation but not all head truncation. Compare
    its actual head with the uncropped token sequence before running the model.
    These private helpers are tied to the exact dependency commit in requirements.
    """
    from laya.common import build_head, render_options, encode_text, _encode_question_text
    for key, definition in questions.items():
        agent._check_question(key, definition)
        question = agent._to_internal(definition)
        tok = agent.tok
        # Upstream replaces literal mask tokens. Refuse that lossy transformation.
        options = render_options(question)
        if tok.mask_token in state or tok.mask_token in question['ins'] or any(tok.mask_token in option for option in options):
            raise RuntimeFailure('State and questions cannot contain the tokenizer mask token.', 422)
        full = [tok.cls_token_id] + _encode_question_text(
            tok, f"{question['t']} question: {question['ins']}", add_special_tokens=False) + [tok.sep_token_id]
        for option in options:
            full += [tok.mask_token_id] + _encode_question_text(tok, ' ' + option, add_special_tokens=False)
        full += [tok.sep_token_id]
        actual, _, _ = build_head(tok, question, agent.cfg.get('head_max_len', 192))
        state_ids = encode_text(tok, state, add_special_tokens=False)['input_ids']
        if full != actual or len(full) + len(state_ids) + 1 > agent.cfg.get('max_len', 512):
            raise RuntimeFailure(f'Question {key!r} or state exceeds the Laya token budget; shorten it.', 422)


def parse_answer(question, answer):
    """Validate native typed output without boolean or integer coercion."""
    kind = question.type.lower()
    if not isinstance(answer, dict) or answer.get('type') != kind:
        raise RuntimeFailure('Laya returned an invalid answer type.', 502)
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
        raise RuntimeFailure('Laya returned an invalid typed value or probability.', 502)
    return cls(key=question.key, type=question.type, value=value,
               confidence=confidence, probabilities=probabilities)


class DecisionManager:
    def __init__(self, loader=load_laya, resources=None, check=preflight, ram_bytes=None):
        """Create an unloaded specialist; injected dependencies allow CPU-free lifecycle tests."""
        self.loader, self.resources, self.check = loader, resources, check
        self.ram_bytes = ram_bytes
        self.agent = None
        self.reservation = None
        self._generation = asyncio.Lock()
        self._owner = f'decisions:{id(self)}'

    def _evict(self):
        """Drop CPU model ownership only when the resource manager sees no active lease."""
        if self.agent is not None:
            clear_tokenizer_cache(tokenizer=getattr(self.agent, 'tok', None))
        self.agent = None
        gc.collect()

    def _run(self, state, questions, cancel):
        """Keep loading and each prediction leased until the worker truly finishes."""
        resources = self.resources or runtime_manager.ensure_resources()
        self.resources = resources
        try:
            if self.agent is None:
                budget = self.ram_bytes
                if budget is None:
                    try:
                        budget = int(os.environ.get('KADAN_LAYA_RAM_BYTES', DEFAULT_RAM_BYTES))
                    except ValueError as exc:
                        raise RuntimeFailure('KADAN_LAYA_RAM_BYTES must be an integer byte budget.') from exc
                    if budget < DEFAULT_RAM_BYTES:
                        raise RuntimeFailure('Laya RAM admission must reserve at least 4 GiB including transient workspace.')
                if self.reservation is not None:
                    self.reservation.release()
                self.reservation = resources.reserve(self._owner, 'decision', host_bytes=budget,
                                                    evict=self._evict, cancel_event=cancel)
            # Eviction between reserve() and lease() safely rejects this attempt;
            # no allocation occurs before the lease is acquired.
            with self.reservation.lease(cancel):
                if self.agent is None:
                    self.agent = self.loader()
                if cancel.is_set():
                    raise ResourceCancelled('Decision cancelled')
                if questions is None:
                    return self.agent
                definitions = translate_questions(questions)
                self.check(self.agent, state, definitions)
                answers = []
                # One question at a time bounds activation workspace independently
                # of the request's question count; all questions retain the full state.
                for question in questions:
                    if cancel.is_set():
                        raise ResourceCancelled('Decision cancelled')
                    document = self.agent.predict(state, {question.key: definitions[question.key]})
                    usage = document.get('usage', {})
                    if usage.get('truncated') or usage.get('state_tokens_dropped') or usage.get('truncated_questions') or usage.get('options'):
                        raise RuntimeFailure('Laya truncated the input; shorten the state or question.', 422)
                    output = document.get('answers')
                    if not isinstance(output, dict) or set(output) != {question.key}:
                        raise RuntimeFailure('Laya returned mismatched question keys.', 502)
                    answers.append(parse_answer(question, output[question.key]))
                return answers
        except (ResourceBusy, ResourceExhausted) as exc:
            raise RuntimeFailure(str(exc), 503) from exc
        except ResourceCancelled:
            return []
        except RuntimeFailure as exc:
            clear_failure_frames(exc)
            raise
        except Exception as exc:
            # A failed constructor's traceback can own its partially loaded model.
            # Clear completed frames before returning the reservation to the pool.
            log.exception('Native Laya evaluation failed')
            clear_failure_frames(exc)
            gc.collect()
            raise RuntimeFailure('CPU Laya evaluation failed; check checkpoint configuration and runtime dependencies.') from exc
        finally:
            # Failed construction must not retain an empty, non-evictable budget.
            if self.agent is None and self.reservation is not None:
                self.reservation.release()
                self.reservation = None

    async def load(self):
        """Acquire the same generation gate and host lease without making a prediction."""
        async with self._generation:
            cancel = threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(self._run, None, None, cancel))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancel.set()
                while not worker.done():
                    with suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(worker)
                with suppress(Exception):
                    worker.result()
                raise

    async def evaluate(self, state, questions):
        """Reject overlap and wait for synchronous CPU work after cancellation."""
        if self._generation.locked():
            raise RuntimeFailure('A CPU decision evaluation is already active.', 429)
        async with self._generation:
            cancel = threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(self._run, state, questions, cancel))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancel.set()
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                # Retrieve any failure after ownership has actually returned.
                with suppress(Exception):
                    worker.result()
                raise

    async def close(self):
        """Wait for active inference before releasing CPU residency at shutdown."""
        async with self._generation:
            await asyncio.to_thread(self._close)

    def _close(self):
        self._evict()
        if self.reservation is not None:
            self.reservation.release()
            self.reservation = None


decision_manager = DecisionManager()
