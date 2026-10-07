import hashlib
import os
from pathlib import Path
from unittest.mock import Mock, patch
import threading
import unittest

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from api.inference.llm.conversation import ConversationCache, tensors_in
from api.inference.llm.generation import autoregressive_generate
from api.inference.resources import ResourceManager


def tiny_model(vocab_size=64):
    config = Qwen3_5MoeTextConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=1,
        linear_num_value_heads=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts=2, num_experts_per_tok=1, layer_types=['linear_attention', 'full_attention'],
        eos_token_id=None, max_position_embeddings=1024,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,2,4]})
    config._attn_implementation = 'eager'
    torch.manual_seed(29)
    return Qwen3_5MoeForCausalLM(config).eval()


# Exact tokenizer/template from the reviewed small checkpoint. No model weights
# are downloaded. Local CPU runs may supply the same hash-checked assets offline.
TOKENIZER_REVISION = '1355db6a052410cfd62085d94b58866fd0f2c3c5'
TOKENIZER_HASHES = {
    'chat_template.jinja': 'e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259',
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b',
}
TEMPLATE = {'enable_thinking': False, 'preserve_thinking': True}


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('KADAN_TEST_QWEN_TOKENIZER') or snapshot_download(
            'nvidia/Qwen3.6-35B-A3B-NVFP4', revision=TOKENIZER_REVISION,
            allow_patterns=list(TOKENIZER_HASHES))
        for filename, expected in TOKENIZER_HASHES.items():
            actual = hashlib.sha256((Path(path) / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f'Pinned tokenizer asset mismatch: {filename}')
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls.model = tiny_model(len(cls.tokenizer))
        cls.model.config.eos_token_id = cls.tokenizer.convert_tokens_to_ids('<|im_end|>')

    def setUp(self):
        self.resources = ResourceManager(16 * 1024**2, {})
        self.cache = ConversationCache(self.resources, max_bytes=256 * 1024, max_entries=2)
        self.addCleanup(self.cache.clear)
        self.question = [{'role': 'user', 'content': 'What is two plus two?'}]

    def ids(self, chat, generation=False, **kwargs):
        options = {**TEMPLATE, **kwargs}
        return self.tokenizer.apply_chat_template(chat, tokenize=True,
            add_generation_prompt=generation, return_tensors='pt', **options)['input_ids'][0].tolist()

    def generate(self, chat, conversation=None, *, answer='Two plus two is four.',
                 eos=True, emit=None, stream=True, cancel=None, template=None):
        # Every forward executes the real tiny hybrid model with the actual
        # tokenizer IDs. Only token selection is controlled, so realistic UTF-8
        # answer round-trips and EOS/length paths are reproducible. Raw model
        # logits and complete state are compared separately below.
        generated = self.tokenizer.encode(answer, add_special_tokens=False)
        script = generated + ([self.model.config.eos_token_id] if eos else [])
        prompt = len(self.ids(chat, generation=True, **(template or {})))
        forward = self.model.forward
        lengths, logits, states, events = [], [], [], []
        def run(*args, **kwargs):
            lengths.append(kwargs['input_ids'].shape[-1])
            output = forward(*args, **kwargs)
            logits.append(output.logits.detach().clone())
            states[:] = [t.clone() for t in tensors_in(output.past_key_values)]
            # Prefill logits are ignored until the consumed sequence reaches
            # the complete prompt. Cache length includes any restored prefix.
            position = output.past_key_values.get_seq_length() - prompt
            choice = script[max(0, min(position, len(script) - 1))]
            output.logits = torch.full_like(output.logits, -1000)
            output.logits[..., choice] = 1000
            return output
        def event(value):
            events.append(value)
            if emit:
                emit(value)
        with patch.object(self.model, 'forward', side_effect=run):
            text = autoregressive_generate(self.model, self.tokenizer, chat, 'cpu',
                max_new_tokens=32 if eos else len(generated), resources=self.resources,
                on_event=event if stream else None, conversation_cache=self.cache,
                conversation_id=conversation, cancel_event=cancel,
                chat_template_kwargs={**TEMPLATE, **(template or {})})
        return text, events, lengths, logits[-1], states

    def assert_state_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for actual, expected in zip(left, right):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_completed_answer_reuse_matches_uncached_logits_state_and_forward_work(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.cache.clear()
                answer, _, first_lengths, _, _ = self.generate(self.question, 'a', stream=streaming)
                entry = self.cache.entries['a']
                saved = [t.clone() for t in tensors_in(entry.state)]
                self.assertTrue(entry.state.layers[0].conv_states)
                self.assertTrue(entry.state.layers[0].recurrent_states)
                expected = self.ids(self.question, True) + self.tokenizer.encode(answer, add_special_tokens=False)
                self.assertEqual(list(entry.tokens), expected)  # EOS was sampled, never consumed.
                self.assertEqual(first_lengths[0], len(self.ids(self.question, True)))
                followup = self.question + [{'role': 'assistant', 'content': answer},
                                           {'role': 'user', 'content': 'What about three plus three?'}]
                full = self.generate(followup, answer='Three plus three is six.', stream=streaming)
                cached = self.generate(followup, 'a', answer='Three plus three is six.', stream=streaming)
                self.assertEqual(cached[0], full[0])
                self.assertEqual(sum(full[2]) - sum(cached[2]), len(expected))
                torch.testing.assert_close(cached[3], full[3], rtol=1e-4, atol=1e-5)
                self.assert_state_equal(cached[4], full[4])
                # Restore copied state; saved tensors themselves were not mutated.
                self.assert_state_equal(list(tensors_in(entry.state)) if entry.state else saved, saved)
                if streaming:
                    self.assertEqual(''.join(e.get('content', '') for e in cached[1]), cached[0])
                    usage = next(e['cache'] for e in cached[1] if 'cache' in e)
                    self.assertEqual(usage['reused_tokens'], len(expected))
                    self.assertEqual(usage['retention_reason'], 'retained')

    def test_length_stop_retains_exact_consumed_state_without_extra_forward(self):
        answer, _, lengths, _, _ = self.generate(self.question, 'a', eos=False)
        generated = self.tokenizer.encode(answer, add_special_tokens=False)
        expected = self.ids(self.question, True) + generated[:-1]
        self.assertEqual(list(self.cache.entries['a'].tokens), expected)
        self.assertEqual(sum(lengths), len(expected))
        followup = self.question + [{'role': 'assistant', 'content': answer},
                                   {'role': 'user', 'content': 'Please explain.'}]
        full = self.generate(followup, eos=False)
        cached = self.generate(followup, 'a', eos=False)
        self.assertEqual(cached[0], full[0])
        self.assertEqual(sum(full[2]) - sum(cached[2]), len(expected))
        self.assert_state_equal(cached[4], full[4])

    def test_default_template_rewrites_history_but_preserving_empty_markers_is_exact(self):
        answer = 'Two plus two is four.'
        consumed = self.ids(self.question, True) + self.tokenizer.encode(answer, add_special_tokens=False)
        followup = self.question + [{'role': 'assistant', 'content': answer},
                                   {'role': 'user', 'content': 'Next question.'}]
        self.assertEqual(self.ids(followup, True)[:len(consumed)], consumed)
        self.assertNotEqual(self.ids(followup, True, preserve_thinking=False)[:len(consumed)], consumed)
        rendered = self.tokenizer.decode(self.ids(followup, True))
        self.assertIn('<think>\n\n</think>\n\n' + answer, rendered)
        self.assertEqual(self.generate(self.question)[0], answer)

    def test_noncanonical_whitespace_declines_instead_of_rewinding_hybrid_state(self):
        # The pinned template trims assistant text. Keeping state after leading
        # whitespace would not represent the next prompt, so report a miss.
        answer, events, _, _, _ = self.generate(self.question, 'a', answer='  Four.')
        self.assertEqual(answer, '  Four.')
        self.assertNotIn('a', self.cache.entries)
        self.assertEqual(next(e['cache']['retention_reason'] for e in events if 'cache' in e),
                         'noncanonical_completed_tokens')

    def test_history_model_and_template_invalidation(self):
        for change in ('history', 'model', 'template'):
            self.generate(self.question, 'a')
            followup = self.question + [{'role': 'assistant', 'content': 'Two plus two is four.'},
                                       {'role': 'user', 'content': 'Next question.'}]
            template = None
            if change == 'history':
                followup[1] = {'role': 'assistant', 'content': 'Changed answer.'}
            elif change == 'model':
                # Model identity is independent of shape or tokenizer identity.
                self.model = tiny_model(len(self.tokenizer))
                self.model.config.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            else:
                template = {'preserve_thinking': False}
            _, events, _, _, _ = self.generate(followup, 'a', template=template)
            self.assertEqual(next(e['cache']['reason'] for e in events if 'cache' in e), 'invalidated')

    def test_cancelled_or_failed_stream_releases_all_state(self):
        for stage in ('content', 'cache', 'finish_reason'):
            self.generate(self.question, 'a')
            def fail(event):
                if stage in event:
                    raise InterruptedError('client disconnected')
            with self.assertRaises(InterruptedError):
                self.generate(self.question, 'a', emit=fail)
            self.assertFalse(self.cache.entries)
            self.assertFalse(self.cache.pending)
            self.assertFalse(self.resources.snapshot()['reservations'])

    def test_lru_pressure_and_explicit_bound_rejection(self):
        for key in ('a', 'b', 'c'):
            self.generate(self.question, key)
        self.assertEqual(list(self.cache.entries), ['b', 'c'])
        reservations = self.resources.snapshot()['reservations']
        self.assertEqual(sum(r['host_bytes'] for r in reservations.values()),
                         sum(e.size for e in self.cache.entries.values()))
        pressure = self.resources.reserve('pressure', 'tts', host_bytes=16 * 1024**2)
        self.assertFalse(self.cache.entries)
        pressure.release()
        self.cache.max_bytes = 1
        actual = self.generate(self.question, 'a')
        reference = self.generate(self.question)
        self.assertEqual(actual[0], reference[0])
        usage = next(e['cache'] for e in actual[1] if 'cache' in e)
        self.assertEqual(usage['stored_tokens'], 0)
        self.assertEqual(usage['retention_reason'], 'limit_exceeded')
        self.assertEqual(usage['limit_bytes'], 1)

    def test_capture_adopts_owned_cpu_state_and_restore_isolates_it(self):
        ids = torch.tensor([self.ids(self.question, True)])
        with torch.inference_mode():
            output = self.model(input_ids=ids, use_cache=True)
            state = output.past_key_values
            self.assertEqual(self.cache.capture('a', ids[0].tolist(), 'id', state, adopt=True), 'retained')
            self.cache.commit('a')
            self.assertIs(self.cache.entries['a'].state, state)
            restored, reused, _ = self.cache.restore('a', ids[0].tolist() + [1], 'id', 'cpu', ids.shape[-1])
        self.assertEqual(reused, ids.shape[-1])
        self.assertIsNot(restored, state)
        self.assert_state_equal(list(tensors_in(restored)), list(tensors_in(state)))
        for left, right in zip(tensors_in(restored), tensors_in(state)):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())

    def test_unmodified_greedy_model_followup_matches_without_scripted_selection(self):
        torch.manual_seed(0)
        self.model = type(self.model)(self.model.config).eval()
        def generate(chat, conversation=None):
            counts = []
            raw_logits = []
            forward = self.model.forward
            def observe(*args, **kwargs):
                counts.append(kwargs['input_ids'].shape[-1])
                result = forward(*args, **kwargs)
                raw_logits[:] = [result.logits.detach().clone()]
                return result
            with patch.object(self.model, 'forward', side_effect=observe):
                text = autoregressive_generate(self.model, self.tokenizer, chat, 'cpu',
                    max_new_tokens=3, resources=self.resources, conversation_cache=self.cache,
                    conversation_id=conversation, chat_template_kwargs=TEMPLATE)
            return text, sum(counts), raw_logits[0]
        answer = generate(self.question, 'a')[0]
        entry = self.cache.entries.get('a')
        # This fixed seed's greedy output must round-trip; fail visibly rather
        # than calling a cache miss proof of completed-answer reuse.
        self.assertIsNotNone(entry)
        retained = len(entry.tokens)
        history = self.question + [{'role': 'assistant', 'content': answer},
                                   {'role': 'user', 'content': 'Explain your answer.'}]
        full = generate(history)
        cached = generate(history, 'a')
        self.assertEqual(cached[0], full[0])
        self.assertEqual(full[1] - cached[1], retained)
        torch.testing.assert_close(cached[2], full[2], rtol=1e-4, atol=1e-5)

    def test_device_retention_admission_and_cpu_fallback_without_cuda_execution(self):
        resources = ResourceManager(16 * 1024**2, {0: 1024**2})
        cache = ConversationCache(resources)
        self.addCleanup(cache.clear)
        # Accounting-only device stand-in: never allocate or execute on CUDA.
        tensor = Mock(device=torch.device('cuda:0'))
        tensor.numel.return_value = 1024
        tensor.element_size.return_value = 4
        state = object()
        with patch('api.inference.llm.conversation.tensors_in', return_value=[tensor]), \
             patch('api.inference.llm.conversation.clone_state', return_value='cpu-copy') as clone:
            self.assertEqual(cache.capture('a', [1, 2], 'id', state, adopt=True), 'retained')
            clone.assert_not_called()
            self.assertIs(cache.pending['a'].state, state)
            usage = cache.commit('a')
            self.assertEqual(usage['device_bytes'], {0: 4096})
            cache.clear()
            active = resources.reserve('busy', 'llm', device_bytes={0: 1024**2})
            with active.lease():
                self.assertEqual(cache.capture('b', [1, 2], 'id', state, adopt=True), 'retained')
            active.release()
            clone.assert_called_once_with(state, torch.device('cpu'))
            self.assertEqual(cache.commit('b')['device_bytes'], {})
        cache.clear()
        self.assertFalse(resources.snapshot()['reservations'])

    def test_fresh_entry_above_old_512_mib_limit_is_accounted_within_new_bound(self):
        resources = ResourceManager(4 * 1024**3, {})
        cache = ConversationCache(resources)
        self.addCleanup(cache.clear)
        tensor = Mock(device=torch.device('cpu'))
        tensor.numel.return_value = 600 * 1024**2
        tensor.element_size.return_value = 1
        with patch('api.inference.llm.conversation.tensors_in', return_value=[tensor]):
            self.assertEqual(cache.capture('long', list(range(23000)), 'id', object(), adopt=True), 'retained')
        usage = cache.commit('long')
        self.assertGreater(usage['host_bytes'], 512 * 1024**2)
        self.assertLess(usage['host_bytes'], usage['limit_bytes'])
        self.assertEqual(sum(r['host_bytes'] for r in resources.snapshot()['reservations'].values()), usage['host_bytes'])
