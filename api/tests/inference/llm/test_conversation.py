import threading
import unittest

import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from api.inference.llm.conversation import ConversationCache, clone_state, tensors_in
from api.inference.llm.generation import autoregressive_generate
from api.inference.resources import ResourceManager


def tiny_model():
    config = Qwen3_5MoeTextConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=1,
        linear_num_value_heads=2, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts=2, num_experts_per_tok=1, layer_types=['linear_attention', 'full_attention'],
        eos_token_id=None, max_position_embeddings=1024,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,2,4]})
    config._attn_implementation = 'eager'
    torch.manual_seed(29)
    return Qwen3_5MoeForCausalLM(config).eval()


class Tokens:
    eos_token_id = None
    chat_template = 'stable-test-template'
    def __init__(self, ids):
        self.ids=ids
    def apply_chat_template(self, chat, *, add_generation_prompt, **kwargs):
        return torch.tensor([self.ids + ([60,61] if add_generation_prompt else [])])
    def decode(self, values, **kwargs):
        return ''.join(str(int(x))+' ' for x in values)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.resources=ResourceManager(16*1024**2,{})
        self.cache=ConversationCache(self.resources,max_bytes=256*1024,max_entries=2)
        self.model=tiny_model()
        self.addCleanup(self.cache.clear)

    def generate(self, ids, conversation=None, *, template=None, emit=None):
        tokenizer=Tokens(ids)
        if template is not None:tokenizer.chat_template=template
        events=[]
        def event(value):
            events.append(value)
            if emit:emit(value)
        text=autoregressive_generate(self.model,tokenizer,[{'role':'user','text':'fixture'}], 'cpu',
            max_new_tokens=3, resources=self.resources, on_event=event,
            conversation_cache=self.cache,conversation_id=conversation)
        return text,events

    def test_hybrid_prefix_reuse_matches_full_prefill_and_streamed_output(self):
        ids=list(range(1,21))
        self.generate(ids,'a')
        entry=self.cache.entries['a']
        layers=entry.state.layers
        self.assertTrue(layers[0].conv_states)
        self.assertTrue(layers[0].recurrent_states)
        self.assertTrue(all(t.device.type=='cpu' for t in tensors_in(entry.state)))
        saved=[t.clone() for t in tensors_in(entry.state)]
        full,_=self.generate(ids+[22,23,24,25])
        cached,events=self.generate(ids+[22,23,24,25],'a')
        self.assertEqual(cached,full)
        self.assertEqual(''.join(e.get('content','') for e in events),cached)
        usage=next(e['cache'] for e in events if 'cache' in e)
        self.assertTrue(usage['hit']);self.assertEqual(usage['reused_tokens'],20)
        self.assertEqual(usage['device_bytes'],{})
        for expected,actual in zip(saved,tensors_in(entry.state)):
            torch.testing.assert_close(actual,expected)

    def test_conversation_history_and_template_are_isolated(self):
        ids=list(range(1,21));self.generate(ids,'a')
        _,events=self.generate(ids,'b')
        self.assertFalse(next(e['cache']['hit'] for e in events if 'cache' in e))
        for changed,template in [(list(range(2,22)),None),(ids,'changed-template')]:
            _,events=self.generate(changed,'a',template=template)
            usage=next(e['cache'] for e in events if 'cache' in e)
            self.assertFalse(usage['hit']);self.assertEqual(usage['reason'],'invalidated')

    def test_cancelled_or_failed_stream_invalidates_candidate_and_prior_state(self):
        ids=list(range(1,21));self.generate(ids,'a')
        def fail(event):
            if 'content' in event:raise InterruptedError('client disconnected')
        with self.assertRaises(InterruptedError):self.generate(ids,'a',emit=fail)
        self.assertNotIn('a',self.cache.entries);self.assertFalse(self.cache.pending)
        self.assertFalse(any('conversation' in key for key in self.resources.snapshot()['reservations']))

    def test_lru_and_shared_memory_pressure_release_real_state(self):
        ids=list(range(1,21))
        for key in ('a','b','c'):self.generate(ids,key)
        self.assertEqual(list(self.cache.entries),['b','c'])
        reservations=self.resources.snapshot()['reservations']
        self.assertEqual(sum(r['host_bytes'] for r in reservations.values()),sum(e.size for e in self.cache.entries.values()))
        pressure=self.resources.reserve('pressure','tts',host_bytes=16*1024**2)
        self.assertFalse(self.cache.entries)
        pressure.release()

    def test_budget_can_decline_retention_without_changing_output(self):
        self.cache.max_bytes=1
        actual,events=self.generate(list(range(1,21)),'a')
        reference,_=self.generate(list(range(1,21)))
        self.assertEqual(actual,reference);self.assertFalse(self.cache.entries)
        self.assertEqual(next(e['cache']['stored_tokens'] for e in events if 'cache' in e),0)

    def test_hybrid_resumed_logits_and_recurrent_state_match_uninterrupted_forward(self):
        ids=torch.tensor([list(range(1,29))])
        with torch.inference_mode():
            full=self.model(input_ids=ids,use_cache=True)
            prefix=self.model(input_ids=ids[:,:20],use_cache=True)
            self.cache.capture('a',ids[0,:20].tolist(),'identity',prefix.past_key_values)
            self.cache.commit('a')
            restored,reused,_=self.cache.restore('a',ids[0].tolist(),'identity','cpu',27)
            self.assertEqual(reused,20)
            resumed=self.model(input_ids=ids[:,20:],past_key_values=restored,use_cache=True)
        torch.testing.assert_close(resumed.logits,full.logits[:,20:],rtol=1e-4,atol=1e-5)
        left=list(tensors_in(resumed.past_key_values));right=list(tensors_in(full.past_key_values))
        self.assertEqual(len(left),len(right))
        for actual,expected in zip(left,right):
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)

    def test_edit_to_previous_history_invalidates_even_before_snapshot_boundary(self):
        ids=list(range(1,21));self.generate(ids,'a')
        old=[{'role':'user','content':'old'}]
        self.cache.commit('a',history=old)
        state,reused,reason=self.cache.restore('a',ids+[60,61],self.cache.entries['a'].identity,'cpu',20,
                                              history=[{'role':'user','content':'edited'}])
        self.assertIsNone(state);self.assertEqual(reused,0);self.assertEqual(reason,'invalidated')
        self.assertNotIn('a',self.cache.entries)
