import threading
from types import SimpleNamespace
import unittest

import torch

from api.inference.llm.generation import autoregressive_generate


class StreamingGenerationTests(unittest.TestCase):
    def run_generation(self, budget=8, emit=None):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return torch.tensor([[7]])
            def decode(self, tokens, **kwargs):
                return ''.join({1:'Hello',2:' world',3:'!',4:''}[int(t)] for t in tokens)
        class Model:
            config = SimpleNamespace(eos_token_id=4, max_position_embeddings=4096)
            count = 0
            def __call__(self, **kwargs):
                self.count += 1
                logits = torch.zeros(1,1,8)
                logits[0,0,self.count] = 10
                return SimpleNamespace(logits=logits,past_key_values='cache')
        return autoregressive_generate(Model(),Tokenizer(),[{'role':'user','text':'hi'}],
            'cpu',max_new_tokens=budget,on_event=emit)

    def test_stream_matches_final_text_and_eos(self):
        events=[]
        text=self.run_generation(emit=events.append)
        self.assertEqual(text,'Hello world!')
        self.assertEqual(''.join(x.get('content','') for x in events),text)
        self.assertEqual(events[-1],{'finish_reason':'stop'})
        self.assertGreater(len(events),2)

    def test_output_budget_is_length_not_stop(self):
        events=[]
        text=self.run_generation(budget=2,emit=events.append)
        self.assertEqual(text,'Hello world')
        self.assertEqual(events[-1],{'finish_reason':'length'})

    def test_disconnected_consumer_interrupts_before_more_generation(self):
        def disconnected(event):
            raise InterruptedError('disconnected')
        with self.assertRaises(InterruptedError):
            self.run_generation(emit=disconnected)
