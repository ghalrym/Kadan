"""Explicit CPU-only reference capture for the complete native executor.

Run in the pinned Laya/Torch environment with CHECKPOINT_ROOT. This script is
validation only; the C++ executable never imports Python or these packages.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch
import laya
import laya.common
import transformers.models.modernbert.modeling_modernbert as modernbert

root = Path(sys.argv[1]).resolve()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
agent = laya.load(str(root), device='cpu', fast=False, compile=False)
agent.amp_enabled = False
choice = dict(key='color', type='Choice', instructions='Choose the color of the apple.',
              options=[dict(key='red', description='Red'), dict(key='blue', description='Blue')])
score = dict(key='quality', type='Score', instructions='Rate the product quality.',
             levels=['Broken and unusable', 'Works with problems', 'Excellent and reliable'])
noul = dict(key='red', type='Noul', instructions='The apple is red.',
            trueWhen='The state says the apple is red.', falseWhen='The state says another color.')
cases = [
    dict(name='choice', state='The apple is red.', questions=[choice]),
    dict(name='score', state='The product works perfectly and never fails.', questions=[score]),
    dict(name='noul', state='The apple is blue.', questions=[noul]),
    dict(name='unicode', state='Le café est excellent. Cafe\u0301, 日本語, 🙂.', questions=[score]),
    dict(name='sliding', state='The apple is red. ' * 35, questions=[choice]),
    dict(name='one-option', state='The apple is red.', questions=[dict(choice, options=choice['options'][:1])]),
    dict(name='many-options', state='The selected number is eleven.', questions=[dict(
        key='number', type='Choice', instructions='Choose the selected number.',
        options=[dict(key=str(n), description=str(n)) for n in range(1,12)])]),
    dict(name='mixed', state='The apple is red and excellent.', questions=[choice,score,noul]),
]
results=[]
for case in cases:
    answers=[];sequences=[]
    for q in case['questions']:
        criteria=({o['key']:o['description'] for o in q['options']} if q['type']=='Choice' else
                  q['levels'] if q['type']=='Score' else {'true':q['trueWhen'],'false':q['falseWhen']})
        definition=dict(type=q['type'].lower(),instructions=q['instructions'],criteria=criteria)
        internal=agent._to_internal(definition)
        ids,markers=laya.common.build_sequence(agent.tok,case['state'],internal,512,192)
        sequences.append(dict(ids=ids,markers=markers))
        answer=agent.predict(case['state'],{q['key']:definition})['answers'][q['key']]
        answers.append(dict(key=q['key'],type=q['type'],value=answer[q['type'].lower()],
                            confidence=answer['answer_confidence'],probabilities=answer.get('probabilities')))
    results.append(dict(name=case['name'],request={k:case[k] for k in ['state','questions']},
                        expected=dict(answers=answers),sequences=sequences))
pins={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
      [laya.common.__file__,modernbert.__file__,root/'tokenizer/tokenizer.json',root/'encoder/config.json',root/'rl_agent_config.json']}
print(json.dumps(dict(packages={p:importlib.metadata.version(p) for p in ['torch','transformers','tokenizers','laya']},
                     reference_sha256=pins,cpu_threads=1,cases=results),ensure_ascii=False,indent=2))
