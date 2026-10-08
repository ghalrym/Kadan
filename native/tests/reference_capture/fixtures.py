"""Adapt only generated miniature checkpoints; independent expected equations."""
import contextlib
import io
import json
from pathlib import Path
import runpy

from .artifacts import encode

TESTS = Path(__file__).resolve().parents[1]


def create(root, representative=False, tied=False):
    generator = runpy.run_path(str(TESTS / 'model_fixture.py'))['create']
    header, size = generator(root, 4, representative)
    if tied:
        path = root / 'fixture.safetensors'
        payload = bytearray(path.read_bytes())
        for name, item in header.items():
            if name.endswith('.mlp.gate.weight'):
                a,b = item['data_offsets'];payload[8+size+a:8+size+b] = b'\0' * (b-a)
        path.write_bytes(payload)
    # Local-only tokenizer to exercise the unchanged adapter's loader. No text
    # tokenization is called by the capture wrapper.
    # Imports are delayed so artifact/unit tests need no numerical dependencies.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({f't{i}': i for i in range(16)}, unk_token='t0'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='t0', bos_token='t2', eos_token='t15').save_pretrained(root)
    expected = independent_one_token(representative, tied)
    (root/'one-token-equations.capture').write_bytes(expected)
    return expected


def independent_one_token(representative=False, tied=False):
    """Recompute one prefix from the existing stdlib equations, no HF outputs."""
    with contextlib.redirect_stdout(io.StringIO()):
        module = runpy.run_path(str(TESTS / 'stack_golden.py'))
    g = module['evaluate'].__globals__
    if representative:
        runpy.run_path(str(TESTS / 'model_equations.py'))['configure'](g)
    bf,f = g['bf'],g['f']
    g['dot'] = lambda matrix,x: [bf(sum(bf(f(a))*b for a,b in zip(row,x))) for row in matrix]
    g['head'] = [[bf(f(v*f(.753)/.75)) for v in row] for row in g['head']]
    if tied:g['router'] = [[0]*len(row) for row in g['router']]
    g['inputs'] = [g['embedding'][2]]
    for i in range(4):
        scale=f(.503+.03125*i);g['moe_scale']=f(.503+.015625*i)
        for name,original in g['base_matrices'].items():g[name]=[[bf(f(v*scale)) for v in row] for row in original]
        attention = g['attention_full']() if i==3 else g['attention_linear']()
        mixed = g['mixture'](g['norm'](attention[0][0],g['postnorm']))
        g['inputs'] = [[bf(a+b) for a,b in zip(attention[0][0],mixed)]]
    logits = g['dot'](g['head'],g['norm'](g['inputs'][0],g['final_norm']))
    selected = max(range(16),key=logits.__getitem__)
    return encode(2,selected,selected in (14,15),logits)
