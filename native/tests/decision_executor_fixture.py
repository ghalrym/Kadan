"""Small deterministic full-model fixture; no ML packages or downloads in CI."""
import hashlib
import fcntl
import time
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import tempfile


def make_fixture(root):
    (root / 'encoder').mkdir(parents=True)
    (root / 'tokenizer').mkdir()
    config = dict(encoder='local', head_layers=2, max_len=512, head_max_len=192,
                  max_prefixes=6, act_costs={'escalate': .5}, temperature=[1.2, 1.5, 2.0],
                  temperature_by_options={'choice:2': 1.8})
    encoder = dict(model_type='modernbert', hidden_size=64, intermediate_size=80,
                   num_hidden_layers=3, num_attention_heads=2, vocab_size=262,
                   hidden_activation='gelu', attention_bias=False, mlp_bias=False,
                   norm_bias=False, norm_eps=1e-5, local_attention=8,
                   max_position_embeddings=512, bos_token_id=256, eos_token_id=257, pad_token_id=259, cls_token_id=256,
                   sep_token_id=257, layer_types=['full_attention', 'sliding_attention', 'sliding_attention'],
                   rope_parameters={k: dict(rope_type='default', rope_theta=v) for k, v in
                                    [('full_attention', 160000), ('sliding_attention', 10000)]})
    (root / 'rl_agent_config.json').write_text(json.dumps(config))
    (root / 'encoder/config.json').write_text(json.dumps(encoder))
    alphabet = []
    extra = 256
    for b in range(256):
        if 33 <= b <= 126 or 161 <= b <= 172 or b >= 174:
            alphabet.append(chr(b))
        else:
            alphabet.append(chr(extra))
            extra += 1
    special = ['[CLS]', '[SEP]', '[MASK]', '[PAD]']
    vocab = {c: i for i, c in enumerate(alphabet + special + ['ab', 'abc'])}
    tokenizer = dict(version='1.0', truncation=None, padding=None,
                     normalizer=dict(type='NFC'), pre_tokenizer=dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=True),
                     post_processor=None, decoder=dict(type='ByteLevel', add_prefix_space=True, trim_offsets=True, use_regex=True),
                     added_tokens=[dict(id=256+i, content=s, single_word=False, lstrip=s=='[MASK]', rstrip=False, normalized=False, special=True) for i, s in enumerate(special)],
                     model=dict(type='BPE', dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab=vocab, merges=[['a','b'], ['ab','c']]))
    (root / 'tokenizer/tokenizer.json').write_text(json.dumps(tokenizer))
    (root / 'tokenizer/tokenizer_config.json').write_text(json.dumps(dict(
        tokenizer_class='PreTrainedTokenizerFast', cls_token='[CLS]', sep_token='[SEP]',
        mask_token='[MASK]', pad_token='[PAD]', model_max_length=512)))
    shapes = {}
    def add(name, *shape):
        shapes[name] = shape
    h, inter = 64, 80
    add('temperature', 3)
    add('encoder.embeddings.tok_embeddings.weight', 262, h)
    add('encoder.embeddings.norm.weight', h)
    add('encoder.final_norm.weight', h)
    add('type_emb.weight', 3, h)
    for layer in range(3):
        p = f'encoder.layers.{layer}'
        if layer:
            add(p + '.attn_norm.weight', h)
        add(p + '.mlp_norm.weight', h)
        add(p + '.attn.Wqkv.weight', 3*h, h)
        add(p + '.attn.Wo.weight', h, h)
        add(p + '.mlp.Wi.weight', 2*inter, h)
        add(p + '.mlp.Wo.weight', h, inter)
    for layer in range(2):
        p = f'head.layers.{layer}'
        for norm in ['norm1', 'norm2']:
            add(p + f'.{norm}.weight', h)
            add(p + f'.{norm}.bias', h)
        for name, out, inp in [('self_attn.in_proj', 3*h, h), ('self_attn.out_proj', h, h), ('linear1', 4*h, h), ('linear2', h, 4*h)]:
            sep = '_' if name.endswith('in_proj') else '.'
            add(p + '.' + name + sep + 'weight', out, inp)
            add(p + '.' + name + sep + 'bias', out)
    add('scorer.0.weight', h)
    add('scorer.0.bias', h)
    for name, out, inp in [('scorer.1', h, h), ('scorer.3', 1, h), ('act_head.0', 256, h+4), ('act_head.2', 2, 256)]:
        add(name + '.weight', out, inp)
        add(name + '.bias', out)
    metadata, data = {}, bytearray()
    for name, shape in shapes.items():
        rng = random.Random(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little'))
        base = 1.0 if name.endswith('.weight') and ('norm' in name or name == 'scorer.0.weight') else 0.0
        values = [base + rng.uniform(-.15, .15) for _ in range(math.prod(shape))]
        payload = struct.pack('<' + 'e' * len(values), *values)
        metadata[name] = dict(dtype='F16', shape=shape, data_offsets=[len(data), len(data)+len(payload)])
        data.extend(payload)
    header = json.dumps(metadata, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 8)
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + data)


def cli_checks(worker, root):
    request = dict(state='red apple', questions=[dict(key='x', type='Choice', instructions='color?', options=[dict(key='red', description='red'), dict(key='blue', description='blue')])])
    encoded = json.dumps(request)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([worker, str(root)], input=encoded+'\n{}\n'+encoded, text=True, capture_output=True, env=env, timeout=20)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 3 and rows[0] == rows[2] and 'error' in rows[1], rows
    assert 'resident_bytes=0' in result.stderr, result.stderr
    # Reject malformed model metadata and a nonfinite weight without leaking the
    # queue-owned load reservation. Mutations affect only this temporary fixture.
    config_path = root / 'encoder/config.json'
    original = config_path.read_text()
    bad = json.loads(original)
    bad['hidden_size'] = 64.5
    config_path.write_text(json.dumps(bad))
    failed = subprocess.run([worker, str(root)], input=encoded+'\n', text=True, capture_output=True, env=env, timeout=5)
    config_path.write_text(original)
    assert 'decision_integer_config' in failed.stdout and 'resident_bytes=0' in failed.stderr, failed
    weights_path = root / 'model.safetensors'
    weights = weights_path.read_bytes()
    header_size = struct.unpack('<Q', weights[:8])[0]
    header = json.loads(weights[8:8+header_size])
    offset = 8 + header_size + header['encoder.embeddings.norm.weight']['data_offsets'][0]
    corrupt = bytearray(weights)
    corrupt[offset:offset+2] = struct.pack('<H', 0x7c00)
    weights_path.write_bytes(corrupt)
    failed = subprocess.run([worker, str(root)], input=encoded+'\n', text=True, capture_output=True, env=env, timeout=5)
    weights_path.write_bytes(weights)
    assert 'decision_nonfinite_weight' in failed.stdout and 'resident_bytes=0' in failed.stderr, failed
    # A signalled worker must wake even while stdin remains open and idle.
    child = subprocess.Popen([worker, str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    child.stdin.write(encoded+'\n'); child.stdin.flush()
    assert json.loads(child.stdout.readline()) == rows[0]
    child.send_signal(signal.SIGTERM)
    stdout, stderr = child.communicate(timeout=5)
    assert child.returncode == 0 and not stdout and 'resident_bytes=0' in stderr, (stdout, stderr)
    result = subprocess.run([worker, str(root)], input='x'*65537, text=True, capture_output=True, env=env, timeout=5)
    assert result.returncode != 0 and not result.stdout and 'resident_bytes=0' in result.stderr, result
    # A broken output pipe still acknowledges physical cleanup.
    child = subprocess.Popen([worker, str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    child.stdout.close()
    child.stdin.write(encoded+'\n'); child.stdin.close()
    stderr = child.stderr.read()
    assert child.wait(timeout=5) != 0 and 'resident_bytes=0' in stderr, stderr

    # Fill stdout before spawning: leave its reader open and undrained throughout
    # SIGTERM and wait. Unlike communicate(), wait cannot unblock publication.
    reader, writer = os.pipe()
    fcntl.fcntl(writer, fcntl.F_SETFL, os.O_NONBLOCK)
    try:
        while True:
            os.write(writer, b'x' * 4096)
    except BlockingIOError:
        pass
    child = subprocess.Popen([worker, str(root)], stdin=subprocess.PIPE,
                             stdout=writer, stderr=subprocess.PIPE, text=True, env=env)
    os.close(writer)
    try:
        child.stdin.write(encoded + '\n'); child.stdin.flush()
        time.sleep(1)
        assert child.poll() is None
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=5) == 0
        assert 'resident_bytes=0' in child.stderr.read()
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)
        child.stdin.close(); child.stderr.close(); os.close(reader)


if __name__ == '__main__':
    if sys.argv[1] == '--write':
        make_fixture(Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory(prefix='kadan-decision-test-') as directory:
            root = Path(directory)
            make_fixture(root)
            result = subprocess.run([sys.argv[1], str(root)], check=False, capture_output=True, text=True, timeout=30, env=dict(os.environ, OPENBLAS_NUM_THREADS='1'))
            assert result.returncode == 0, (result.stdout, result.stderr)
            # Independently captured with laya 0.3.27 / transformers 5.17.0 /
            # torch 2.14.1 on CPU, eager FP32, from this exact F16 fixture.
            expected = dict(answers=[
                dict(key='x', type='Choice', value='red', confidence=.5003, probabilities=dict(red=.5003, blue=.4997)),
                dict(key='s', type='Score', value=.9899, confidence=.339, probabilities={'0':.339, '1':.3322, '2':.3289}),
                dict(key='n', type='Noul', value=.4918, confidence=.5082, probabilities=None),
            ])
            assert json.loads(result.stdout) == expected, result.stdout
            cli_checks(sys.argv[2], root)
