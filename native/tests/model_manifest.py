"""Tiny local-file fixtures only. No database, model download or GPU access."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

inspect, unit = sys.argv[1:]


def fixture():
    text = dict(model_type='qwen3_5_moe_text', dtype='bfloat16', hidden_act='silu',
                mamba_ssm_dtype='float32', attention_bias=False, attention_dropout=0,
                attn_output_gate=True, tie_word_embeddings=False, use_cache=True,
                hidden_size=16, vocab_size=32, num_hidden_layers=2, num_experts=2,
                num_experts_per_tok=1, moe_intermediate_size=16, shared_expert_intermediate_size=16,
                num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=4,
                max_position_embeddings=128, bos_token_id=1, eos_token_id=2,
                rms_norm_eps=1e-6, partial_rotary_factor=.5, full_attention_interval=2,
                layer_types=['linear_attention', 'full_attention'],
                rope_parameters=dict(rope_type='default', mrope_interleaved=True,
                                     partial_rotary_factor=.5, rope_theta=10000, mrope_section=[1, 1, 0]))
    declarations, tensors = {}, {}

    def add(name, dtype, shape):
        shard = 'b.safetensors' if '.layers.1.' in name or name.startswith('mtp.') else 'a.safetensors'
        tensors[name] = [dtype, shape, shard]

    def dense(name, shape):
        add(name, 'BF16', shape)

    def projection(prefix, rows, columns, fp4, declaration=None):
        declarations[declaration or prefix] = {'quant_algo': 'W4A16_NVFP4' if fp4 else 'FP8'}
        if fp4:
            declarations[declaration or prefix]['group_size'] = 16
        add(prefix + '.weight', 'U8' if fp4 else 'F8_E4M3', [rows, columns // 2 if fp4 else columns])
        add(prefix + '.weight_scale', 'F8_E4M3' if fp4 else 'F32', [rows, columns // 16] if fp4 else [])
        if fp4:
            add(prefix + '.weight_scale_2', 'F32', [])
        add(prefix + '.input_scale', 'F32', [])

    dense('model.language_model.embed_tokens.weight', [32, 16])
    dense('model.language_model.norm.weight', [16])
    projection('lm_head', 32, 16, True)
    for layer in range(2):
        base = f'model.language_model.layers.{layer}'
        dense(base + '.input_layernorm.weight', [16])
        dense(base + '.post_attention_layernorm.weight', [16])
        dense(base + '.mlp.gate.weight', [2, 16])
        dense(base + '.mlp.shared_expert_gate.weight', [1, 16])
        if layer == 0:
            for role, rows, columns in [('in_proj_qkv', 64, 16), ('in_proj_z', 32, 16), ('out_proj', 16, 32)]:
                projection(base + '.linear_attn.' + role, rows, columns, False)
            for role, shape in [('in_proj_a.weight', [4, 16]), ('in_proj_b.weight', [4, 16]),
                                ('A_log', [4]), ('dt_bias', [4]), ('norm.weight', [8]), ('conv1d.weight', [64, 1, 4])]:
                dense(base + '.linear_attn.' + role, shape)
        else:
            for role, rows, columns in [('q_proj', 32, 16), ('k_proj', 8, 16), ('v_proj', 8, 16), ('o_proj', 16, 16)]:
                projection(base + '.self_attn.' + role, rows, columns, False)
            dense(base + '.self_attn.q_norm.weight', [8])
            dense(base + '.self_attn.k_norm.weight', [8])
        for expert in range(3):
            family = base + ('.mlp.shared_expert' if expert == 2 else '.mlp.experts')
            prefix = family if expert == 2 else family + f'.{expert}'
            for role in ['gate_proj', 'up_proj', 'down_proj']:
                projection(prefix + '.' + role, 16, 16, True, None if expert == 2 else family)
    dense('model.visual.explicitly_excluded.weight', [1])
    dense('mtp.explicitly_excluded.weight', [1])
    config = dict(model_type='qwen3_5_moe', architectures=['Qwen3_5MoeForConditionalGeneration'],
                  tie_word_embeddings=False, text_config=text,
                  quantization_config=dict(quant_method='modelopt', quant_algo='MIXED_PRECISION',
                                           producer={'name': 'modelopt'}, quantized_layers=declarations))
    return config, tensors


def write(root, config, tensors):
    headers, payloads = {x: {} for x in ['a.safetensors', 'b.safetensors']}, {x: bytearray() for x in ['a.safetensors', 'b.safetensors']}
    patterns = {'BF16': b'\x80\x3f', 'F32': struct.pack('<f', 1), 'U8': b'\x22', 'F8_E4M3': b'\x38'}
    for name, (dtype, shape, shard) in tensors.items():
        count = 1
        for dimension in shape:
            count *= dimension
        start = len(payloads[shard])
        payloads[shard].extend(patterns[dtype] * count)
        headers[shard][name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payloads[shard])])
    for shard in headers:
        path = root / shard
        if path.is_symlink():
            path.unlink()
        header = json.dumps(headers[shard], separators=(',', ':')).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + payloads[shard])
    (root / 'config.json').write_text(json.dumps(config))
    (root / 'model.safetensors.index.json').write_text(json.dumps(dict(
        metadata={'total_size': sum(map(len, payloads.values()))},
        weight_map={name: parts[2] for name, parts in tensors.items()})))


def run(root, error=None, quota=16*1024*1024):
    result = subprocess.run([inspect, str(root), str(quota)], text=True, capture_output=True, timeout=10)
    if error:
        assert result.returncode == 1 and error in result.stderr, (error, result.returncode, result.stdout, result.stderr)
    else:
        assert result.returncode == 0 and 'tensors=117 excluded=2 items=44' in result.stdout, (result.stdout, result.stderr)


with tempfile.TemporaryDirectory(prefix='kadan-manifest-') as directory:
    root = Path(directory)
    config, tensors = fixture()
    write(root, config, tensors)
    run(root)
    result = subprocess.run([unit, str(root)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, (result.stdout, result.stderr)
    cases = [
        ('missing projection', lambda c, t: t.pop('lm_head.weight_scale_2'), 'manifest_missing_tensor'),
        ('unknown text role', lambda c, t: t.update({'model.language_model.surprise': ['BF16', [1], 'a.safetensors']}), 'unexpected_text_tensor'),
        ('wrong shape', lambda c, t: t['lm_head.weight'].__setitem__(1, [31, 8]), 'manifest_tensor_shape_or_dtype'),
        ('wrong dtype', lambda c, t: t['lm_head.weight'].__setitem__(0, 'F8_E4M3'), 'manifest_tensor_shape_or_dtype'),
        ('split companion', lambda c, t: t['lm_head.weight_scale'].__setitem__(2, 'b.safetensors'), 'split_projection_unsupported'),
        ('wrong layer type', lambda c, t: c['text_config']['layer_types'].__setitem__(0, 'full_attention'), 'layer_types'),
        ('gated Q mismatch', lambda c, t: c['text_config'].__setitem__('attn_output_gate', False), 'unsupported_text_flags'),
        ('unsupported model', lambda c, t: c.__setitem__('model_type', 'different'), 'unsupported_model_type'),
        ('dimension cap', lambda c, t: c['text_config'].__setitem__('num_hidden_layers', 257), 'architecture_dimension'),
        ('zero dimension', lambda c, t: c['text_config'].__setitem__('hidden_size', 0), 'architecture_dimension'),
        ('shape alignment', lambda c, t: c['text_config'].__setitem__('hidden_size', 17), 'architecture_alignment'),
        ('bad rope', lambda c, t: c['text_config']['rope_parameters'].__setitem__('mrope_section', [1, 1, 1]), 'rope_sections'),
        ('wrong quant', lambda c, t: c['quantization_config']['quantized_layers']['lm_head'].__setitem__('quant_algo', 'FP8'), 'projection_quantization_declaration'),
        ('wrong block size', lambda c, t: c['quantization_config']['quantized_layers']['lm_head'].__setitem__('group_size', 32), 'projection_group_size'),
        ('unbound declaration', lambda c, t: c['quantization_config']['quantized_layers'].__setitem__('surprise', {'quant_algo': 'FP8'}), 'unused_quantization_declaration'),
    ]
    for name, mutation, error in cases:
        c, t = copy.deepcopy(config), copy.deepcopy(tensors)
        mutation(c, t)
        write(root, c, t)
        run(root, error)
    write(root, config, tensors)
    run(root, 'std::bad_alloc', 64)
    for change, error in [
        (lambda i: i['metadata'].__setitem__('total_size', 0), 'index_total_size'),
        (lambda i: i['weight_map'].__setitem__('lm_head.weight', 'b.safetensors'), 'missing_tensor'),
        (lambda i: i['weight_map'].pop('lm_head.weight'), 'index_coverage'),
        (lambda i: i['weight_map'].__setitem__('lm_head.weight', '../outside'), 'invalid_shard_name'),
    ]:
        write(root, config, tensors)
        path = root / 'model.safetensors.index.json'
        index = json.loads(path.read_text())
        change(index)
        path.write_text(json.dumps(index))
        run(root, error)
    write(root, config, tensors)
    path = root / 'config.json'
    valid = path.read_text()
    for invalid, error in [
        ('{"model_type":"duplicate",' + valid[1:], 'metadata_duplicate_key'),
        (valid + '{}', 'metadata_json_trailing'),
        ('['*34 + '0' + ']'*34, 'metadata_structure_limit'),
        ('{"bad":1e999}', 'metadata_number'),
        ('{"bad":01}', 'metadata_json_syntax'),
        ('{"bad":"' + 'x'*513 + '"}', 'metadata_string_limit'),
        ('{"bad":"\\u0080"}', 'metadata_ascii'),
    ]:
        path.write_text(invalid)
        run(root, error)
    path.write_text(' '*(1024*1024+1)); run(root, 'metadata_file_size')
    write(root, config, tensors)
    (root / 'a.safetensors').unlink()
    (root / 'a.safetensors').symlink_to(root / 'b.safetensors')
    run(root, 'open_failed')
print('manifest fixtures passed: full index/roles, placement, bounded loading, malformed metadata and quotas')
