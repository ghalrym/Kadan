"""Complete native pipeline smoke with sparse real-dimension model tensors."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from h3_text_fixture import fixture as text_fixture
from h3_denoiser_fixture import fixture as denoiser_fixture
from video_decode_fixture import fixture as vae_fixture


def tokenizer_fixture(path):
    chars=[]
    extra=256
    for b in range(256):
        if 33<=b<=126 or 161<=b<=172 or b>=174: chars.append(chr(b))
        else: chars.append(chr(extra));extra+=1
    vocab={c:i for i,c in enumerate(chars)}
    merges=[]
    for a,b in itertools.product(chars,repeat=2):
        vocab[a+b]=len(vocab);merges.append([a,b])
    pairs=list(vocab)[256:]
    for a,b in itertools.product(pairs,chars):
        if len(vocab)==151643:break
        vocab[a+b]=len(vocab);merges.append([a,b])
    regex=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    byte=dict(type='ByteLevel',add_prefix_space=False,trim_offsets=False,use_regex=False)
    model=dict(type='BPE',dropout=None,unk_token=None,continuing_subword_prefix='',end_of_word_suffix='',byte_fallback=False,fuse_unk=False,vocab=vocab,merges=merges)
    path.write_text(json.dumps(dict(normalizer={'type':'NFC'},pre_tokenizer={'type':'Sequence','pretokenizers':[dict(type='Split',pattern={'Regex':regex},behavior='Isolated',invert=False),byte]},post_processor=byte,model=model,added_tokens=[])))

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        tokenizer_fixture(root/'tokenizer.json')
        text_fixture(root/'text.safetensors')
        denoiser_fixture(root)
        vae_fixture(root/'vae.safetensors')
        subprocess.run([sys.argv[1],str(root)],check=True)
