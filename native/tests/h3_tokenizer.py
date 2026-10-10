"""Full vocabulary shape with boundary checks before narrowing numeric IDs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from h3_generation_fixture import tokenizer_fixture
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'tokenizer.json'
    tokenizer_fixture(path)
    original=json.loads(path.read_text())
    def run(text='!'):
        return subprocess.run([sys.argv[1],str(path),text],capture_output=True,text=True)
    result=run();assert result.returncode==0 and json.loads(result.stdout)==[33],result.stderr
    for value in (2**32+33,33.0,-1):
        original['model']['vocab']['!']=value;path.write_text(json.dumps(original))
        result=run();assert result.returncode==1 and 'h3_tokenizer_vocab_id' in result.stderr,result.stderr
    original['model']['vocab']['!']=33;path.write_text(json.dumps(original))
    prompt='🎬'*8000;result=run(prompt);assert result.returncode==0,result.stderr
    ids=json.loads(result.stdout);inverse={value:key for key,value in original['model']['vocab'].items()};byte_for={key:value for key,value in original['model']['vocab'].items() if value<256}
    assert len(ids)>512 and b''.join(bytes(byte_for[c] for c in inverse[token]) for token in ids)==prompt.encode('utf-8')
    result=run('!'*32001);assert result.returncode==1 and 'h3_tokenizer_input_limit' in result.stderr,result.stderr
    original['added_tokens']=[dict(id=2**32+151643,content='<X>',single_word=False,lstrip=False,rstrip=False,normalized=False,special=True)]
    path.write_text(json.dumps(original));result=run();assert result.returncode==1 and 'h3_tokenizer_added_id' in result.stderr,result.stderr
    original['added_tokens'][0]['id']=151643;path.write_text(json.dumps(original));result=run('<X>!');assert result.returncode==0 and json.loads(result.stdout)==[151643,33],result.stderr
    print('PASS byte BPE, literal special tokens, and oversized/fractional/negative ID rejection')
