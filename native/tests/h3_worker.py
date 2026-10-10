"""Native H3 framing, malformed-request recovery and idle cancellation."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

capabilities=json.loads(subprocess.check_output([sys.argv[1],"--capabilities"],text=True))
assert capabilities==dict(protocol=1,cuda=False,audio=True,host_budget=128*1024**3,device_budget=2*1024**3)

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    command=[sys.argv[1],*(str(root/'missing') for _ in range(5))]
    base={'prompt':'red ball','output':str(root/'result.y4m')}
    requests=[json.dumps(base|{'width':33}),'{"prompt":"a","prompt":"b"}',json.dumps(base|{'seed':1.5}),json.dumps(base|{'width':-1}),json.dumps(base|{'unknown':1}),json.dumps(base),json.dumps(base|{'height':31})]
    result=subprocess.run(command,input='\n'.join(requests)+'\n',text=True,capture_output=True,check=True)
    errors=[json.loads(line)['error'] for line in result.stdout.splitlines()]
    assert errors==['h3_generation_dimensions','h3_request_duplicate_key','h3_request_integer','h3_request_integer','h3_request_field','h3_tokenizer_file_size','h3_generation_dimensions'],errors
    assert result.stderr.strip()=='resident_bytes=0' and not list(root.iterdir()),result.stderr
    profile=base|{'short_edge':480,'aspect':'16:9','duration':4}
    cases=[(profile,'h3_generation_dimensions'),(profile|{'width':480},'h3_request_profile_conflict'),(profile|{'duration':3},'h3_api_profile'),(profile|{'aspect':'4:3'},'h3_api_aspect')]
    for request,expected in cases:
        outcome=subprocess.run(command,input=json.dumps(request)+'\n',text=True,capture_output=True,check=True)
        assert json.loads(outcome.stdout)['error']==expected,outcome.stdout
        assert outcome.stderr.strip()=='resident_bytes=0' and not list(root.iterdir())
    result=subprocess.run(command,input='{}',text=True,capture_output=True)
    assert result.returncode==1 and 'h3_incomplete_frame' in result.stderr and 'resident_bytes=0' in result.stderr
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    child.stdin.write('{}\n');child.stdin.flush()
    assert json.loads(child.stdout.readline())['error']=='h3_request_string'
    child.send_signal(signal.SIGTERM)
    stdout,stderr=child.communicate(timeout=10)
    assert child.returncode==0 and not stdout and 'resident_bytes=0' in stderr
    print('PASS malformed recovery, strict fields, bounded frames and cancellation')
