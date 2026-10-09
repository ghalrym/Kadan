"""Compare raw native JSONL answers to an explicit laya_reference.py capture.

Usage: python laya_parity.py WORKER CHECKPOINT_ROOT REFERENCE_JSON
Requires the local checkpoint, not Python ML packages. Not an automatic download.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

worker, root, capture = sys.argv[1:]
reference = json.loads(Path(capture).read_text())
cases = reference['cases']
run = subprocess.run([worker, root], input=''.join(json.dumps(c['request'])+'\n' for c in cases),
                     capture_output=True, text=True, timeout=180,
                     env=dict(os.environ, OPENBLAS_NUM_THREADS='1'))
assert run.returncode == 0, run.stderr
assert 'resident_bytes=0' in run.stderr, run.stderr
results = [json.loads(line) for line in run.stdout.splitlines()]
assert len(results) == len(cases), (len(results), len(cases))
for case, result in zip(cases, results):
    assert result == case['expected'], (case['name'], result, case['expected'])
print(json.dumps(dict(cases=len(cases), exact_rounded_matches=len(cases), resident_bytes=0,
                     reference_packages=reference['packages'], answers=results), indent=2))
