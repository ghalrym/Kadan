"""Exercise real worker startup/load failures without a model or CUDA context."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

binary = sys.argv[1]
env = dict(os.environ, KADAN_IMAGE_DEVICES='', CUDA_VISIBLE_DEVICES='')
with tempfile.TemporaryDirectory() as directory:
    missing = str(Path(directory) / 'private-checkpoint-missing')
    for args, phase in (([], 'startup'), ([missing, directory], 'load')):
        result = subprocess.run([binary, *args], env=env, capture_output=True, timeout=5)
        assert result.returncode == 1, result
        assert result.stdout == f'error image_{phase}_failed\n'.encode(), result.stdout
        assert f'image_worker {phase}: '.encode() in result.stderr, result.stderr
        assert missing.encode() not in result.stdout
