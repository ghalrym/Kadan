"""Import actual native Wan and each worker without weights or model construction.

CPU CI substitutes only the upstream import-time current-device query. It does
not validate CUDA, FlashAttention kernels, or inference. Package imports are real.
"""
from importlib.metadata import version
from pathlib import Path
import runpy
import sys

import torch

workers = Path(__file__).resolve().parents[1] / 'api/inference/workers'
# Match direct-script launch resolution: a sibling wan.py must fail this check.
sys.path.insert(0, str(workers))
if not torch.cuda.is_available():
    torch.cuda.current_device = lambda: 0
for worker in sorted(workers.glob('wan*.py')):
    runpy.run_path(str(worker), run_name='kadan_import_smoke')
    print(f'Imported worker: {worker.name}')
wan = sys.modules['wan']
assert Path(wan.__file__).parent != workers, 'Worker shadows native wan package'
for name in ('WanT2V', 'WanTI2V', 'WanI2V', 'WanS2V', 'WanAnimate'):
    assert callable(getattr(wan, name)), name
print(f'Native package: {wan.__file__}')
print(f'Transformers {version("transformers")}; PEFT {version("peft")}; Diffusers {version("diffusers")}')
