# Wan worker import verification

The optional Wan environment pins Transformers 4.51.3, PEFT 0.17.1 and Diffusers
0.35.1 together. PEFT 0.17.1 imports HybridCache, which is absent from Transformers
5. The native worker is named `wan_worker.py` so direct Python execution cannot
shadow the installed `wan` package with a sibling `wan.py` file.

CI installs the actual pinned Wan source and its declared dependencies in a fresh
Python 3.12 environment with CPU Torch/torchvision wheels. Only CUDA-only
FlashAttention is excluded from this import smoke; the production requirements
still retain the upstream CUDA dependency. `scripts/check_wan_imports.py` puts the
worker directory first on the import path, matching direct script execution, and
imports every Wan worker plus all five native provider classes without loading
weights or constructing models. Upstream evaluates `torch.cuda.current_device()`
in function defaults at import time, so CPU-only smoke substitutes that query
with device index 0. Package imports are not mocked. This does not validate CUDA
kernels, preprocessing dependencies, memory requirements or actual generation.

Sources: [PEFT's cache imports](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/peft_model.py),
[pinned Wan source](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff).
