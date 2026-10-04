"""Fail image builds and clean-install checks if a core inference import is broken."""


def main():
    """Import every catalog architecture without weights, GPU allocation or downloads."""
    import accelerate
    import safetensors
    import torch
    from transformers import AutoTokenizer, GptOssForCausalLM, Qwen3_5MoeForCausalLM
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    from api.inference.model_adapter import build_runtime
    from api.inference.qwen import build_qwen
    from api.inference.glm import build_glm

    assert all((accelerate, safetensors, AutoTokenizer, GptOssForCausalLM,
                Qwen3_5MoeForCausalLM, Glm5NextTextModel, build_runtime, build_qwen, build_glm))
    print(f'Core inference imports OK (torch {torch.__version__}, CUDA build {torch.version.cuda}).')


if __name__ == '__main__':
    main()
