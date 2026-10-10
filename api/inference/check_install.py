"""Check worker orchestration dependencies without executing models."""
from transformers import AutoTokenizer
from api.inference.llm.qwen_subprocess import QwenSubprocessAdapter
from api.inference.llm.qwen_residency import build_resident_qwen
from api.inference.decisions.laya_subprocess import resolve_laya_command
from api.inference.image.native_worker import NativeImageRuntime
from api.inference.video.h3_worker import H3Provider
from api.inference.tts.check_install import check_speech_install


def main():
    assert all((AutoTokenizer, QwenSubprocessAdapter, build_resident_qwen,
                resolve_laya_command, NativeImageRuntime, H3Provider))
    check_speech_install()
    print('All six C++ worker adapters import successfully; no model execution.')


if __name__ == '__main__':
    main()
