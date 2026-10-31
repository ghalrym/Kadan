"""Explicit, bounded CPU d1 executor; never selected for ordinary decisions."""

import time

from experimental_worker.decision import _questions, evaluate

MODEL_ID = "d1-2b-cpu-f32"
REVISION = "10e7671d44e4b37700020b1eed3f594c5f2af0b4"
CONTEXT_LIMIT = 1024
REQUIREMENT = (16 * 1024**3, 0)
MAX_EXECUTOR_SECONDS = 180
MODEL = {
    "id": MODEL_ID,
    "repo_id": "LiquidAI/d1-2B",
    "revision": REVISION,
    "license": "lfm1.0",
    "license_url": f"https://huggingface.co/LiquidAI/d1-2B/blob/{REVISION}/LICENSE",
    "license_notice": "CPU F32 d1 trial, including pinned executable checkpoint code. Review the LFM Open License v1.0 before downloading.",
    "display_name": "d1 2B CPU F32 (experimental)",
    "kind": "decision",
    "context_limit": CONTEXT_LIMIT,
}


class BoundedDecisionModel:
    def __init__(self, model):
        self.model = model
        self.engine = model.engine
        self.input_tokens = 0

    def system_one(self, state, questions, *, score_readout):
        # The pinned engine's batching budget does not reject an oversized single
        # sequence. Encode exactly its text-only input before allowing a forward.
        parsed = list(self.engine.named(questions).values())
        sequences = self.engine._sequences([(state, parsed, [])])
        lengths = [len(sequence[2]) for sequence in sequences]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("d1 produced an empty encoded input.")
        self.input_tokens = sum(lengths)
        if self.input_tokens > CONTEXT_LIMIT:
            raise ValueError(
                "The d1 trial input exceeds 1024 encoded tokens; nothing was truncated."
            )
        return self.model.system_one(state, questions, score_readout=score_readout)


def execute(task: dict, operation: str, model_path: str, emit) -> list[dict]:
    if operation != "generate":
        raise ValueError("The d1 trial supports decision evaluation only.")
    if task.get("model_id") != MODEL_ID:
        raise ValueError("The d1 trial requires its explicit model ID.")
    _questions(task)

    # The parent process owns admission, deadlines and kill/join cancellation.
    # Only its disposable child may import inference libraries or checkpoint code.
    import torch
    from transformers import AutoModel

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map="cpu",
        use_kernels=False,
    ).eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("The d1 trial requires CPU parameters only.")
    bounded = BoundedDecisionModel(model)
    emit({"state": "progress", "phase": "loaded", "value": 1, "total": 1})
    with torch.inference_mode():
        answers = evaluate(task, bounded, lambda: None)
    # Results must retain the decision API's list shape, without a timing wrapper.
    print(
        f"D1_CPU_METRIC input_tokens={bounded.input_tokens} seconds={time.perf_counter() - started:.6f}",
        flush=True,
    )
    return answers
