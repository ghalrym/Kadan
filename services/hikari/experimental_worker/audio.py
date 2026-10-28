"""Buffered BF16 LFM3 audio turns, executed only inside the experimental worker."""

import base64
import binascii
import io
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

MODEL = {
    "id": "lfm3-audio-2b-bf16",
    "repo_id": "LiquidAI/LFM3-Audio-2B",
    "revision": "9d49701c372d6011d900f6abb40c6b296fb706dd",
    "license": "lfm1.0",
    "license_url": "https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/9d49701c372d6011d900f6abb40c6b296fb706dd/LICENSE",
    "display_name": "LFM3 Audio 2B BF16 (experimental)",
    "short_name": "LFM3 Audio",
    "estimated_bytes": 4_170_027_447,
    "kind": "audio",
    "context_limit": 8192,
}
# Provisional admission bounds, not a measured fit guarantee. Include model,
# codec, preprocessing and bounded generation; the engine uses one visible GPU.
REQUIREMENT = (6 * 1024**3, 10 * 1024**3)
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 1024


def audio_bytes(uri: str) -> bytes:
    """Accept the browser's bounded mono 16-kHz PCM utterance, never a URL."""
    prefix = "data:audio/wav;base64,"
    if not isinstance(uri, str) or len(uri) > 900_000 or not uri.startswith(prefix):
        raise ValueError("Audio trial requires a bounded WAV data URI")
    try:
        data = base64.b64decode(uri[len(prefix) :], validate=True)
        with wave.open(io.BytesIO(data), "rb") as source:
            if (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            ) != (1, 2, 16000, "NONE"):
                raise ValueError("Audio trial requires mono 16-kHz PCM16 WAV")
            frames = source.getnframes()
            if not 1 <= frames <= 20 * 16000:
                raise ValueError("Audio utterance must be at most twenty seconds")
            if len(source.readframes(frames)) != frames * 2:
                raise ValueError("Audio utterance is truncated")
    except (binascii.Error, wave.Error, EOFError) as error:
        raise ValueError("Audio utterance is not a valid PCM WAV") from error
    return data


def conversation(task: dict, audio_path: str | None = None) -> list[dict]:
    if task.get("mode") == "transcribe":
        if audio_path is None or task.get("messages"):
            raise ValueError("Transcription needs one audio utterance and no history")
        return [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
        ]
    if task.get("mode") != "reply" or task.get("audio") is not None:
        raise ValueError("Unknown audio call operation")
    messages = task.get("messages", [])
    if (
        not messages
        or len(messages) > 14
        or sum(len(message.get("text", "")) for message in messages) > 20_000
    ):
        raise ValueError("Audio call history exceeds the trial limit")
    result = []
    for message in messages:
        role, text = message.get("role"), message.get("text")
        if (
            role not in ("system", "user", "assistant")
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError(
                "Audio call history must contain text messages without tools"
            )
        result.append({"role": role, "content": text})
    if result[0]["role"] != "system":
        raise ValueError("Audio call needs the session's spoken instructions")
    result[0]["content"] = (
        "Respond with interleaved text and audio.\n\n" + result[0]["content"]
    )
    result.append({"role": "assistant", "content": "<|mixed_start|>"})
    return result


def generate(task: dict, model, processor, codec, torch, audio_path=None) -> dict:
    """Use the pinned model-card contract; dependencies are passed for unit tests."""
    transcribe = task["mode"] == "transcribe"
    inputs = processor.apply_chat_template(
        conversation(task, audio_path),
        **(
            {"add_generation_prompt": True}
            if transcribe
            else {"continue_final_message": True}
        ),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")
    count = inputs["input_ids"].shape[1]
    if count > MAX_INPUT_TOKENS:
        raise ValueError(
            "Audio call exceeds the 8192-token trial context; start a new call"
        )
    sampling = (
        {"suppress_mixed": True, "suppress_audio_out": True}
        if transcribe
        else {
            "audio_temperature": 1.0,
            "audio_top_k": 128,
            "audio_repetition_penalty": 1.1,
            "audio_repetition_window": 4096,
        }
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            return_dict_in_generate=True,
            **sampling,
        )
        tokens = output.sequences[0, count:]
        eos = model.config.text_config.eos_token_id
        endings = set(eos if isinstance(eos, (list, tuple)) else [eos])
        endings.add(model.config.text_end_token_id)
        if not len(tokens) or int(tokens[-1]) not in endings:
            raise ValueError(
                "Audio generation reached the trial limit before completing the reply"
            )
        text = processor.decode(tokens, skip_special_tokens=True).strip()
        if len(text) > (4000 if transcribe else 8000):
            raise ValueError("Audio model output exceeded the call limit")
        if transcribe:
            return {"text": text}
        if not text or output.audio_codes is None or int(output.audio_lengths[0]) <= 0:
            raise ValueError("Audio model did not return both speech and text")
        waveform = codec.decode(
            output.audio_codes[:1, :, : int(output.audio_lengths[0])]
        ).audio_values[0]
        samples = waveform.float().cpu().numpy().reshape(-1)
    # This import stays in the per-job process: the API and transport do not
    # import optional inference libraries or initialize CUDA.
    import numpy as np

    if not np.isfinite(samples).all() or not 1 <= samples.size <= 60 * 24000:
        raise ValueError("Audio model returned invalid or oversized speech")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(24000)
        sink.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return {
        "text": text,
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def execute(
    task: dict, operation: str, model_path: str, emit: Callable[[dict], None]
) -> dict:
    if (
        operation != "generate"
        or task.get("feature") != "audio_call"
        or task.get("model_id", MODEL["id"]) != MODEL["id"]
    ):
        raise ValueError(
            "The audio worker only serves explicitly selected audio-call turns"
        )
    data = audio_bytes(task.get("audio")) if task.get("mode") == "transcribe" else None
    if data is None:
        conversation(task)
    # Optional inference dependencies must be imported only in the spawned job
    # process. Native Hikari and its API never load the experimental runtime.
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("The BF16 audio trial needs a CUDA GPU supporting BF16")
    path = str(Path(model_path).resolve())
    model = AutoModelForMultimodalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )
    codec = processor.audio_tokenizer.to(device="cuda:0").eval()
    if task.get("mode") == "transcribe":
        with tempfile.TemporaryDirectory(prefix="lfm3-audio-") as directory:
            audio_path = Path(directory) / "utterance.wav"
            audio_path.write_bytes(data)
            return generate(task, model, processor, codec, torch, str(audio_path))
    return generate(task, model, processor, codec, torch)
