"""Buffered BF16 LFM3 audio turns, executed only inside the experimental worker."""

import base64
import binascii
import io
import re
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path

from experimental_worker.audio_stream_probe import create_probe
from experimental_worker.call_history import append_call_history
from experimental_worker.speech_segments import speech_scripts

MODEL = {
    "id": "lfm3-audio-2b-bf16",
    "repo_id": "LiquidAI/LFM3-Audio-2B",
    "revision": "9d49701c372d6011d900f6abb40c6b296fb706dd",
    "license": "lfm1.0",
    "license_url": "https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/9d49701c372d6011d900f6abb40c6b296fb706dd/LICENSE",
    "display_name": "LFM3 Audio 2B BF16",
    "short_name": "LFM3 Audio",
    "estimated_bytes": 4_170_027_447,
    "kind": "audio",
    "context_limit": 32768,
    "architecture_context_limit": 32768,
}
# Provisional admission bounds, not a measured fit guarantee. Include model,
# codec, preprocessing and bounded generation; the engine uses one visible GPU.
REQUIREMENT = (8 * 1024**3, 16 * 1024**3)
MAX_CONTEXT_TOKENS = 32768
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - 1
MAX_AUDIO_SAMPLES = 60 * 24000
CODEC_FRAME_SAMPLES = 1920


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
    if task.get("mode") == "tts":
        script = task.get("script")
        if not isinstance(script, str) or not script.strip() or len(script) > 240:
            raise ValueError("TTS needs one bounded known script")
        return [
            {
                "role": "system",
                "content": "Perform TTS in American English using a female voice.",
            },
            {"role": "user", "content": script},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
    if task.get("mode") not in ("reply", "answer") or task.get("audio") is not None:
        raise ValueError("Unknown audio call operation")
    messages = task.get("messages", [])
    if not messages:
        raise ValueError("Audio call needs text history")
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
    if task["mode"] == "reply":
        result[0]["content"] = (
            "Respond with interleaved text and audio.\n\n" + result[0]["content"]
        )
    if task["mode"] == "reply":
        result.append({"role": "assistant", "content": "<|mixed_start|>"})
    return result


def public_text(raw: str, *, transcribe: bool) -> str:
    """Keep control markers visible until reasoning and tool output are checked."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    if "<think>" in text or "</think>" in text:
        raise ValueError("Audio model returned unfinished reasoning")
    if text != raw and not transcribe:
        # There is no verified alignment for removing thought speech from a
        # buffered waveform. Reject that reply before publishing text or audio.
        raise ValueError("Audio model returned reasoning in a spoken reply")
    for marker in (
        "<|audio_out|>",
        "<|audio_start|>",
        "<|text_end|>",
        "<|mixed_start|>",
        "<|mixed_end|>",
        "<|im_end|>",
        "<|endoftext|>",
    ):
        text = text.replace(marker, "")
    if any(
        marker in text
        for marker in (
            "<|",
            "<tool_call",
            "</tool_call",
            "<function",
            "<parameter",
            "<tool_response",
            "</tool_response",
        )
    ):
        raise ValueError("Audio model returned tool or unknown control output")
    return text.strip()


def validate_context(input_tokens: int, output_tokens: int | None = None) -> int:
    """Use all remaining released context unless a smaller output is requested."""
    if output_tokens is None:
        output_tokens = MAX_CONTEXT_TOKENS - input_tokens
    if (
        type(input_tokens) is not int
        or type(output_tokens) is not int
        or input_tokens < 1
        or output_tokens < 1
        or input_tokens + output_tokens > MAX_CONTEXT_TOKENS
    ):
        raise ValueError(
            "Audio call exceeds the 32768-token model context; start a new call"
        )
    return output_tokens


class AudioDurationGuard:
    """Stop non-EOS speech frames before decoding could exceed call memory bounds."""

    def __init__(self, max_samples: int):
        self.maximum_frames = max_samples // CODEC_FRAME_SAMPLES
        self.frames = 0
        if self.maximum_frames < 1:
            raise ValueError(
                "No speech duration remains within this call turn's sixty-second limit"
            )

    def put(self, frame):
        if self.frames >= self.maximum_frames:
            raise ValueError(
                "Audio generation exceeded the call turn's sixty-second speech limit"
            )
        self.frames += 1

    def end(self):
        pass


def generate(
    task: dict,
    model,
    processor,
    codec,
    torch,
    audio_path=None,
    *,
    codec_streaming_probe=False,
    max_audio_samples=MAX_AUDIO_SAMPLES,
    emit=None,
) -> dict:
    """Use the pinned model-card contract; dependencies are passed for unit tests."""
    transcribe = task["mode"] == "transcribe"
    answer = task["mode"] == "answer"
    tts = task["mode"] == "tts"
    if codec_streaming_probe and task["mode"] != "reply":
        raise ValueError("The private codec probe cannot run during transcription")
    inputs = processor.apply_chat_template(
        conversation(task, audio_path),
        **(
            {"add_generation_prompt": True}
            if transcribe or answer
            else {"continue_final_message": True}
        ),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")
    count = inputs["input_ids"].shape[1]
    budget = validate_context(count)
    sampling = (
        {"suppress_mixed": True, "suppress_audio_out": True}
        if transcribe or answer
        else {
            "audio_temperature": 1.0,
            "audio_top_k": 128,
            "audio_repetition_penalty": 1.1,
            "audio_repetition_window": 4096,
            **({"suppress_mixed": True, "suppress_audio_out": True} if tts else {}),
        }
    )
    probe = create_probe(codec, torch, emit) if codec_streaming_probe else None
    audio_streamer = (
        probe
        if probe is not None
        else (
            AudioDurationGuard(max_audio_samples)
            if not (transcribe or answer)
            else None
        )
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            return_dict_in_generate=True,
            **(
                {"audio_streamer": audio_streamer} if audio_streamer is not None else {}
            ),
            **({"generation_mode": "text"} if answer else {}),
            **sampling,
        )
        tokens = output.sequences[0, count:]
        eos = model.config.text_config.eos_token_id
        endings = set(eos if isinstance(eos, (list, tuple)) else [eos])
        # In mixed output, text_end starts the audio tail; only an actual
        # turn EOS proves that the buffered speech finished before the cap.
        if transcribe or answer or tts:
            endings.add(model.config.text_end_token_id)
        if not len(tokens) or int(tokens[-1]) not in endings:
            raise ValueError(
                "Audio generation reached the trial limit before completing the reply"
            )
        decoded = processor.decode(tokens, skip_special_tokens=False)
        if tts:
            # Pure TTS has only audio placeholders and its terminal markers.
            # A new mode opener could make text_end mean an unfinished mixed
            # tail, so reject it before the ordinary mixed-output parser.
            remaining = decoded
            for marker in (
                "<|audio_out|>",
                "<|text_end|>",
                "<|im_end|>",
                "<|endoftext|>",
            ):
                remaining = remaining.replace(marker, "")
            if remaining.strip():
                raise ValueError(
                    "TTS returned unexpected public text or control markers"
                )
        text = public_text(decoded, transcribe=transcribe or answer)
        if transcribe or answer:
            if answer and not text:
                raise ValueError("Audio model returned no public answer")
            return {"text": text}
        if tts and text:
            raise ValueError("TTS returned unexpected public text")
        if (
            (not tts and not text)
            or output.audio_codes is None
            or int(output.audio_lengths[0]) <= 0
        ):
            raise ValueError("Audio model did not return the required speech output")
        if int(output.audio_lengths[0]) * CODEC_FRAME_SAMPLES > max_audio_samples:
            raise ValueError(
                "Audio output exceeds the call turn's sixty-second speech limit"
            )
        waveform = codec.decode(
            output.audio_codes[:1, :, : int(output.audio_lengths[0])]
        ).audio_values[0]
        samples = waveform.float().cpu().numpy().reshape(-1)
    # This import stays in the per-job process: the API and transport do not
    # import optional inference libraries or initialize CUDA.
    import numpy as np

    if not np.isfinite(samples).all() or not 1 <= samples.size <= max_audio_samples:
        raise ValueError("Audio model returned invalid or oversized speech")
    probe_timing = probe.finish(samples) if probe is not None else None
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(24000)
        sink.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return {
        "text": text,
        "audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        **(
            {"script": task["script"], "sample_count": int(samples.size)} if tts else {}
        ),
        **({"timing": probe_timing} if probe_timing is not None else {}),
    }


def sentence_reply(messages, model, processor, codec, torch, emit):
    started = time.perf_counter()
    answer = generate(
        {"mode": "answer", "messages": messages}, model, processor, codec, torch
    )["text"]
    torch.cuda.synchronize()
    answered = time.perf_counter()
    scripts = speech_scripts(answer)
    emit({"state": "audio_answer", "answer": answer})
    total_samples, total_tts, first_segment = 0, 0.0, None
    for index, script in enumerate(scripts):
        segment_started = time.perf_counter()
        result = generate(
            {"mode": "tts", "script": script},
            model,
            processor,
            codec,
            torch,
            max_audio_samples=MAX_AUDIO_SAMPLES - total_samples,
        )
        torch.cuda.synchronize()
        segment_finished = time.perf_counter()
        total_tts += segment_finished - segment_started
        if (
            result["script"] != script
            or result["sample_count"] < 1
            or not result.get("audio_base64")
        ):
            raise ValueError("TTS returned an invalid known-script segment")
        total_samples += result["sample_count"]
        if total_samples > 60 * 24000:
            raise ValueError("Sentence TTS exceeded sixty seconds for this turn")
        if first_segment is None:
            first_segment = segment_finished - started
        emit(
            {
                "state": "audio_segment",
                "audio_segment": {
                    "index": index,
                    "script": script,
                    "audio_base64": result["audio_base64"],
                },
            }
        )
    return {
        "text": answer,
        "scripts": scripts,
        "segment_count": len(scripts),
        "timing": {
            "public_text_answer_seconds": answered - started,
            "first_tts_segment_seconds": first_segment,
            "total_tts_seconds": total_tts,
            "tts_segment_count": len(scripts),
            "tts_sample_count": total_samples,
        },
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
    mode = task.get("mode")
    streaming_probe = task.get("codec_streaming_probe", False)
    sentence_tts = task.get("sentence_tts", False)
    if not isinstance(sentence_tts, bool) or (
        sentence_tts and (mode != "turn" or streaming_probe)
    ):
        raise ValueError(
            "Sentence TTS requires a complete turn without the private codec probe"
        )
    if not isinstance(streaming_probe, bool) or (streaming_probe and mode != "turn"):
        raise ValueError("The private codec probe requires a complete audio turn")
    data = audio_bytes(task.get("audio")) if mode in ("transcribe", "turn") else None
    if mode == "turn":
        conversation({"mode": "reply", "messages": task.get("messages")})
    elif data is None:
        conversation(task)
    # Optional inference dependencies must be imported only in the spawned job
    # process. Native Hikari and its API never load the experimental runtime.
    started = time.perf_counter()
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    imported = time.perf_counter()
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
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )
    codec = processor.audio_tokenizer.to(device="cuda:0").eval()
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    stage_timing = {}
    if mode in ("transcribe", "turn"):
        asr_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="lfm3-audio-") as directory:
            audio_path = Path(directory) / "utterance.wav"
            audio_path.write_bytes(data)
            result = generate(
                {"mode": "transcribe"}, model, processor, codec, torch, str(audio_path)
            )
        torch.cuda.synchronize()
        asr_finished = time.perf_counter()
        if mode == "turn":
            stage_timing["asr_operation_seconds"] = asr_finished - asr_started
            stage_timing["reply_operation_seconds"] = 0.0
            transcript = result["text"]
            # This typed event reaches the call before reply generation, so a
            # later failure or interruption does not erase the user's utterance.
            emit({"state": "audio_transcript", "transcript": transcript})
            if transcript:
                history = append_call_history(
                    task["messages"][1:], "user", transcript, model_bounded=True
                )
                reply_started = time.perf_counter()
                if sentence_tts:
                    result = sentence_reply(
                        [task["messages"][0], *history],
                        model,
                        processor,
                        codec,
                        torch,
                        emit,
                    )
                else:
                    result = generate(
                        {"mode": "reply", "messages": [task["messages"][0], *history]},
                        model,
                        processor,
                        codec,
                        torch,
                        **(
                            {"codec_streaming_probe": True, "emit": emit}
                            if streaming_probe
                            else {}
                        ),
                    )
                torch.cuda.synchronize()
                stage_timing["reply_operation_seconds"] = (
                    time.perf_counter() - reply_started
                )
            else:
                result = {"text": "", "audio_base64": None}
            result["transcript"] = transcript
    else:
        result = generate(task, model, processor, codec, torch)
    torch.cuda.synchronize()
    result["timing"] = {
        **result.get("timing", {}),
        "audio_import_seconds": imported - started,
        "model_load_seconds": loaded - imported,
        "processor_load_seconds": prepared - loaded,
        "audio_operation_seconds": time.perf_counter() - prepared,
        **stage_timing,
    }
    return result
