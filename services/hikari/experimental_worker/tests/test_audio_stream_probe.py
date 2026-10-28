import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experimental_worker.audio import generate
from experimental_worker.audio_stream_probe import PrivateCodecProbe
from experimental_worker.tests import test_audio as audio_tests


class Samples:
    def __init__(self, values):
        self.values = list(values)

    def float(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        return self

    def tolist(self):
        return self.values


class Codes:
    def __init__(self, frames):
        self.frames = frames

    def unsqueeze(self, dimension):
        return self


class Frame:
    shape = (12,)

    def __init__(self, code=1):
        self.code = code

    def min(self):
        return self.code

    def max(self):
        return self.code

    def detach(self):
        return self

    def clone(self):
        return Frame(self.code)

    def new_empty(self, shape):
        return Codes(shape[-1])


class Codec:
    def __init__(self, outputs):
        self.config = SimpleNamespace(n_q=12, codebook_size=2016, frame_size=1920)
        self.outputs = iter(outputs)
        self.calls = []

    def decode(self, codes, **kwargs):
        self.calls.append((codes, kwargs))
        return SimpleNamespace(audio_values=[Samples(next(self.outputs, []))])


TORCH = SimpleNamespace(
    no_grad=nullcontext, stack=lambda frames, dim: Codes(len(frames))
)


class PrivateCodecProbeTests(unittest.TestCase):
    def test_one_cache_spans_chunks_and_one_explicit_empty_final_flush(self):
        codec, cache = Codec([[0.1, 0.2], [0.3, 0.4], [0.5]]), object()
        events = []
        probe = PrivateCodecProbe(codec, TORCH, cache, events.append)
        for _ in range(24):
            probe.put(Frame())
        self.assertEqual([call[0].frames for call in codec.calls], [12, 12])
        self.assertTrue(
            all(call[1]["streaming_cache"] is cache for call in codec.calls)
        )
        probe.end()
        self.assertEqual(len(codec.calls), 2)
        timing = probe.finish(Samples([0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertEqual(
            [call[1]["finalize"] for call in codec.calls], [False, False, True]
        )
        self.assertEqual(codec.calls[-1][0].frames, 0)
        self.assertIs(codec.calls[-1][1]["streaming_cache"], cache)
        self.assertEqual(timing["streamed_chunk_count"], 3)
        self.assertEqual(timing["streamed_sample_count"], 5)
        self.assertLess(timing["streamed_max_abs_difference"], 1e-7)
        self.assertTrue(all(math.isfinite(value) for value in timing.values()))
        self.assertEqual(
            events,
            [
                {
                    "state": "progress",
                    "phase": "audio_chunk_decoded",
                    "value": 1,
                    "total": 0,
                }
            ],
        )
        with self.assertRaises(ValueError):
            probe.finish(Samples([0.1]))
        self.assertEqual(len(codec.calls), 3)

    def test_short_trailing_chunk_is_private_until_caller_finalizes(self):
        codec = Codec([[0.1], [0.2]])
        probe = PrivateCodecProbe(codec, TORCH, object())
        for _ in range(5):
            probe.put(Frame())
        probe.end()
        self.assertFalse(codec.calls)
        probe.finish(Samples([0.1, 0.2]))
        self.assertEqual([call[0].frames for call in codec.calls], [5, 0])
        self.assertEqual([call[1]["finalize"] for call in codec.calls], [False, True])

    def test_end_after_error_or_output_cap_does_not_flush_or_publish(self):
        codec = Codec([[0.1]])
        probe = PrivateCodecProbe(codec, TORCH, object())
        probe.put(Frame())
        probe.end()
        probe.end()
        self.assertFalse(codec.calls)
        self.assertFalse(probe.finalized)
        self.assertFalse(probe.samples)

    def test_eos_and_invalid_frames_never_reach_codec(self):
        codec = Codec([])
        for value in (2016, 2017, -1):
            probe = PrivateCodecProbe(codec, TORCH, object())
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "EOS or invalid"),
            ):
                probe.put(Frame(value))
            self.assertIsNone(probe.template)
        self.assertFalse(codec.calls)

    def test_nonfinite_samples_and_comparison_length_mismatch_fail_probe(self):
        codec = Codec([[float("nan")]])
        probe = PrivateCodecProbe(codec, TORCH, object())
        with self.assertRaisesRegex(ValueError, "invalid or oversized"):
            for _ in range(12):
                probe.put(Frame())
        codec = Codec([[0.1], [0.2]])
        probe = PrivateCodecProbe(codec, TORCH, object())
        probe.put(Frame())
        probe.end()
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            probe.finish(Samples([0.1]))

    def test_frame_buffer_rejects_audio_past_sixty_seconds(self):
        probe = PrivateCodecProbe(Codec([]), TORCH, object())
        for _ in range(750):
            probe.put(Frame())
        self.assertLess(len(probe.frames), 12)
        with self.assertRaisesRegex(ValueError, "sixty seconds"):
            probe.put(Frame())


class ProbeCompletionGateTests(unittest.TestCase):
    def fixture(self, ending):
        return audio_tests.AudioReplyCompletionTests().fixture(ending)

    def task(self):
        return audio_tests.AudioReplyCompletionTests().task()

    def test_output_cap_and_reasoning_rejection_never_finalize_probe(self):
        for ending, text in ((8, "Hello"), (7, "<think>private</think>Hello")):
            model, processor, codec = self.fixture(ending)
            processor.decode.return_value = text
            probe = Mock()
            with (
                self.subTest(ending=ending),
                patch("experimental_worker.audio.create_probe", return_value=probe),
                self.assertRaises(ValueError),
            ):
                generate(
                    self.task(),
                    model,
                    processor,
                    codec,
                    TORCH,
                    codec_streaming_probe=True,
                )
            probe.finish.assert_not_called()
            codec.decode.assert_not_called()

    def test_generate_exception_never_flushes_probe(self):
        model, processor, codec = self.fixture(7)
        probe = Mock()

        def failed(**kwargs):
            kwargs["audio_streamer"].end()
            raise ValueError("decode failed")

        model.generate.side_effect = failed
        with (
            patch("experimental_worker.audio.create_probe", return_value=probe),
            self.assertRaisesRegex(ValueError, "decode failed"),
        ):
            generate(
                self.task(), model, processor, codec, TORCH, codec_streaming_probe=True
            )
        probe.end.assert_called_once()
        probe.finish.assert_not_called()
        codec.decode.assert_not_called()

    def test_flag_off_does_not_create_or_supply_streamer(self):
        model, processor, codec = self.fixture(8)
        with (
            patch("experimental_worker.audio.create_probe") as create,
            self.assertRaises(ValueError),
        ):
            generate(self.task(), model, processor, codec, TORCH)
        create.assert_not_called()
        self.assertNotIn("audio_streamer", model.generate.call_args.kwargs)

    def test_successful_probe_returns_the_original_offline_waveform(self):
        model, processor, codec = self.fixture(7)
        waveform, array = Mock(), Mock()
        samples = SimpleNamespace(size=2)
        waveform.float.return_value.cpu.return_value.numpy.return_value.reshape.return_value = samples
        codec.decode.return_value = SimpleNamespace(audio_values=[waveform])
        array.__mul__ = Mock(return_value=array)
        array.astype.return_value.tobytes.return_value = b"\x00\x00\xff\x7f"
        numpy = Mock()
        numpy.isfinite.return_value.all.return_value = True
        numpy.clip.return_value = array
        probe = Mock()
        probe.finish.return_value = {
            "streamed_max_abs_difference": 0.01,
            "streamed_sample_count": 2,
        }
        with (
            patch.dict("sys.modules", {"numpy": numpy}),
            patch("experimental_worker.audio.create_probe", return_value=probe),
        ):
            baseline = generate(self.task(), model, processor, codec, TORCH)
            measured = generate(
                self.task(), model, processor, codec, TORCH, codec_streaming_probe=True
            )
        self.assertEqual(measured["audio_base64"], baseline["audio_base64"])
        self.assertEqual(measured["text"], baseline["text"])
        self.assertEqual(measured["timing"]["streamed_max_abs_difference"], 0.01)
        probe.finish.assert_called_once_with(samples)
