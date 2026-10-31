"""Private codec streaming diagnostics; this module never publishes audio."""

import importlib
import math
import time
from array import array

MAX_SAMPLES = 60 * 24000
CHUNK_FRAMES = 12


class PrivateCodecProbe:
    def __init__(self, codec, torch, cache, emit=None, clock=time.perf_counter):
        self.codec, self.torch, self.cache = codec, torch, cache
        self.emit, self.clock = emit, clock
        self.started = clock()
        self.frames = []
        self.template = None
        self.frame_count = 0
        self.samples = array("f")
        self.chunk_count = 0
        self.first_frame = None
        self.first_chunk = None
        self.ended = False
        self.finalized = False

    def put(self, frame):
        if self.ended or self.finalized:
            raise ValueError("Codec probe received a frame after stream end")
        if tuple(frame.shape) != (self.codec.config.n_q,):
            raise ValueError("Codec probe received an invalid audio frame shape")
        if int(frame.min()) < 0 or int(frame.max()) >= self.codec.config.codebook_size:
            raise ValueError("Codec probe refuses EOS or invalid codec codes")
        if self.frame_count >= MAX_SAMPLES // self.codec.config.frame_size:
            raise ValueError("Codec probe exceeded sixty seconds of audio frames")
        if self.first_frame is None:
            self.first_frame = self.clock() - self.started
        self.frame_count += 1
        self.template = frame
        self.frames.append(frame.detach().clone())
        if len(self.frames) == CHUNK_FRAMES:
            self._decode_pending()

    def _capture(self, waveform):
        values = waveform.float().cpu().reshape(-1).tolist()
        if len(self.samples) + len(values) > MAX_SAMPLES or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("Codec probe returned invalid or oversized samples")
        self.samples.extend(values)
        if values:
            self.chunk_count += 1
            if self.first_chunk is None:
                self.first_chunk = self.clock() - self.started
                if self.emit is not None:
                    self.emit(
                        {
                            "state": "progress",
                            "phase": "audio_chunk_decoded",
                            "value": self.chunk_count,
                            "total": 0,
                        }
                    )

    def _decode(self, codes, *, finalize):
        with self.torch.no_grad():
            output = self.codec.decode(
                codes, streaming_cache=self.cache, finalize=finalize, return_dict=True
            )
        self._capture(output.audio_values[0])

    def _decode_pending(self):
        if not self.frames:
            return
        codes = self.torch.stack(self.frames, dim=-1).unsqueeze(0)
        self.frames = []
        self._decode(codes, finalize=False)

    def end(self):
        # generate calls end on success, exception and output-cap exhaustion.
        # Only the caller's separately validated turn EOS permits a final flush.
        self.ended = True

    def finish(self, offline_samples):
        if self.finalized or not self.ended or self.template is None:
            raise ValueError(
                "Codec probe cannot finalize an unfinished or empty stream"
            )
        self._decode_pending()
        self.finalized = True
        self._decode(
            self.template.new_empty((1, self.codec.config.n_q, 0)), finalize=True
        )
        reference = offline_samples.reshape(-1).tolist()
        if len(reference) != len(self.samples):
            raise ValueError(
                f"Codec probe sample count mismatch: {len(self.samples)} versus {len(reference)}"
            )
        if not reference or not all(math.isfinite(value) for value in reference):
            raise ValueError(
                "Codec probe comparison needs finite nonempty offline samples"
            )
        # No unmeasured equality tolerance is asserted. Report the observed
        # difference while the caller returns the original offline waveform.
        maximum = max(
            abs(left - right)
            for left, right in zip(self.samples, reference, strict=True)
        )
        return {
            "first_audio_frame_seconds": self.first_frame,
            "first_decoded_chunk_seconds": self.first_chunk,
            "streamed_chunk_count": self.chunk_count,
            "streamed_sample_count": len(self.samples),
            "streamed_max_abs_difference": maximum,
        }


def create_probe(codec, torch, emit=None):
    # The codec class came from the approved, pinned local checkpoint code.
    # Reuse that exact module's cache type rather than another codec version.
    module = importlib.import_module(type(codec).__module__)
    return PrivateCodecProbe(
        codec, torch, module.LiquidCodecStreamingCache(codec.config), emit
    )
