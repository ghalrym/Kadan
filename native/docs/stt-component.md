# Bounded CPU Whisper log-mel frontend

`kadan_stt` executes an original scalar C++ DFT and mel frontend. It accepts one
finite F32 mono 16 kHz PCM span, 201–16000 samples, and an explicit nonnegative
F32 filterbank [80,201] or [128,201]. It does not decode audio containers, resample,
load Whisper model weights, run an encoder/decoder, or produce transcription.
Full STT remains unavailable; this component is not registered with a queue/API.

Numerics: periodic Hann, centered reflect padding by 200, 400-point DFT, hop 160,
omit the final STFT column, magnitude squared, filterbank projection, log10 floor
1e-10, global max-minus-eight clipping and (x+4)/4. Output is band-major
[bins, floor(samples/160)], at most 12800 values. DFT/window intermediates use
FP64; filter coefficients, PCM and output use F32. This is a bounded correctness
foundation, not an optimized FFT or real-time throughput claim.

`LogMel` requires a serialized owner; only its atomic cancellation flag may be
changed concurrently. It checks cancellation during load, each frame/frequency,
mel band and normalization. Discard partially written output after any failure.
The optional observation callback is for instrumentation and cannot reenter.
Loaded filters/tables stay pinned throughout execution. Unload while pinned
fails; successful unload frees arrays before releasing their reservation.

Memory accounting uses the existing Resources ledger, Workload::speech and no
GPU reservation. Resident tables occupy 1,289,600 bytes, plus 64,320 or 102,912
filter bytes. Execution charges 4,808 bytes of fixed scratch. The caller must
admit input/output/filter spans it owns. The CLI admits all three heap buffers
through output publication, explicitly frees them, then releases admission and
reports resident_bytes=0. At 128 bins and 16000 samples the caller charge is
218,112 bytes. Allocator/control-plane overhead is outside payload accounting,
consistent with Resources. No storage tier or model backing is needed for this
small resident frontend.

CLI (little-endian F32 raw files; no model download):

```
kadan-stt-component 80 canonical-mel-80.f32 audio-16khz-mono.f32
```

JSON contains mel, bins, frames, residency counts, full_transcription=false and
gpu_execution=false. Bounded inputs, malformed layouts, nonfinite PCM and
negative/nonfinite filter coefficients fail. SIGINT/SIGTERM request cancellation
at the next frame and prevent publication when already observed.

## Verification

Automatic CTest: `stt-component` covers analytic silence/DC values, retained
execution, both filterbank sizes, cancellation after a completed frame, exceptions,
pinned eviction, failed admission/load, accounting, unload and destructor cleanup.
`stt-component-cli` covers length limits, output layout, publication accounting
and malformed/nonfinite inputs without external Python dependencies.

Explicit independent reference (CPU only; never included in automatic GPU work):

```
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python native/tests/stt_whisper.py /absolute/path/kadan-stt-component
```

Requires openai-whisper==20250625 and canonical mel_filters.npz SHA256
7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c.
The script verifies raw coefficient hashes for both banks, uses the pinned
whisper.audio.log_mel_spectrogram directly with Torch CPU threads=1, and compares
all values at atol=rtol=2e-4. It records source, binary, inputs and output hashes.
40 cases: silence, boundary impulses, 440 Hz sine and seeded PCM at lengths
201, 319, 320, 15999 and 16000 for both banks. Actual installed-reference run:
all passed, maximum absolute error 2.4199485778808594e-05. No model checkpoint,
transcription, CUDA execution or live API source change was involved.

Subsequent native work must separately implement/validate encoder and decoder,
then an executor/queue adapter. API and production orchestration changes belong
in separate PRs; this component alone must not advertise speech recognition.
