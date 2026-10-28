# Transcript formatting

Transcript formatting reuses the fixed LFM conversation worker after native Whisper returns raw text. `api/services/transcription.py` owns the sequence and submits a separate completion through the same worker hub and physical memory admission. There is no S1-mini worker or Qwen-derived formatter in the active stack.

The formatter receives the raw transcript serialized in a JSON `transcript` field, with separate instructions to preserve its meaning and language, correct punctuation and obvious errors, and return exactly one JSON object containing a single string field, `text`. A short worked example demonstrates that contract. It has no tools, conversation identity or prefix reuse. Input is limited to 32,000 characters; language-model context and memory checks still apply.

A valid result must name the fixed LFM checkpoint, finish normally, contain a valid single-field JSON object with nonempty string text. Duplicate fields, additional keys, Markdown fences and conversational text outside the object are rejected. Failure, incomplete output or rejected input returns the raw transcript with a visible formatting warning. Cancellation remains cancellation and does not turn into successful raw fallback. This validates completion and structure, not semantic equivalence; formatting quality still needs review against the retained raw text.

Because formatting shares the conversation model and worker queue, it also shares that worker's admission priority. It does not acquire a separate CPU-only formatter allocation.
