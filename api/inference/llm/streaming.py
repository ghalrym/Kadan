"""Incremental text decoding; unfinished words/UTF-8 pieces stay in the tokenizer buffer."""
from transformers import TextStreamer


class TextEvents(TextStreamer):
    def __init__(self, tokenizer, emit):
        super().__init__(tokenizer, skip_special_tokens=True)
        self.emit = emit

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.emit({"content": text})
