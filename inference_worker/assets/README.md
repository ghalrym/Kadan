# Whisper assets

Tokenizer ranks and mel filters come from the MIT-licensed `openai-whisper` 20250625 source distribution (SHA-256 `37a91a3921809d9f44748ffc73c0a55c9f366c85a3ef5c2ae0cc09540432eb96`). `whisper_languages.json` preserves its tokenizer language order. `WHISPER_LICENSE` contains the upstream license.

The C++ worker verifies `manifest.json`, builds its English token table, and extracts the mel arrays. No Python package is used at runtime.
