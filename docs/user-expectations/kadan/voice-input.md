# Voice input

Kadan's chat composer can record one utterance of up to 20 seconds. Recording starts only after the user presses the microphone button and grants browser permission. Microphone access requires HTTPS or localhost and a browser with microphone recording support.

Stopping the recording releases the microphone and transcribes the audio through Hikari's existing Whisper endpoint. Hikari uses its saved transcription model selection. The user must configure the required models in Hikari; a transcription failure leaves the existing draft intact and shows an error.

The transcript is appended to the current draft for review and editing. It is not sent automatically. Pressing Send uses the same agent conversation, persona, history, tools and retry behavior as a typed message. Raw audio is not saved as a chat attachment.

Discarding the recording or leaving the composer releases its microphone and ignores any late transcript. Voice input does not implement continuous listening, spoken replies, barge-in, or a complete voice call.
