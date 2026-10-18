# Transcript formatting worker (`formatting`)

Cleans up Whisper transcripts with S1-mini by Superwhisper into semi-formal
prose. It is optional per request and saved as a setting.

## Serves

- Formatting of transcripts after the transcription worker, when enabled.

## Engine

- S1-mini on the CPU, with its weights resident as floats. English input up to
  1,000 tokens and output up to 1,024 tokens.

## Memory

- **Minimum and preferred:** twice the checkpoint size plus 1 GiB of RAM. It
  uses no VRAM.

## Scheduling

- Priority 3, shared with transcription. Its budget never includes a GPU.

## Constraints

- A formatter failure returns the raw transcript with an error. It never fails
  the transcription.
