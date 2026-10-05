import { transcribeAudio } from './generated/sdk.gen'
import type { TranscriptionRequest } from './generated/types.gen'

/**
 * Validate and submit an opaque audio reference plus the formatting preference.
 * Forward the caller signal; reject unavailable providers, HTTP/network failures
 * and blank/malformed transcripts. No recording, upload or reference fetch occurs here.
 */
export async function requestTranscription(audio: string, formatting: boolean, signal: AbortSignal): Promise<string> {
  const reference = audio.trim()
  if (!reference) {
    throw new Error('Enter an audio reference.')
  }
  const body: TranscriptionRequest = { audio: reference, formatting }
  const result = await transcribeAudio({ body, signal })
  if (!result.response?.ok) {
    if (result.response?.status === 503) {
      throw new Error('Transcription is unavailable: no speech-to-text provider is configured. Recording and file upload are not supported yet.')
    }
    if (result.response?.status === 422) throw new Error('The API rejected this audio reference. Check the input and retry.')
    throw new Error(result.response ? `Transcription failed (HTTP ${result.response.status}). Please retry.` : 'Cannot reach the transcription API. Check that the backend is running.')
  }
  if (typeof result.data?.text !== 'string' || !result.data.text.trim()) throw new Error('The API returned an invalid transcript.')
  return result.data.text
}
