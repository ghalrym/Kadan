import { generateSpeech, listSpeech } from './generated/sdk.gen'
import type { GeneratedSpeech, SpeechRequest } from './generated/types.gen'

/**
 * Translate an HTTP status or missing response into a user-facing speech error.
 */
function speechError(status?: number): Error {
  if (status === 503)
    return new Error(
      'Speech generation is unavailable. Check the selected checkpoint and server model setup.',
    )
  if (status === 422)
    return new Error(
      'The API rejected these speech settings. Check the script and voice details.',
    )
  return new Error(
    status
      ? `Speech API request failed (HTTP ${status}). Please retry.`
      : 'Cannot reach the speech API. Check that the backend is running, then retry.',
  )
}

/**
 * Recognize the four-string speech metadata shape without implying playable audio.
 */
function validAudio(value: unknown): value is GeneratedSpeech {
  return (
    !!value &&
    typeof value === 'object' &&
    ['voice', 'meta', 'script', 'time'].every(
      (key) => typeof (value as Record<string, unknown>)[key] === 'string',
    )
  )
}

/**
 * Fetch cancellable speech history and reject transport errors or malformed metadata.
 * The current backend returns an empty list rather than seeded audio.
 */
export async function fetchSpeechHistory(
  signal: AbortSignal,
): Promise<GeneratedSpeech[]> {
  const result = await listSpeech({ signal })
  if (!result.response?.ok) throw speechError(result.response?.status)
  if (
    !Array.isArray(result.data?.audio) ||
    !result.data.audio.every(validAudio)
  )
    throw new Error('The speech API returned invalid history.')
  return result.data.audio
}

/**
 * Send a typed speech request with cancellation and validate complete WAV metadata.
 */
export async function requestSpeech(
  body: SpeechRequest,
  signal: AbortSignal,
): Promise<GeneratedSpeech> {
  const result = await generateSpeech({ body, signal })
  if (!result.response?.ok) throw speechError(result.response?.status)
  if (!validAudio(result.data?.audio) || !result.data.audio.audio_base64 || result.data.audio.mime_type !== 'audio/wav')
    throw new Error('The speech API returned an invalid response.')
  return result.data.audio
}
