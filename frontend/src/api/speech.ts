import { generateSpeech, listSpeech } from './generated/sdk.gen'
import type { GeneratedSpeech, SpeechRequest } from './generated/types.gen'

export function speechRequest(
  script: string,
  mode: 'describe' | 'clone',
  voice: string,
): SpeechRequest {
  script = script.trim()
  voice = voice.trim()
  if (!script || script.length > 5000)
    throw new Error('Enter a script of 1–5,000 characters.')
  if (!voice || voice.length > 2000)
    throw new Error(
      mode === 'clone'
        ? 'Enter a sample reference of 1–2,000 characters.'
        : 'Describe the voice in 1–2,000 characters.',
    )
  return {
    script,
    voice:
      mode === 'clone' ? { mode, sample: voice } : { mode, description: voice },
  }
}

function speechError(status?: number): Error {
  if (status === 503)
    return new Error(
      'No speech provider is configured. Speech generation and voice cloning are unavailable.',
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

function validAudio(value: unknown): value is GeneratedSpeech {
  return (
    !!value &&
    typeof value === 'object' &&
    ['voice', 'meta', 'script', 'time'].every(
      (key) => typeof (value as Record<string, unknown>)[key] === 'string',
    )
  )
}

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

export async function requestSpeech(
  body: SpeechRequest,
  signal: AbortSignal,
): Promise<GeneratedSpeech> {
  const result = await generateSpeech({ body, signal })
  if (!result.response?.ok) throw speechError(result.response?.status)
  if (!validAudio(result.data?.audio))
    throw new Error('The speech API returned an invalid response.')
  return result.data.audio
}
