import type { ConversationCapabilities, SpeechBackendOption } from './generated/types.gen'

export type VoiceBackend = SpeechBackendOption['id']
export type VoiceMessage = { role: 'user' | 'assistant'; content: string }
export type VoiceCapabilities = ConversationCapabilities
export type VoiceTiming = {
  llm_first_token_ms: number | null
  speech_first_audio_ms: number
  turn_first_audio_ms: number
  audio_duration_seconds: number
  speech_rtf: number
  turn_ms: number
  native: Record<string, unknown> | null
}
export type VoiceEvent =
  | { type: 'text'; text: string }
  | { type: 'audio'; audio: string; sample_rate: number }
  | { type: 'done'; timing?: VoiceTiming }
  | { type: 'error'; detail: string }

async function checkResponse(response: Response) {
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Voice request failed (HTTP ${response.status}).`)
  }
}

export async function voiceCapabilities(signal: AbortSignal): Promise<VoiceCapabilities> {
  const response = await fetch('/v1/audio/conversation/capabilities', { signal })
  await checkResponse(response)
  return response.json()
}

export async function transcribeVoice(audio: string, tier: string, model: string, signal: AbortSignal): Promise<string> {
  const response = await fetch('/v1/audio/conversation/transcriptions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio, tier, model }), signal,
  })
  await checkResponse(response)
  const result = await response.json()
  if (typeof result.text !== 'string') throw new Error('The voice API returned an invalid transcript.')
  return result.text.trim()
}

/** Backpressure from playback bounds queued audio; EOF without done is a failure. */
export async function streamVoice(
  tier: string, backend: VoiceBackend, messages: VoiceMessage[], signal: AbortSignal,
  onEvent: (event: VoiceEvent) => Promise<void> | void,
) {
  const response = await fetch('/v1/audio/conversation/turn', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ tier, backend, messages, speaker: 'Ryan' }), signal,
  })
  await checkResponse(response)
  if (!response.body || !response.headers.get('content-type')?.includes('text/event-stream'))
    throw new Error('The voice API did not return an event stream.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let done = false
  try {
    while (!done) {
      const part = await reader.read()
      signal.throwIfAborted()
      buffer += decoder.decode(part.value, { stream: !part.done })
      let boundary: RegExpExecArray | null
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        if (frame.length > 1024 * 1024) throw new Error('Oversized voice event.')
        const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:'))
          .map(line => line.slice(5).replace(/^ /, '')).join('\n')
        if (!data) continue
        const event = JSON.parse(data) as VoiceEvent
        if (event.type === 'error') throw new Error(event.detail)
        if (event.type === 'text' && typeof event.text !== 'string') throw new Error('Invalid voice text event.')
        if (event.type === 'audio' && (typeof event.audio !== 'string' || !Number.isInteger(event.sample_rate) || event.sample_rate < 8000 || event.sample_rate > 96000))
          throw new Error('Invalid voice audio event.')
        if (!['text', 'audio', 'done'].includes(event.type)) throw new Error('Unknown voice event.')
        signal.throwIfAborted()
        await onEvent(event)
        if (event.type === 'done') { done = true; break }
      }
      if (buffer.length > 1024 * 1024) throw new Error('Oversized voice event.')
      if (part.done && !done) throw new Error('The voice stream ended before completion. Start a new turn to retry.')
    }
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
