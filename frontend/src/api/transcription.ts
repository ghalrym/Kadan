import { transcribeAudio } from './generated/sdk.gen'
import type { TranscriptionRequest } from './generated/types.gen'

/** Submit inline audio; preserve backend errors and valid silent transcripts. */
export async function requestTranscription(audio: string, formatting: boolean, signal: AbortSignal): Promise<string> {
  const reference = audio.trim()
  if (!reference) throw new Error('Enter an audio reference.')
  const body: TranscriptionRequest = { audio: reference, formatting }
  const result = await transcribeAudio({ body, signal })
  if (!result.response?.ok) {
    const detail = result.error && typeof result.error === 'object' && 'detail' in result.error ? result.error.detail : null
    throw new Error(typeof detail === 'string' ? detail : 'Transcription failed. Check the backend and selected checkpoint.')
  }
  if (typeof result.data?.text !== 'string') throw new Error('The API returned an invalid transcript.')
  return result.data.text
}

/** Convert browser audio to Whisper's PCM input without sending it to another service. */
export async function recordingToWav(blob: Blob): Promise<string> {
  const decoder = new AudioContext()
  try {
    const decoded = await decoder.decodeAudioData(await blob.arrayBuffer())
    const renderer = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * 16000)), 16000)
    const source = renderer.createBufferSource()
    source.buffer = decoded
    source.connect(renderer.destination)
    source.start()
    const samples = (await renderer.startRendering()).getChannelData(0)
    const buffer = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(buffer)
    const write = (offset: number, text: string) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)) }
    write(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); write(8, 'WAVEfmt ')
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true)
    view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true)
    write(36, 'data'); view.setUint32(40, samples.length * 2, true)
    samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, sample)) * (sample < 0 ? 32768 : 32767)), true))
    return await new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.onerror = () => reject(new Error('Cannot read recorded audio.'))
      reader.readAsDataURL(new Blob([buffer], { type: 'audio/wav' }))
    })
  } finally { await decoder.close() }
}
