import type { VoiceMessage } from '../api/voice'

type Exchange = { user: string; assistant: string }
const encoder = new TextEncoder()
const byteLength = (text: string) => encoder.encode(text).length
const exchangeBytes = (exchange: Exchange) => byteLength(exchange.user) + byteLength(exchange.assistant)

/** Only complete user/assistant pairs enter history; the current request stays provisional. */
export class VoiceHistory {
  private exchanges: Exchange[] = []

  request(user: string): VoiceMessage[] {
    if (Array.from(user).length > 8000 || byteLength(user) > 32000)
      throw new Error('The transcript exceeds the conversation limit (8000 characters / 32000 UTF-8 bytes). Try a shorter utterance.')
    this.prune(1, byteLength(user))
    return [
      ...this.exchanges.flatMap(exchange => [
        { role: 'user' as const, content: exchange.user },
        { role: 'assistant' as const, content: exchange.assistant },
      ]),
      { role: 'user', content: user },
    ]
  }

  commit(user: string, response: string, interrupted: boolean) {
    if (!response.trim()) return
    const annotation = interrupted
      ? '[Speech playback stopped before completion. This is generated response text; some or all may not have been heard.]\n'
      : ''
    let assistant = annotation + response
    const characters = Array.from(assistant)
    if (characters.length > 8000) {
      const suffix = '\n[Response truncated in conversation context.]'
      assistant = characters.slice(0, 8000 - suffix.length).join('') + suffix
    }
    this.exchanges.push({ user, assistant })
    this.prune(0, 0)
  }

  private prune(reservedMessages: number, reservedBytes: number) {
    let bytes = this.exchanges.reduce((total, exchange) => total + exchangeBytes(exchange), reservedBytes)
    while (this.exchanges.length && (this.exchanges.length * 2 + reservedMessages > 40 || bytes > 32000))
      bytes -= exchangeBytes(this.exchanges.shift()!)
  }
}
