import { createCompletion } from './generated/sdk.gen'
import type { ChatMessage } from './generated/types.gen'

export type ConversationMessage = Pick<ChatMessage, 'role' | 'text' | 'meta'>

/**
 * Build the API conversation without selecting a model or imposing product length caps.
 * Rejects empty/blank turns and drops display-only metadata; the input is not mutated.
 */
export function chatRequest(messages: ConversationMessage[]) {
  if (!messages.length) {
    throw new Error('Enter a message before sending.')
  }
  if (messages.some(({ text }) => !text.trim())) {
    throw new Error('Messages must not be blank.')
  }
  return { messages: messages.map(({ role, text }) => ({ role, text })) }
}

/**
 * Send the complete conversation and return one validated assistant turn.
 * The supplied signal cancels the browser request; it does not guarantee server
 * inference has stopped. Network, API, and malformed-response failures reject.
 */
export async function requestChat(
  messages: ConversationMessage[],
  signal: AbortSignal,
): Promise<ConversationMessage> {
  const result = await createCompletion({ body: chatRequest(messages), signal })
  if (!result.response?.ok) {
    const status = result.response?.status
    if (status === 503)
      throw new Error(
        'Model unavailable. Check model readiness in Settings, then retry.',
      )
    if (status === 409 || status === 429)
      throw new Error(
        'The model is busy. Wait for the current operation, then retry.',
      )
    const apiError = result.error as { detail?: unknown } | undefined
    const detail =
      typeof apiError?.detail === 'string' ? apiError.detail : undefined
    if (status === 413)
      throw new Error(
        detail ??
          'The API rejected the conversation size. Check the loaded model’s configured token context and server request limits.',
      )
    if (status === 422)
      throw new Error(
        detail ??
          'The API rejected this conversation. Check the message format and the loaded model’s configured token context.',
      )

    throw new Error(
      status
        ? (detail ?? `Chat request failed (HTTP ${status}). Please retry.`)
        : 'Cannot reach the chat API. Check that the backend is running, then retry.',
    )
  }
  const message = result.data?.message
  if (
    !message ||
    message.role !== 'assistant' ||
    typeof message.text !== 'string' ||
    !message.text.trim()
  ) {
    throw new Error('The API returned an invalid chat response. Please retry.')
  }
  return message
}

/** Read SSE incrementally; an EOF without the terminal chunk and [DONE] is a failure. */
export async function streamChat(
  messages: ConversationMessage[],
  signal: AbortSignal,
  onText: (text: string) => void,
  conversationId?: string,
): Promise<ConversationMessage> {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      ...chatRequest(messages),
      stream: true,
      conversation_id: conversationId,
    }),
    signal,
  })
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    throw new Error(
      error?.detail ?? `Chat request failed (HTTP ${response.status}).`,
    )
  }
  if (
    !response.body ||
    !response.headers.get('content-type')?.includes('text/event-stream')
  )
    throw new Error('The API did not return a chat stream.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = '',
    text = '',
    terminal = false,
    done = false
  try {
    while (!done) {
      const part = await reader.read()
      signal.throwIfAborted()
      buffer += decoder.decode(part.value, { stream: !part.done })
      // Normalize CRLF only after a complete event, preserving fragmented CRLF.
      let boundary: RegExpExecArray | null
      while ((boundary = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        const data = frame
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).replace(/^ /, ''))
          .join('\n')
        if (!data) continue
        if (data === '[DONE]') {
          if (!terminal)
            throw new Error(
              'The chat stream ended without a completion status.',
            )
          done = true
          break
        }
        const event = JSON.parse(data)
        if (event.error)
          throw new Error(event.error.message ?? 'Chat generation failed.')
        const choice = event.choices?.[0]
        if (
          event.object !== 'chat.completion.chunk' ||
          choice?.index !== 0 ||
          !choice.delta
        )
          throw new Error('The API returned an invalid chat stream.')
        if (terminal) throw new Error('The API sent content after completion.')
        if (choice.delta.content !== undefined) {
          if (typeof choice.delta.content !== 'string')
            throw new Error('Invalid streamed content.')
          text += choice.delta.content
          if (choice.delta.content) onText(text)
        }
        if (choice.finish_reason != null) {
          if (!['stop', 'length'].includes(choice.finish_reason))
            throw new Error('Unsupported chat completion status.')
          terminal = true
        }
      }
      if (buffer.length > 1024 * 1024)
        throw new Error('The API returned an oversized stream event.')
      if (part.done && !done)
        throw new Error('The chat stream was interrupted. Please retry.')
    }
    if (!text.trim())
      throw new Error('The API returned an empty chat response.')
    return { role: 'assistant', text }
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
