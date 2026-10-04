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
