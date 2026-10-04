import { createCompletion } from './generated/sdk.gen'
import type { ChatMessage } from './generated/types.gen'

export const MAX_MESSAGE_LENGTH = 8_000
export const MAX_HISTORY_MESSAGES = 24
export type ConversationMessage = Pick<ChatMessage, 'role' | 'text' | 'meta'>

export function chatRequest(messages: ConversationMessage[]) {
  if (!messages.length || messages.length > MAX_HISTORY_MESSAGES) {
    throw new Error('Start a new chat before sending more messages.')
  }
  if (
    messages.some(
      ({ text }) => !text.trim() || text.length > MAX_MESSAGE_LENGTH,
    )
  ) {
    throw new Error(
      `Each message must contain 1–${MAX_MESSAGE_LENGTH} characters.`,
    )
  }
  return { messages: messages.map(({ role, text }) => ({ role, text })) }
}

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
    if (status === 413 || status === 422)
      throw new Error(
        'The API rejected this conversation. Start a new chat or shorten your message.',
      )
    throw new Error(
      status
        ? `Chat request failed (HTTP ${status}). Please retry.`
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
