import type { ChatMessage, RequestRecord } from './generated/types.gen'

export function chatPendingStatus(
  pending: boolean,
  messages: Pick<ChatMessage, 'role' | 'text'>[],
) {
  if (!pending) return null
  const latest = messages.at(-1)
  return latest?.role === 'assistant' && latest.text.length > 0
    ? 'Receiving response…'
    : 'Waiting for the chat API…'
}

export function requestStatus(request: RequestRecord) {
  if (request.type === 'Video' && request.status === 202) {
    const state = request.video_status
    return {
      label: state === 'Done' ? 'Completed' : state ?? 'Accepted — job status unavailable',
      className: state === 'Done' ? 'success' : state === 'Failed' ? 'error' : state === 'Cancelled' ? 'warning' : 'muted',
    }
  }
  return {
    label: `HTTP ${request.status}`,
    className: request.status >= 500 ? 'error' : request.status >= 400 ? 'warning' : 'success',
  }
}
