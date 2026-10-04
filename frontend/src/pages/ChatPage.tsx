import { useEffect, useRef, useState } from 'react'
import {
  MAX_HISTORY_MESSAGES,
  MAX_MESSAGE_LENGTH,
  requestChat,
  type ConversationMessage,
} from '../api/chat'

export default function ChatPage() {
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryMessages, setRetryMessages] = useState<
    ConversationMessage[] | null
  >(null)
  const [notice, setNotice] = useState('')
  const active = useRef<AbortController | null>(null)
  const historyEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => () => active.current?.abort(), [])
  useEffect(() => {
    historyEnd.current?.scrollIntoView?.({ block: 'nearest' })
  }, [messages, pending])

  async function send(conversation: ConversationMessage[]) {
    if (active.current) return
    const controller = new AbortController()
    active.current = controller
    setPending(true)
    setError(null)
    setNotice('')
    setRetryMessages(null)
    setMessages(conversation)
    try {
      const message = await requestChat(conversation, controller.signal)
      if (active.current !== controller || controller.signal.aborted) return
      setMessages([...conversation, message])
    } catch (failure) {
      if (active.current !== controller || controller.signal.aborted) return
      setError(
        failure instanceof Error
          ? failure.message
          : 'Chat request failed. Please retry.',
      )
      setRetryMessages(conversation)
    } finally {
      if (active.current === controller) {
        active.current = null
        setPending(false)
      }
    }
  }

  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    setRetryMessages(messages)
    setNotice(
      'Request cancelled in this browser. The backend may still be finishing inference.',
    )
  }

  function submit() {
    const text = draft.trim()
    if (
      !text ||
      pending ||
      retryMessages ||
      messages.length >= MAX_HISTORY_MESSAGES
    )
      return
    setDraft('')
    void send([...messages, { role: 'user', text }])
  }

  return (
    <div className="chat-page">
      <div
        className="chat-history"
        role="log"
        aria-label="Chat conversation"
        aria-busy={pending}
      >
        <div className="chat-messages">
          {messages.length === 0 && (
            <p className="muted">
              Send a message to test the chat API. Conversation history stays in
              this browser tab; the backend determines the response and model.
            </p>
          )}
          {messages.map((message, index) => (
            <article
              key={index}
              className={`chat-message chat-message--${message.role}`}
              aria-label={
                message.role === 'user' ? 'Your message' : 'Kadan response'
              }
            >
              <p>{message.text}</p>
              {message.meta && (
                <span className="mono faint">{message.meta}</span>
              )}
            </article>
          ))}
          {pending && <p role="status">Waiting for the chat API…</p>}
          <div ref={historyEnd} />
        </div>
      </div>
      <div className="chat-composer-wrap">
        {error && <p role="alert">{error}</p>}
        {notice && <p role="status">{notice}</p>}
        {retryMessages && (
          <button
            type="button"
            className="button button--secondary"
            onClick={() => void send(retryMessages)}
          >
            Retry last message
          </button>
        )}
        {messages.length >= MAX_HISTORY_MESSAGES && (
          <p role="status">
            Conversation limit reached. Start a new chat to continue.
          </p>
        )}
        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault()
            submit()
          }}
        >
          <textarea
            rows={2}
            value={draft}
            maxLength={MAX_MESSAGE_LENGTH}
            disabled={
              pending ||
              !!retryMessages ||
              messages.length >= MAX_HISTORY_MESSAGES
            }
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (
                event.key === 'Enter' &&
                !event.shiftKey &&
                !event.nativeEvent.isComposing
              ) {
                event.preventDefault()
                submit()
              }
            }}
            placeholder="Message Kadan…"
            aria-label="Message Kadan"
            aria-describedby="chat-input-help"
          />
          <div className="row">
            <button
              type="button"
              className="button button--secondary"
              disabled={pending || (!messages.length && !draft)}
              onClick={() => {
                setMessages([])
                setDraft('')
                setError(null)
                setRetryMessages(null)
                setNotice('')
              }}
            >
              New chat
            </button>
            {pending ? (
              <button
                type="button"
                className="button button--secondary push-right"
                onClick={cancel}
              >
                Cancel
              </button>
            ) : (
              <button
                type="submit"
                className="button button--primary push-right"
                disabled={
                  !draft.trim() ||
                  !!retryMessages ||
                  messages.length >= MAX_HISTORY_MESSAGES
                }
              >
                Send ↵
              </button>
            )}
          </div>
          <p id="chat-input-help" className="faint">
            {draft.length}/{MAX_MESSAGE_LENGTH} characters · Enter to send,
            Shift+Enter for a new line · Up to {MAX_HISTORY_MESSAGES / 2} turns
            per chat
          </p>
        </form>
      </div>
    </div>
  )
}
