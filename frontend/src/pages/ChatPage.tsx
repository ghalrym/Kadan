import { useEffect, useRef, useState } from 'react'
import { streamChat, type ConversationMessage } from '../api/chat'

/** Keep tab-local conversation history and one cancellable request with explicit retry. */
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
  const activeConversation = useRef<ConversationMessage[]>([])
  const historyEnd = useRef<HTMLDivElement | null>(null)

  useEffect(
    () => () => {
      active.current?.abort()
      active.current = null
    },
    [],
  )
  useEffect(() => {
    historyEnd.current?.scrollIntoView?.({ block: 'nearest' })
  }, [messages, pending])

  /**
   * Submit a conversation snapshot once, retaining it for retry on failure.
   * Controller ownership prevents canceled or superseded responses from updating UI.
   */
  async function send(conversation: ConversationMessage[]) {
    if (active.current) return
    const controller = new AbortController()
    active.current = controller
    activeConversation.current = conversation
    setPending(true)
    setError(null)
    setNotice('')
    setRetryMessages(null)
    setMessages(conversation)
    try {
      const message = await streamChat(
        conversation,
        controller.signal,
        (text) => {
          if (active.current === controller && !controller.signal.aborted)
            setMessages([...conversation, { role: 'assistant', text }])
        },
      )
      if (active.current !== controller || controller.signal.aborted) return
      setMessages([...conversation, message])
    } catch (failure) {
      if (active.current !== controller || controller.signal.aborted) return
      setMessages(conversation)
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

  /** Stop accepting this response; retry reuses the existing user turn in this UI. */
  function cancel() {
    active.current?.abort()
    active.current = null
    setPending(false)
    const retry = activeConversation.current
    setMessages(retry)
    setRetryMessages(retry)
    setNotice(
      'Request cancelled in this browser. The backend may still be finishing inference.',
    )
  }

  /** Append a nonblank draft once; a failed request must be retried before another turn. */
  function submit() {
    const text = draft
    if (!text.trim() || pending || retryMessages) return
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
        {(error || notice || retryMessages) && (
          <div className={`chat-banner${error ? ' chat-banner--error' : ''}`}>
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
          </div>
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
            disabled={pending || !!retryMessages}
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
                disabled={!draft.trim() || !!retryMessages}
              >
                Send ↵
              </button>
            )}
          </div>
          <p id="chat-input-help" className="faint">
            {draft.length} characters · Enter to send, Shift+Enter for a new
            line. The loaded model’s configured token context determines
            conversation capacity.
          </p>
        </form>
      </div>
    </div>
  )
}
