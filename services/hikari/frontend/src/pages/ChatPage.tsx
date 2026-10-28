import { useEffect, useRef, useState } from 'react'
import { streamChat, type ConversationMessage } from '../api/chat'
import { fetchExperimentalModels, type CheckpointStatus } from '../api/settings'

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
  const [models, setModels] = useState<CheckpointStatus[]>([])
  const [model, setModel] = useState('')
  const [modelError, setModelError] = useState(false)
  const active = useRef<AbortController | null>(null)
  const activeConversation = useRef<ConversationMessage[]>([])
  const conversationId = useRef<string | null>(null)
  const historyEnd = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    void fetchExperimentalModels(controller.signal).then(
      (available) => {
        if (!controller.signal.aborted)
          setModels(
            available.filter(
              (item) =>
                item.id === 'lfm3-1.7b-bf16' &&
                item.kind === 'llm' &&
                item.status === 'complete' &&
                item.inference_available,
            ),
          )
      },
      () => {
        if (!controller.signal.aborted) setModelError(true)
      },
    )
    return () => controller.abort()
  }, [])

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
    if (!conversationId.current) {
      const bytes = crypto.getRandomValues(new Uint8Array(16))
      conversationId.current = Array.from(bytes, (byte) =>
        byte.toString(16).padStart(2, '0'),
      ).join('')
    }
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
        conversationId.current,
        model || undefined,
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
      <div className="row">
        <label htmlFor="chat-model">Model</label>
        <select
          id="chat-model"
          value={model}
          disabled={pending || messages.length > 0 || !!retryMessages}
          onChange={(event) => setModel(event.target.value)}
        >
          <option value="">Hikari Settings model</option>
          {models.map((item) => (
            <option key={item.id} value={item.id}>
              {item.display_name ?? 'LFM3 1.7B BF16 (experimental)'}
            </option>
          ))}
        </select>
        {(messages.length > 0 || retryMessages) && (
          <span className="faint">New chat to change model.</span>
        )}
        {modelError && (
          <span role="status" className="faint">
            Trial models unavailable. Reopen Chat after checking Settings.
          </span>
        )}
      </div>
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
              this browser tab. Choose a downloaded trial model above, or use
              the model selected in Settings.
            </p>
          )}
          {messages.map((message, index) => (
            <article
              key={index}
              className={`chat-message chat-message--${message.role}`}
              aria-label={
                message.role === 'user' ? 'Your message' : 'Hikari response'
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
            placeholder="Message Hikari…"
            aria-label="Message Hikari"
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
                conversationId.current = null
                activeConversation.current = []
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
            line. The selected model’s token context determines conversation
            capacity.
          </p>
        </form>
      </div>
    </div>
  )
}
