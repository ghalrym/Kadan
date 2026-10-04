import { ActionButton } from '../components/Controls'
import { chatMessages } from '../data/playground'

export default function ChatPage() {
  return (
    <div className="chat-page">
      <div className="chat-history">
        <div className="chat-messages">
          {chatMessages.map((message) => (
            <article
              key={message.role}
              className={`chat-message chat-message--${message.role}`}
              aria-label={
                message.role === 'user' ? 'Your message' : 'Kadan response'
              }
            >
              <p>{message.text}</p>
              {message.role === 'assistant' && (
                <div className="row chat-message-actions">
                  <ActionButton variant="text">Copy</ActionButton>
                  <ActionButton variant="text">Regenerate</ActionButton>
                  <span className="push-right mono faint">{message.meta}</span>
                </div>
              )}
            </article>
          ))}
        </div>
      </div>
      <div className="chat-composer-wrap">
        <div className="chat-composer">
          <textarea
            rows={2}
            readOnly
            value=""
            placeholder="Message Kadan…"
            aria-label="Message Kadan"
          />
          <div className="row">
            <ActionButton>Attach</ActionButton>
            <ActionButton variant="muted" className="push-right">
              Send ↵
            </ActionButton>
          </div>
        </div>
      </div>
    </div>
  )
}
