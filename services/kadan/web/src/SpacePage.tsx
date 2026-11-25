import { useState } from "react";
import { ArrowUp, FolderOpen, MessageSquare, Settings2, Trash2 } from "lucide-react";
import { listedConversations, relativeTime } from "./conversations";
import type { ConversationSummary, Space } from "./types";

interface Props {
  space: Space;
  conversations: ConversationSummary[];
  disabled: boolean;
  onStart: (text: string) => Promise<void>;
  onOpenChat: (id: string) => void;
  onEdit: () => void;
  onDelete: () => void;
}

/** A Space's home: start a chat in it, review its instructions and reopen its chats. */
export default function SpacePage({ space, conversations, disabled, onStart, onOpenChat, onEdit, onDelete }: Props) {
  const [text, setText] = useState("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const chats = listedConversations(conversations, space.id, null);
  const now = Date.now();
  async function start(event: React.FormEvent) {
    event.preventDefault();
    if (!text.trim() || starting || disabled) return;
    setStarting(true);
    setError("");
    try {
      await onStart(text);
    } catch (e) {
      setError((e as Error).message);
      setStarting(false);
    }
  }
  return (
    <section className="page space-page" aria-labelledby="space-title">
      <div className="page-inner">
        <header className="page-head">
          <span className="page-icon" aria-hidden="true"><FolderOpen size={22} /></span>
          <h1 id="space-title">{space.title}</h1>
          <div className="page-actions">
            <button className="icon" aria-label="Space settings" title="Space settings" disabled={disabled} onClick={onEdit}>
              <Settings2 size={17} />
            </button>
            <button className="icon" aria-label="Delete Space" title="Delete Space" disabled={disabled} onClick={onDelete}>
              <Trash2 size={17} />
            </button>
          </div>
        </header>
        <form className="space-start" onSubmit={start}>
          <label className="sr-only" htmlFor="space-start">New chat in {space.title}</label>
          <textarea id="space-start" rows={2} value={text} disabled={starting || disabled}
            placeholder={`New chat in ${space.title}`}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }} />
          <button className="send" aria-label="Start chat" disabled={!text.trim() || starting || disabled}>
            <ArrowUp size={18} />
          </button>
        </form>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="instructions-card" onClick={onEdit} disabled={disabled}>
          <span className="instructions-card-head">
            <strong>Instructions</strong>
            <span>{space.instructions.trim() ? "Edit" : "Add"}</span>
          </span>
          <span className="instructions-card-body">
            {space.instructions.trim() || "Tell the agent how to respond in this Space: goals, tone or background to keep in mind."}
          </span>
        </button>
        <h2 className="page-section-title">Chats</h2>
        {chats.length === 0 ? (
          <p className="page-empty">
            <MessageSquare size={18} aria-hidden="true" />
            Chats you start in this Space appear here.
          </p>
        ) : (
          <ul className="space-chat-list">
            {chats.map((chat) => (
              <li key={chat.id}>
                <button onClick={() => onOpenChat(chat.id)}>
                  <span className="space-chat-text">
                    <strong>{chat.title}</strong>
                    {chat.preview && chat.preview !== chat.title && <span>{chat.preview}</span>}
                  </span>
                  <time dateTime={chat.updated_at}>{relativeTime(chat.updated_at, now)}</time>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
