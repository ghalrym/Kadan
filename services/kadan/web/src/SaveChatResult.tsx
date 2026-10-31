import { useRef, useState } from "react";
import { agentPath, api } from "./api";
import type { Message, Resource } from "./types";

export default function SaveChatResult({ agentKey, conversationId, message, disabled }: {
  agentKey: string;
  conversationId: string;
  message: Message;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState(() => Array.from(
    message.content.trim().split("\n")[0].replace(/^#+\s*/, ""),
  ).slice(0, 100).join("") || "Saved reply");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<Resource | null>(null);
  const [error, setError] = useState("");
  const locked = useRef(false);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (locked.current || disabled || !title.trim()) return;
    locked.current = true;
    setSaving(true);
    setError("");
    try {
      const document = await api<Resource>(
        `${agentPath(agentKey)}/conversations/${conversationId}/messages/${message.id}/document`,
        { title: title.trim() },
      );
      setSaved(document);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      locked.current = false;
      setSaving(false);
    }
  }

  if (saved) return <p className="field-hint" role="status">
    Saved in Workspace: <strong>{saved.title}</strong>. Open Documents to read it or Tsuchi to edit it.
  </p>;
  return <div className="save-chat-result">
    <button type="button" className="text-link" disabled={saving || disabled}
      aria-expanded={open} onClick={() => setOpen(!open)}>
      {open ? "Close save form" : "Save to Workspace"}
    </button>
    {open && <form onSubmit={save}>
      <label>Document title
        <input value={title} maxLength={500} required disabled={saving || disabled}
          onChange={event => setTitle(event.target.value)} />
      </label>
      <p className="field-hint">Saves the stored reply and its linked files as an editable document.
        One document per reply; saving again returns that document and preserves any later edits.
        This does not schedule work or add the document to future chat context.</p>
      {error && <p role="alert" className="error">{error} Save was not confirmed.
        Retrying checks or creates the same document.</p>}
      <button type="submit" className="button" disabled={saving || disabled || !title.trim()}>
        {saving ? "Saving…" : error ? "Retry save" : "Save document"}
      </button>
    </form>}
  </div>;
}
