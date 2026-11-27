import { useEffect, useRef, useState } from "react";
import { Check, Eye, Pencil, X } from "lucide-react";
import { agentPath, api } from "./api";
import Markdown from "./Markdown";
import type { Resource } from "./types";

interface Props {
  agentKey: string;
  agentName: string;
  onClose: () => void;
}

const MAX_LENGTH = 16000;

/** Markdown editor modal for an agent's single saved persona. */
export default function PersonaEditor({ agentKey, agentName, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const pending = useRef(false);
  const [saved, setSaved] = useState<Resource | null>(null);
  const [markdown, setMarkdown] = useState("");
  const [mode, setMode] = useState<"write" | "preview">("write");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [latest, setLatest] = useState<Resource | null>(null);

  const dirty = !loading && markdown !== (saved?.markdown ?? "");

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    void loadInitial();
    return () => {
      active.current = false;
    };
  }, [agentKey]);

  async function loadInitial() {
    setLoading(true);
    setError("");
    try {
      const value = await api<Resource | null>(`${agentPath(agentKey)}/persona`);
      if (!active.current) return;
      setSaved(value);
      setMarkdown(value?.markdown ?? "");
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setLoading(false);
    }
  }

  function close() {
    if (pending.current) return;
    if (dirty && !window.confirm("Discard unsaved persona changes?")) return;
    active.current = false;
    onClose();
  }

  async function reloadLatest() {
    if (pending.current) return;
    pending.current = true;
    setSaving(true);
    setError("");
    try {
      const value = await api<Resource | null>(`${agentPath(agentKey)}/persona`);
      if (!active.current) return;
      setSaved(value);
      setLatest(value);
      setNotice("Latest saved version loaded below. Your draft is unchanged.");
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      pending.current = false;
      if (active.current) setSaving(false);
    }
  }

  async function save(event?: React.FormEvent) {
    event?.preventDefault();
    if (pending.current || loading || !markdown.trim()) return;
    pending.current = true;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const value = await api<Resource>(
        `${agentPath(agentKey)}/persona`,
        {
          markdown,
          expected_version: saved?.version ?? null,
        },
        "PUT",
      );
      if (!active.current) return;
      setSaved(value);
      setMarkdown(value.markdown);
      setLatest(null);
      setNotice("Saved");
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      pending.current = false;
      if (active.current) setSaving(false);
    }
  }

  const busy = loading || saving;

  return (
    <dialog
      ref={dialog}
      className="persona-modal"
      aria-labelledby="persona-editor-title"
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
    >
      <form className="persona-modal-form" onSubmit={save}>
        <header className="persona-modal-head">
          <div>
            <h2 id="persona-editor-title">{agentName}’s persona</h2>
            <p className="field-hint">
              Markdown instructions describing how {agentName} should respond and work with you.
            </p>
          </div>
          <button
            type="button"
            className="icon"
            aria-label="Close persona editor"
            disabled={saving}
            onClick={close}
          >
            <X size={18} />
          </button>
        </header>

        <div className="persona-modal-toolbar">
          <div className="persona-modal-tabs" role="group" aria-label="Editor view">
            <button
              type="button"
              aria-pressed={mode === "write"}
              className={mode === "write" ? "active" : ""}
              disabled={loading}
              onClick={() => setMode("write")}
            >
              <Pencil size={14} />
              Write
            </button>
            <button
              type="button"
              aria-pressed={mode === "preview"}
              className={mode === "preview" ? "active" : ""}
              disabled={loading}
              onClick={() => setMode("preview")}
            >
              <Eye size={14} />
              Preview
            </button>
          </div>
          <span className="persona-modal-count">
            {markdown.length.toLocaleString()} / {MAX_LENGTH.toLocaleString()}
          </span>
        </div>

        {loading ? (
          <p className="field-hint" role="status">Loading persona…</p>
        ) : mode === "write" ? (
          <label className="persona-modal-field">
            <span className="sr-only">Persona Markdown instructions</span>
            <textarea
              autoFocus
              required
              rows={14}
              maxLength={MAX_LENGTH}
              value={markdown}
              disabled={busy}
              placeholder={`How should ${agentName} talk, think, and work with you?`}
              onChange={(event) => setMarkdown(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                  event.preventDefault();
                  void save();
                }
              }}
            />
          </label>
        ) : (
          <div className="persona-modal-preview">
            {markdown.trim() ? (
              <Markdown>{markdown}</Markdown>
            ) : (
              <p className="field-hint">No persona instructions yet.</p>
            )}
          </div>
        )}

        {error && (
          <div role="alert" className="error">
            <span>{error}</span>
            <button type="button" disabled={busy} onClick={() => void reloadLatest()}>
              Load latest for comparison
            </button>
          </div>
        )}

        {latest && (
          <details open className="persona-modal-latest">
            <summary>Saved version · v{latest.version}</summary>
            <Markdown>{latest.markdown}</Markdown>
          </details>
        )}

        {notice && (
          <p role="status" className="success">
            {notice}
          </p>
        )}

        <div className="form-actions">
          <button type="button" disabled={saving} onClick={close}>
            {dirty ? "Discard" : "Close"}
          </button>
          <button className="primary" disabled={busy || !markdown.trim()}>
            <Check size={16} />
            {saving ? "Saving…" : "Save persona"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
