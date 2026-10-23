import { useEffect, useState } from "react";
import { Check, Plus, UserRound } from "lucide-react";
import { agentPath, api } from "./api";
import type { Resource } from "./types";
import Markdown from "./Markdown";

interface PersonaDraft {
  selected: Resource | null;
  title: string;
  markdown: string;
}
function savedDraft(key: string): PersonaDraft | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || "null");
    return value &&
      typeof value.title === "string" &&
      typeof value.markdown === "string"
      ? value
      : null;
  } catch {
    return null;
  }
}

export default function Personas({
  agentKey,
  personas,
  onChange,
  onBusy,
  blocked,
}: {
  agentKey: string;
  personas: Resource[];
  onChange: (rows: Resource[]) => void;
  onBusy: (busy: boolean) => void;
  blocked: boolean;
}) {
  const draftKey = `kadan.persona-draft.${agentKey}`;
  const [initialDraft] = useState(() => savedDraft(draftKey));
  const [selected, setSelected] = useState<Resource | null>(
    initialDraft?.selected || null,
  );
  const [editing, setEditing] = useState(Boolean(initialDraft));
  const [title, setTitle] = useState(initialDraft?.title || "");
  const [markdown, setMarkdown] = useState(initialDraft?.markdown || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [latest, setLatest] = useState<Resource | null>(null);
  const dirty =
    editing &&
    (title !== (selected?.title || "") ||
      markdown !== (selected?.markdown || ""));
  useEffect(() => {
    if (dirty)
      sessionStorage.setItem(
        draftKey,
        JSON.stringify({ selected, title, markdown }),
      );
    else sessionStorage.removeItem(draftKey);
  }, [draftKey, dirty, selected, title, markdown]);
  async function choose(value: Resource | null) {
    if (blocked) return;
    if (dirty && !window.confirm("Discard unsaved persona changes?")) return;
    setSaving(true);
    onBusy(true);
    setError("");
    setLatest(null);
    try {
      if (value)
        value = await api<Resource>(
          `${agentPath(agentKey)}/resources/${value.id}`,
        );
      setSelected(value);
      setTitle(value?.title || "");
      setMarkdown(value?.markdown || "");
      setError("");
      setNotice("");
      setEditing(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
      onBusy(false);
    }
  }
  async function compareLatest() {
    if (!selected) return;
    setSaving(true);
    onBusy(true);
    try {
      const value = await api<Resource>(
        `${agentPath(agentKey)}/resources/${selected.id}`,
      );
      setSelected(value);
      setLatest(value);
      onChange([value, ...personas.filter((item) => item.id !== value.id)]);
      setError("");
      setNotice(
        "Latest version loaded below. Your draft is unchanged.",
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
      onBusy(false);
    }
  }
  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (blocked) return;
    setSaving(true);
    onBusy(true);
    setError("");
    setNotice("");
    try {
      const base = `${agentPath(agentKey)}/resources`;
      const value = selected
        ? await api<Resource>(
            `${base}/${selected.id}`,
            {
              title: title.trim(),
              markdown,
              expected_version: selected.version,
            },
            "PUT",
          )
        : await api<Resource>(base, {
            kind: "persona",
            title: title.trim(),
            markdown,
          });
      onChange([value, ...personas.filter((item) => item.id !== value.id)]);
      setSelected(value);
      setTitle((current) => (current === title ? value.title : current));
      setMarkdown((current) => (current === markdown ? value.markdown : current));
      setLatest(null);
      setNotice("Saved");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
      onBusy(false);
    }
  }
  return (
    <section className="personas">
      <div className="persona-library">
        <button
          className="button"
          disabled={saving || blocked}
          onClick={() => choose(null)}
        >
          <Plus size={16} />
          New persona
        </button>
        {personas.map((item) => (
          <button
            disabled={saving || blocked}
            className={`resource-row ${selected?.id === item.id ? "selected" : ""}`}
            onClick={() => choose(item)}
            key={item.id}
          >
            <span className="initial" aria-hidden="true">
              {item.title.trim().charAt(0).toUpperCase()}
            </span>
            <strong>{item.title}</strong>
          </button>
        ))}
      </div>
      {editing ? (
        <form className="persona-editor" onSubmit={save}>
          <label>
            Name
            <input
              required
              value={title}
              maxLength={200}
              onChange={(event) => setTitle(event.target.value)}
              disabled={saving || blocked}
            />
          </label>
          <label>
            Instructions
            <textarea
              required
              value={markdown}
              rows={14}
              maxLength={16000}
              onChange={(event) => setMarkdown(event.target.value)}
              placeholder="How should this agent respond?"
              disabled={saving || blocked}
            />
          </label>
          {error && (
            <div role="alert" className="error">
              {error}
              {selected && (
                <button
                  type="button"
                  disabled={saving || blocked}
                  onClick={() => void compareLatest()}
                >
                  Load latest for comparison
                </button>
              )}
            </div>
          )}
          {latest && (
            <details open>
              <summary>Saved version · {latest.title}</summary>
              <Markdown>{latest.markdown}</Markdown>
            </details>
          )}
          {notice && (
            <p role="status" className="success">
              {notice}
            </p>
          )}
          <div className="form-actions">
            <button
              type="button"
              disabled={saving || blocked}
              onClick={() => setEditing(false)}
            >
              {dirty ? "Discard draft" : "Done"}
            </button>
            <button
              className="primary"
              disabled={saving || blocked || !title.trim() || !markdown.trim()}
            >
              <Check size={16} />
              {saving ? "Saving…" : "Save persona"}
            </button>
          </div>
        </form>
      ) : (
        <div className="detail-empty">
          <UserRound size={26} />
          <p>{personas.length ? "Select a persona" : "No personas yet"}</p>
        </div>
      )}
    </section>
  );
}
