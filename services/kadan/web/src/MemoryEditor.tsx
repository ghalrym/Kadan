import { useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import Markdown from "./Markdown";
import type { Resource } from "./types";

interface Draft {
  base: Resource;
  title: string;
  markdown: string;
}

function readDraft(key: string, memory: Resource): Draft | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || "null");
    return value?.base?.id === memory.id && value.base.agent_key === memory.agent_key &&
      value.base.kind === "memory" && Number.isInteger(value.base.version) && value.base.version > 0 &&
      typeof value.title === "string" && typeof value.markdown === "string" ? value : null;
  } catch {
    return null;
  }
}

export default function MemoryEditor({ agentKey, memory, onSaved }: {
  agentKey: string;
  memory: Resource;
  onSaved: (memory: Resource) => void;
}) {
  const draftKey = `kadan.memory-draft.${agentKey}.${memory.id}`;
  const [saved, setSaved] = useState(memory);
  const [draft, setDraft] = useState<Draft | null>(() => readDraft(draftKey, memory));
  const [needsCheck, setNeedsCheck] = useState(Boolean(draft));
  const [comparison, setComparison] = useState<Resource | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const lock = useRef(false);
  const active = useRef(true);
  const path = `${agentPath(agentKey)}/memories/${memory.id}`;
  const dirty = !!draft && (draft.title.trim() !== draft.base.title || draft.markdown !== draft.base.markdown);

  useEffect(() => { setSaved(memory); }, [memory]);
  useEffect(() => {
    active.current = true;
    return () => { active.current = false; };
  }, []);
  useEffect(() => {
    try {
      if (draft && (dirty || needsCheck)) sessionStorage.setItem(draftKey, JSON.stringify(draft));
      else sessionStorage.removeItem(draftKey);
    } catch {
      if (draft) setError("This tab could not keep your draft. Keep this editor open until you save or discard it.");
    }
  }, [draftKey, draft, dirty, needsCheck]);
  useEffect(() => {
    if (!dirty && !needsCheck) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty, needsCheck]);

  async function load(editing: boolean) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const value = await api<Resource>(path);
      if (!active.current) return;
      setSaved(value);
      onSaved(value);
      if (editing || !draft) {
        setDraft({ base: value, title: value.title, markdown: value.markdown });
        setComparison(null);
      } else {
        setComparison(value);
        setDraft({ ...draft, base: value });
        setNotice(draft.title.trim() === value.title && draft.markdown === value.markdown
          ? "The saved version matches your correction. No further save is needed."
          : "Saved version loaded below. Your correction is unchanged; compare it before saving again.");
      }
      setNeedsCheck(false);
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      lock.current = false;
      if (active.current) setBusy(false);
    }
  }

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (!draft || !dirty || needsCheck || lock.current || !draft.title.trim()) return;
    lock.current = true;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const value = await api<Resource>(path, {
        title: draft.title.trim(), markdown: draft.markdown, expected_version: draft.base.version,
      }, "PUT");
      if (!active.current) return;
      setSaved(value);
      onSaved(value);
      setDraft(null);
      setComparison(null);
      setNotice("Correction saved. Work already prepared keeps its earlier context; future recall is not guaranteed.");
    } catch (e) {
      if (active.current) {
        setNeedsCheck(true);
        setError(`${(e as Error).message} Check the saved version before retrying; the change may already have been saved.`);
      }
    } finally {
      lock.current = false;
      if (active.current) setBusy(false);
    }
  }

  function close() {
    if (lock.current || (dirty && !window.confirm("Discard this unsaved memory correction?"))) return;
    setDraft(null);
    setComparison(null);
    setNeedsCheck(false);
    setError("");
    setNotice("");
  }

  return <div className="memory-editor">
    <dl className="metadata">
      <dt>Created</dt><dd>{new Date(saved.created_at).toLocaleString()}</dd>
      <dt>Saved</dt><dd>{new Date(saved.updated_at).toLocaleString()}</dd>
      <dt>Version</dt><dd>{saved.version}</dd>
    </dl>
    {saved.sources?.length > 0 && <ul aria-label="Memory sources">
      {saved.sources.map((source, index) => <li key={index}>
        <a href={source.url} target="_blank" rel="noreferrer">{source.label || source.url}</a>
      </li>)}
    </ul>}
    {draft ? <form className="persona-editor" onSubmit={save}>
      <p className="field-hint">Correct this saved memory. Its links and other metadata stay intact. Drafts are kept in this browser tab when you navigate away.</p>
      <label>Title<input required maxLength={500} value={draft.title} disabled={busy}
        onChange={event => setDraft({ ...draft, title: event.target.value })} /></label>
      <label>Memory<textarea rows={12} maxLength={2000000} value={draft.markdown} disabled={busy}
        onChange={event => setDraft({ ...draft, markdown: event.target.value })} /></label>
      {comparison && <details open><summary>Saved version · {comparison.title}</summary>
        <Markdown>{comparison.markdown || "No text in this memory."}</Markdown>
      </details>}
      {needsCheck && <button type="button" disabled={busy} onClick={() => void load(false)}>Check saved version</button>}
      <div className="form-actions">
        <button type="button" disabled={busy} onClick={close}>{dirty ? "Discard correction" : "Done"}</button>
        <button className="primary" disabled={busy || needsCheck || !dirty || !draft.title.trim()}>
          {busy ? "Working…" : "Save correction"}
        </button>
      </div>
    </form> : <>
      <Markdown>{saved.markdown || "No text in this memory."}</Markdown>
      <button type="button" disabled={busy} onClick={() => void load(true)}>{busy ? "Loading…" : "Correct memory"}</button>
    </>}
    {error && <p role="alert" className="error">{error}</p>}
    {notice && <p role="status" className="notice">{notice}</p>}
  </div>;
}
