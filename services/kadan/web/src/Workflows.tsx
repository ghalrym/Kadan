import { useRef, useState } from "react";
import { agentPath, api } from "./api";
import Markdown from "./Markdown";
import ContextCheck from "./ContextCheck";
import type { SelectedReference } from "./RequestReferences";
import type { Resource } from "./types";
import { exportWorkflow, parseWorkflow, type WorkflowBundle, type WorkflowSelection } from "./workflows";

interface Props {
  agentKey: string;
  task: string;
  disabled: boolean;
  runDisabled: boolean;
  browserUrl: string | null;
  context: { conversationId: string; version: number; persona: string; references: SelectedReference[] };
  onRun: (selection: WorkflowSelection) => void;
}

export default function Workflows({ agentKey, task, disabled, runDisabled, browserUrl, context, onRun }: Props) {
  const requestVersion = useRef(0);
  const saving = useRef(false);
  const [items, setItems] = useState<Resource[]>([]);
  const [selected, setSelected] = useState<Resource | null>(null);
  const [title, setTitle] = useState("");
  const [instructions, setInstructions] = useState("");
  const [preview, setPreview] = useState<WorkflowBundle | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [needsCheck, setNeedsCheck] = useState(false);
  const path = `${agentPath(agentKey)}/workflows`;

  async function load() {
    if (saving.current) return;
    const request = ++requestVersion.current;
    setItems([]);
    setSelected(null);
    setBusy(true);
    setError("");
    try {
      const rows: Resource[] = [];
      let offset: number | null = 0;
      do {
        const page: { items: Resource[]; next_offset: number | null } = await api(`${path}?offset=${offset}`);
        rows.push(...page.items);
        offset = page.next_offset;
      } while (offset !== null);
      if (request !== requestVersion.current) return;
      setItems(rows);
      setSelected(null);
      setNeedsCheck(false);
    } catch (e) { if (request === requestVersion.current) setError((e as Error).message); }
    finally { if (request === requestVersion.current) setBusy(false); }
  }

  async function choose(id: string) {
    const request = ++requestVersion.current;
    setSelected(null);
    if (!id) { setBusy(false); return; }
    setBusy(true);
    setError("");
    try {
      const resource = await api<Resource>(`${path}/${encodeURIComponent(id)}`);
      if (request === requestVersion.current) setSelected(resource);
    } catch (e) { if (request === requestVersion.current) setError((e as Error).message); }
    finally { if (request === requestVersion.current) setBusy(false); }
  }

  async function save() {
    if (!preview || needsCheck || saving.current) return;
    saving.current = true;
    ++requestVersion.current;
    setBusy(true);
    setError("");
    try {
      const saved = await api<Resource>(path, preview);
      setItems(previous => [saved, ...previous]);
      setSelected(saved);
      setPreview(null);
      setTitle("");
      setInstructions("");
    } catch (e) {
      setNeedsCheck(true);
      setError(`${(e as Error).message} Creation may be unconfirmed. Reload and check the list before saving again.`);
    } finally { saving.current = false; setBusy(false); }
  }

  async function importFile(file: File | undefined) {
    if (!file) return;
    const request = ++requestVersion.current;
    setPreview(null);
    setSelected(null);
    setBusy(true);
    setError("");
    try {
      if (file.size > 80000) throw new Error("Workflow file is too large.");
      const imported = parseWorkflow(await file.text());
      if (request === requestVersion.current) setPreview(imported);
    } catch (e) { if (request === requestVersion.current) setError((e as Error).message); }
    finally { if (request === requestVersion.current) setBusy(false); }
  }

  function download() {
    if (!selected) return;
    try {
      const url = URL.createObjectURL(new Blob([exportWorkflow(selected)], { type: "application/json" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = "workflow.json";
      link.click();
      URL.revokeObjectURL(url);
    } catch (e) { setError((e as Error).message); }
  }

  return <details className="workflows" onToggle={event => { if (event.currentTarget === event.target && event.currentTarget.open) void load(); }}>
    <summary>Workflows</summary>
    <p>Save reusable instructions. Run now uses the current chat draft as a new task. Templates grant no additional permissions.</p>
    <div className="workflow-actions">
      <select aria-label="Workflow" value={selected?.id || ""} disabled={busy || disabled}
        onChange={event => void choose(event.target.value)}>
        <option value="">Choose a workflow…</option>
        {items.map(item => <option key={item.id} value={item.id}>{item.title} · v{item.version}</option>)}
      </select>
      <button disabled={busy} onClick={() => void load()}>Reload list</button>
      {browserUrl && <a href={browserUrl} target="_blank" rel="noreferrer">Edit Spaces and revisions in Tsuchi</a>}
    </div>
    {selected && <div className="workflow-preview">
      <ContextCheck agentKey={agentKey} {...context} task={task}
        workflow={{ id: selected.id, version: selected.version }} disabled={disabled || busy} />
      <strong>{selected.title} · version {selected.version}</strong>
      <Markdown>{selected.markdown}</Markdown>
      <details><summary>Full instructions</summary><pre>{selected.markdown}</pre></details>
      <p>New task: {task.trim() || "Write a new task in the chat draft below."}</p>
      <button disabled={disabled || runDisabled || busy || !task.trim() || !!preview || !selected.markdown.trim() || selected.markdown.length > 12000}
        onClick={() => onRun({ id: selected.id, version: selected.version })}>Run now</button>
      <button disabled={busy} onClick={download}>Export JSON</button>
    </div>}
    <details>
      <summary>Create or import</summary>
      <label>Name<input value={title} maxLength={500} onChange={event => { setTitle(event.target.value); setPreview(null); }} /></label>
      <label>Instructions<textarea rows={5} maxLength={12000} value={instructions}
        onChange={event => { setInstructions(event.target.value); setPreview(null); }} /></label>
      <button disabled={busy || disabled} onClick={() => {
        try { setPreview(parseWorkflow(JSON.stringify({ schema_version: 1, title, instructions }))); setSelected(null); setError(""); }
        catch (e) { setError((e as Error).message); }
      }}>Preview</button>
      <label>Import JSON<input type="file" accept="application/json,.json" disabled={busy || disabled}
        onChange={event => void importFile(event.target.files?.[0])} /></label>
    </details>
    {preview && <div className="workflow-preview">
      <strong>{preview.title} · version 1 bundle</strong>
      <Markdown>{preview.instructions}</Markdown>
      <details><summary>Full instructions</summary><pre>{preview.instructions}</pre></details>
      <p>Save creates a workflow Space. Importing and saving do not run it.</p>
      <button disabled={busy || disabled || needsCheck} onClick={() => void save()}>Save workflow</button>
      <button disabled={busy} onClick={() => setPreview(null)}>Cancel preview</button>
    </div>}
    {error && <p role="alert">{error}</p>}
  </details>;
}
