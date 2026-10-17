import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { X } from "lucide-react";
import { agentPath, api } from "./api";
import Pet from "./Pet";
import { petSheet, usePetImage } from "./PetContext";
import { agentHue, type PetState } from "./pet";
import type { PetJob } from "./types";
import { randomUuid } from "./uuid";

const states: { value: PetState; label: string }[] = [
  { value: "idle", label: "Idle" },
  { value: "running-right", label: "Run right" },
  { value: "running-left", label: "Run left" },
  { value: "waving", label: "Wave" },
  { value: "jumping", label: "Jump" },
  { value: "failed", label: "Failure" },
  { value: "waiting", label: "Wait" },
  { value: "running", label: "Work" },
  { value: "review", label: "Review" },
];
const statuses: Record<PetJob["status"], string> = {
  queued: "Queued", running: "Creating", cancelling: "Cancelling", cancelled: "Cancelled",
  interrupted: "Interrupted", failed: "Failed", ready: "Ready to preview",
};
const active = (job: PetJob) => ["queued", "running", "cancelling"].includes(job.status);

interface CreationRequest { id: string; prompt: string }

function readPendingCreation(key: string): CreationRequest | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? "null");
    return value && typeof value.id === "string" && typeof value.prompt === "string" ? value : null;
  } catch { return null; }
}

interface Props {
  agentKey: string;
  agentName: string;
  selected: PetJob | null;
  onSelect: (id: string | null) => Promise<void>;
  onRefreshSelection: () => Promise<void>;
  onClose: () => void;
}

export default function PetEditor({ agentKey, agentName, selected, onSelect, onRefreshSelection, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const alive = useRef(true);
  const refreshRevision = useRef(0);
  const creationKey = `kadan.pet-creation.${agentKey}`;
  const [initialCreation] = useState(() => readPendingCreation(creationKey));
  const pendingCreation = useRef<CreationRequest | null>(initialCreation);
  const [jobs, setJobs] = useState<PetJob[]>([]);
  const [candidateId, setCandidateId] = useState(selected?.id ?? "");
  const [prompt, setPrompt] = useState(initialCreation?.prompt ?? "");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [preview, setPreview] = useState("idle");
  const candidate = jobs.find((job) => job.id === candidateId) ?? null;
  const sheet = useMemo(() => petSheet(candidate), [candidate]);
  const image = usePetImage(sheet);
  const path = `${agentPath(agentKey)}/pets`;
  const hasActive = jobs.some(active);

  const refresh = useCallback(async () => {
    const revision = ++refreshRevision.current;
    try {
      const rows = await api<PetJob[]>(path);
      if (!alive.current || revision !== refreshRevision.current) return;
      setJobs((current) => rows.map((job) => {
        const previous = current.find((item) => item.id === job.id);
        return previous && previous.version >= job.version ? previous : job;
      }));
      if (rows.some((job) => job.id === pendingCreation.current?.id)) {
        pendingCreation.current = null;
        sessionStorage.removeItem(creationKey);
      }
      setCandidateId((current) => current || rows[0]?.id || "");
      setPollError("");
    } catch (e) {
      if (alive.current && revision === refreshRevision.current) setPollError((e as Error).message);
    } finally {
      if (alive.current && revision === refreshRevision.current) setLoading(false);
    }
  }, [path, creationKey]);

  const refreshSelection = useCallback(async () => {
    try {
      await onRefreshSelection();
      if (alive.current) setSelectionError("");
    } catch (error) {
      if (alive.current) setSelectionError((error as Error).message);
    }
  }, [onRefreshSelection]);

  useEffect(() => {
    alive.current = true;
    dialog.current?.showModal();
    let stopped = false;
    let timer = 0;
    // Serial requests discover agent-created jobs without overlapping slow responses.
    const poll = async () => {
      await Promise.all([refresh(), refreshSelection()]);
      if (!stopped) timer = window.setTimeout(() => void poll(), 3000);
    };
    void poll();
    return () => {
      alive.current = false;
      stopped = true;
      clearTimeout(timer);
    };
  }, [refresh, refreshSelection]);

  function update(job: PetJob) {
    if (!alive.current) return;
    ++refreshRevision.current;
    setJobs((rows) => [job, ...rows.filter((row) => row.id !== job.id)]);
    setCandidateId(job.id);
  }

  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !prompt.trim() || hasActive) return;
    setBusy(true);
    setError("");
    if (pendingCreation.current?.prompt !== prompt.trim()) {
      pendingCreation.current = { id: randomUuid(), prompt: prompt.trim() };
      sessionStorage.setItem(creationKey, JSON.stringify(pendingCreation.current));
    }
    try {
      update(await api<PetJob>(path, pendingCreation.current));
      pendingCreation.current = null;
      sessionStorage.removeItem(creationKey);
      if (alive.current) setPrompt("");
    } catch (e) {
      if (alive.current) setError((e as Error).message);
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  async function changeJob(action: "cancel" | "resume") {
    if (!candidate || busy) return;
    setBusy(true);
    setError("");
    try {
      update(await api<PetJob>(`${path}/${encodeURIComponent(candidate.id)}/${action}`, {}));
    } catch (e) {
      if (alive.current) setError((e as Error).message);
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  async function select(id: string | null) {
    if (busy) return;
    setBusy(true);
    setSelectionError("");
    try {
      // App owns the request so reconciliation survives closing this editor.
      await onSelect(id);
    } catch (e) {
      if (alive.current) setSelectionError((e as Error).message);
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} className="pet-editor" aria-labelledby="pet-editor-title"
      style={{ "--agent-h": agentHue(agentKey) } as React.CSSProperties}
      onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <header className="pet-editor-head">
        <h2 id="pet-editor-title">{agentName}’s pet</h2>
        <button className="icon" aria-label="Close pet editor" onClick={onClose}><X size={18} /></button>
      </header>
      <form className="pet-create" onSubmit={create}>
        <label htmlFor="pet-prompt">Character and art style</label>
        <textarea id="pet-prompt" value={prompt} rows={3} maxLength={4000} required
          placeholder="A small copper fox, illustrated in watercolor"
          onChange={(event) => setPrompt(event.target.value)} />
        <div className="form-actions">
          <button className="primary" disabled={busy || loading || hasActive || !prompt.trim()}>
            {busy ? "Saving…" : "Create pet"}
          </button>
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {selectionError && <div className="banner" role="alert">Could not confirm the selected pet: {selectionError}
        <button disabled={busy} onClick={() => void refreshSelection()}>Retry</button>
      </div>}
      {pollError && <div className="banner" role="alert">Updates unavailable: {pollError}
        <button onClick={() => void refresh()}>Retry</button>
      </div>}
      {loading ? <p role="status">Loading pets…</p> : jobs.length === 0 ? <p className="quiet">No pet candidates yet.</p> : (
        <div className="pet-candidates">
          <label htmlFor="pet-candidate">Candidate</label>
          <select id="pet-candidate" value={candidateId} onChange={(event) => {
            setCandidateId(event.target.value); setPreview("idle"); setError("");
          }}>
            {jobs.map((job) => <option key={job.id} value={job.id}>
              {job.prompt.length > 70 ? `${job.prompt.slice(0, 70)}…` : job.prompt} · {selected?.id === job.id ? "In use" : statuses[job.status]}
            </option>)}
          </select>
          {candidate && <>
            <p className="pet-job-prompt">{candidate.prompt}</p>
            <div className="pet-job-state" role="status">
              <span>{selected?.id === candidate.id ? "In use" : statuses[candidate.status]}</span>
              {candidate.status !== "ready" && <span>{candidate.completed_steps} / {candidate.total_steps} steps</span>}
            </div>
            {active(candidate) && <>
              <progress value={candidate.completed_steps} max={candidate.total_steps} aria-label="Pet creation progress" />
              <p className="quiet">{candidate.stage.replaceAll("_", " ")}</p>
              <p className="field-hint">Creation continues when you close this window. Your current pet stays in use.</p>
            </>}
            {candidate.error && <p className="error">{candidate.error}</p>}
            {candidate.status === "interrupted" && <p className="field-hint">Resume to continue from the saved artwork.</p>}
            {sheet && <>
              <div className="pet-preview" aria-label="Candidate preview">
                {image.ready ? <Pet sheet={sheet} size={192} grounded look
                  state={preview.startsWith("look:") ? "idle" : preview as PetState}
                  lookDirection={preview.startsWith("look:") ? Number(preview.slice(5)) : undefined} />
                  : <p role="status">{image.failed ? "The pet artwork could not be loaded." : "Loading artwork…"}</p>}
              </div>
              <label htmlFor="pet-animation">Preview animation</label>
              <select id="pet-animation" value={preview} onChange={(event) => setPreview(event.target.value)}>
                {states.map((state) => <option key={state.value} value={state.value}>{state.label}</option>)}
                {Array.from({ length: 16 }, (_, index) => <option key={index} value={`look:${index * 22.5}`}>
                  Look {index * 22.5}°{index === 0 ? " · up" : index === 4 ? " · right" : index === 8 ? " · down" : index === 12 ? " · left" : ""}
                </option>)}
              </select>
            </>}
            <div className="form-actions">
              {active(candidate) && <button disabled={busy || candidate.status === "cancelling"} onClick={() => void changeJob("cancel")}>
                {candidate.status === "cancelling" ? "Cancelling…" : "Cancel creation"}
              </button>}
              {["failed", "interrupted", "cancelled"].includes(candidate.status) && <button disabled={busy || hasActive} onClick={() => void changeJob("resume")}>Resume creation</button>}
              {sheet && <button className="primary" disabled={busy || !image.ready || selected?.id === candidate.id}
                onClick={() => void select(candidate.id)}>{selected?.id === candidate.id ? "In use" : "Use pet"}</button>}
            </div>
          </>}
        </div>
      )}
      {selected && <button className="pet-reset" disabled={busy} onClick={() => void select(null)}>Use hatchling</button>}
    </dialog>
  );
}
