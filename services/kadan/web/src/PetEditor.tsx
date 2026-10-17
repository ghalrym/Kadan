import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { X } from "lucide-react";
import { agentPath, api, ApiError } from "./api";
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
interface RevisionRequest {
  parentId: string;
  id: string;
  expected_version: number;
  frame_names: string[];
  reason: string;
}

function readPendingRevision(key: string): RevisionRequest | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? "null");
    return value && typeof value.parentId === "string" && typeof value.id === "string"
      && Number.isInteger(value.expected_version) && typeof value.reason === "string"
      && Array.isArray(value.frame_names) && value.frame_names.length === 1
      && typeof value.frame_names[0] === "string" ? value : null;
  } catch { return null; }
}

function frameLabel(name: string): string {
  const [animation, index] = name.split("/");
  const label = states.find((state) => state.value === animation)?.label
    ?? (animation.startsWith("look-") ? `Look ${Number(animation.slice(5)) * 180 + Number(index) * 22.5}°` : animation);
  return `${label} · frame ${Number(index) + 1}`;
}

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
  const revisionKey = `kadan.pet-revision.${agentKey}`;
  const [initialRevision] = useState(() => readPendingRevision(revisionKey));
  const pendingRevision = useRef<RevisionRequest | null>(initialRevision);
  const [revisionPending, setRevisionPending] = useState(!!initialRevision);
  const [repairFrame, setRepairFrame] = useState(initialRevision?.frame_names[0] ?? "");
  const [repairReason, setRepairReason] = useState(initialRevision?.reason ?? "");
  const [jobs, setJobs] = useState<PetJob[]>([]);
  const [candidateId, setCandidateId] = useState(initialRevision?.parentId ?? selected?.id ?? "");
  const [prompt, setPrompt] = useState(initialCreation?.prompt ?? "");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [preview, setPreview] = useState("idle");
  const candidate = jobs.find((job) => job.id === candidateId) ?? null;
  const repairFrames = candidate?.steps.filter((step) => step.name !== "base" && step.method !== "canonical") ?? [];
  const frame = repairFrames.find((step) => step.name === repairFrame) ?? repairFrames[0];
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
      const completedRevision = pendingRevision.current;
      if (completedRevision && rows.some((job) => job.id === completedRevision.id)) {
        pendingRevision.current = null;
        sessionStorage.removeItem(revisionKey);
        setRevisionPending(false);
        setRepairReason("");
        setCandidateId(completedRevision.id);
        setPreview("idle");
        setError("");
      } else {
        setCandidateId((current) => current || rows[0]?.id || "");
      }
      setPollError("");
    } catch (e) {
      if (alive.current && revision === refreshRevision.current) setPollError((e as Error).message);
    } finally {
      if (alive.current && revision === refreshRevision.current) setLoading(false);
    }
  }, [path, creationKey, revisionKey]);

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
    setJobs((rows) => {
      const previous = rows.find((row) => row.id === job.id);
      return [previous && previous.version > job.version ? previous : job, ...rows.filter((row) => row.id !== job.id)];
    });
    setCandidateId(job.id);
  }

  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !prompt.trim() || hasActive || revisionPending) return;
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
    if (!candidate || busy || revisionPending) return;
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

  async function regenerate() {
    if (busy) return;
    let request = pendingRevision.current;
    if (!request) {
      if (!candidate || active(candidate) || hasActive || !frame || !repairReason.trim()) return;
      request = {
        parentId: candidate.id, id: randomUuid(), expected_version: candidate.version,
        frame_names: [frame.name], reason: repairReason.trim(),
      };
      pendingRevision.current = request;
      sessionStorage.setItem(revisionKey, JSON.stringify(request));
      setRevisionPending(true);
    }
    setBusy(true);
    setError("");
    const { parentId, ...body } = request;
    try {
      update(await api<PetJob>(`${path}/${encodeURIComponent(parentId)}/revisions`, body));
      pendingRevision.current = null;
      sessionStorage.removeItem(revisionKey);
      if (alive.current) {
        setRevisionPending(false);
        setRepairReason("");
        setPreview("idle");
      }
    } catch (e) {
      // A poll can confirm the revision before a lost POST response settles.
      if (pendingRevision.current?.id !== request.id) return;
      if (e instanceof ApiError && [400, 401, 403, 404, 409, 422].includes(e.status)) {
        pendingRevision.current = null;
        sessionStorage.removeItem(revisionKey);
        if (alive.current) setRevisionPending(false);
        await refresh();
      }
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
          <button className="primary" disabled={busy || loading || hasActive || revisionPending || !prompt.trim()}>
            {busy ? "Saving…" : "Create pet"}
          </button>
        </div>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {revisionPending && <div className="banner" role="status">
        Checking whether the frame repair was saved.
        <button disabled={busy} onClick={() => void regenerate()}>Retry confirmation</button>
      </div>}
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
            if (!revisionPending) setRepairReason("");
          }}>
            {jobs.map((job) => <option key={job.id} value={job.id}>
              {job.prompt.length > 70 ? `${job.prompt.slice(0, 70)}…` : job.prompt} · {selected?.id === job.id ? "In use" : statuses[job.status]}
            </option>)}
          </select>
          {candidate && <>
            <p className="pet-job-prompt">{candidate.prompt}</p>
            {candidate.parent_pet_id && <p className="field-hint">Frame revision · other saved artwork retained.</p>}
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
            {!active(candidate) && frame && <div className="pet-repair">
              <label htmlFor="pet-repair-frame">Frame to regenerate</label>
              <select id="pet-repair-frame" value={frame.name} disabled={busy || revisionPending}
                onChange={(event) => setRepairFrame(event.target.value)}>
                {repairFrames.map((step) => <option key={step.name} value={step.name}>{frameLabel(step.name)}</option>)}
              </select>
              <img className="pet-repair-image" width={192} height={208}
                src={`/api${agentPath(agentKey)}/assets/${encodeURIComponent(frame.output_asset_id)}/content`}
                alt={`${frameLabel(frame.name)} saved artwork`} />
              {["idle/0", "idle/5"].includes(frame.name) && <p className="field-hint">
                This neutral frame will reuse the original reference to keep its standing pose consistent.
              </p>}
              <label htmlFor="pet-repair-reason">What needs fixing?</label>
              <textarea id="pet-repair-reason" rows={2} maxLength={1000} value={repairReason}
                disabled={busy || revisionPending} placeholder="The feet float above the floor during the blink."
                onChange={(event) => setRepairReason(event.target.value)} />
              <p className="field-hint">Creates a revised candidate with the other saved frames. Preview it before choosing Use pet.</p>
              <div className="form-actions">
                <button disabled={busy || hasActive || revisionPending || !repairReason.trim()}
                  onClick={() => void regenerate()}>Regenerate frame</button>
              </div>
            </div>}
            <div className="form-actions">
              {active(candidate) && <button disabled={busy || revisionPending || candidate.status === "cancelling"} onClick={() => void changeJob("cancel")}>
                {candidate.status === "cancelling" ? "Cancelling…" : "Cancel creation"}
              </button>}
              {["failed", "interrupted", "cancelled"].includes(candidate.status) && <button disabled={busy || hasActive || revisionPending} onClick={() => void changeJob("resume")}>Resume creation</button>}
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
