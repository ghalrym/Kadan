import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Egg, FileArchive, FolderOpen, Plus, Sparkles, Stethoscope, Upload, Wrench, X } from "lucide-react";
import { agentPath, api, ApiError } from "./api";
import { agentPet } from "./Pet";
import { petSheet, usePetImage } from "./PetContext";
import PetStage, { animationLabel } from "./PetStage";
import { agentHue, cellPosition, COLUMNS, sheetRows, type PetSheet } from "./pet";
import { atlasVersion, resolvePetPackage, type PetPackage, type PickedFile } from "./petImport";
import type { PetJob } from "./types";
import { randomUuid } from "./uuid";

const statuses: Record<PetJob["status"], string> = {
  queued: "Queued", running: "Creating", cancelling: "Cancelling", cancelled: "Cancelled",
  interrupted: "Interrupted", failed: "Failed", ready: "Ready to preview",
};
const active = (job: PetJob) => ["queued", "running", "cancelling"].includes(job.status);
const statusLabel = (job: PetJob) => job.doctor && job.status === "queued" ? "Doctor queued"
  : job.doctor && job.status === "running" ? "Doctor working" : statuses[job.status];
const NEW = "new";
const HATCHLING = "hatchling";
const ideas = [
  "A small copper fox, illustrated in watercolor",
  "A round moss golem with glowing eyes, soft clay style",
  "A tiny teal robot with a cracked antenna, crisp pixel art",
  "A sleepy ink-drop octopus, flat vector style",
];

interface CreationRequest { id: string; prompt: string }
interface RevisionRequest {
  action: "revisions" | "doctor";
  parentId: string;
  id: string;
  expected_version: number;
  frame_names: string[];
  reason: string;
}
interface ImportPreview { url: string; version: 1 | 2 | null; size: string }

function readPendingRevision(key: string): RevisionRequest | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? "null");
    return value && typeof value.parentId === "string" && typeof value.id === "string"
      && Number.isInteger(value.expected_version) && typeof value.reason === "string"
      && Array.isArray(value.frame_names)
      && ((value.action === "revisions" && value.frame_names.length === 1 && typeof value.frame_names[0] === "string")
        || (value.action === "doctor" && value.frame_names.length === 0)) ? value : null;
  } catch { return null; }
}

function readPendingCreation(key: string): CreationRequest | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? "null");
    return value && typeof value.id === "string" && typeof value.prompt === "string" ? value : null;
  } catch { return null; }
}

function frameLabel(name: string): string {
  const [animation, index] = name.split("/");
  const label = animation.startsWith("look-")
    ? `Look ${Number(animation.slice(5)) * 180 + Number(index) * 22.5}°`
    : animationLabel(animation);
  return `${label} · frame ${Number(index) + 1}`;
}

/** Generated prompts are one description; imports store the name then the description. */
function petTitle(job: PetJob) {
  const [first, ...rest] = job.prompt.split("\n");
  return job.origin === "imported"
    ? { title: first, detail: rest.join("\n").trim() }
    : first.length > 60 || rest.length
      ? { title: `${first.slice(0, 60).trimEnd()}…`, detail: job.prompt }
      : { title: first, detail: "" };
}

/** Collect files from a drop, walking dropped folders like a folder picker would. */
async function droppedFiles(transfer: DataTransfer): Promise<PickedFile[]> {
  // Entries must be taken synchronously, before the drop event finishes.
  const entries = Array.from(transfer.items, (item) => item.webkitGetAsEntry?.() ?? null);
  if (!entries.some(Boolean)) return Array.from(transfer.files, (file) => ({ file, path: file.name }));
  const picked: PickedFile[] = [];
  const walk = async (entry: FileSystemEntry, depth: number): Promise<void> => {
    if (picked.length >= 200) return;
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
      picked.push({ file, path: entry.fullPath.replace(/^\//, "") });
    } else if (entry.isDirectory && depth < 4) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await walk(child, depth + 1);
      }
    }
  };
  for (const entry of entries) if (entry) await walk(entry, 0);
  return picked;
}

function SheetThumb({ sheet }: { sheet: PetSheet }) {
  return <span className={`pet-thumb-art ${sheet.pixelated ? "pixelated" : ""}`} style={{
    backgroundImage: `url("${sheet.url}")`,
    backgroundSize: `${COLUMNS * 100}% ${sheetRows(sheet.version) * 100}%`,
    backgroundPosition: cellPosition(0, 0, sheet.version),
  }} />;
}

function PetThumb({ job, agentKey }: { job: PetJob; agentKey: string }) {
  const sheet = petSheet(job);
  if (sheet) return <SheetThumb sheet={sheet} />;
  const base = job.steps.find((step) => step.name === "base");
  if (base) return <img className="pet-thumb-art" alt=""
    src={`/api${agentPath(agentKey)}/assets/${encodeURIComponent(base.output_asset_id)}/content`} />;
  return <Sparkles size={18} aria-hidden="true" />;
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
  // A job id, NEW for the creation panel, or HATCHLING for the default pet.
  const [view, setView] = useState(initialRevision?.parentId ?? (initialCreation ? NEW : selected?.id ?? ""));
  const [newTab, setNewTab] = useState<"describe" | "import">("describe");
  const [prompt, setPrompt] = useState(initialCreation?.prompt ?? "");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [pollError, setPollError] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [importPackage, setImportPackage] = useState<PetPackage | null>(null);
  const [importPreview, setImportPreview] = useState<ImportPreview | null>(null);
  const [importName, setImportName] = useState("");
  const [dragging, setDragging] = useState(false);
  const pendingImport = useRef<{ id: string; signature: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const candidate = jobs.find((job) => job.id === view) ?? null;
  const repairFrames = candidate?.steps.filter((step) => step.name !== "base" && step.method !== "canonical") ?? [];
  const frame = repairFrames.find((step) => step.name === repairFrame) ?? repairFrames[0];
  const sheet = useMemo(() => petSheet(candidate), [candidate]);
  const image = usePetImage(sheet);
  const hatchling = useMemo(() => agentPet(agentKey), [agentKey]);
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
        setView(completedRevision.id);
        setError("");
      } else {
        setView((current) => current || (rows[0]?.id ?? NEW));
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

  useEffect(() => () => { if (importPreview) URL.revokeObjectURL(importPreview.url); }, [importPreview]);
  useEffect(() => {
    folderInput.current?.setAttribute("webkitdirectory", "");
  }, [view, newTab]);

  function show(next: string) {
    setView(next);
    setError("");
    if (!revisionPending) setRepairReason("");
  }

  function update(job: PetJob) {
    if (!alive.current) return;
    ++refreshRevision.current;
    setJobs((rows) => {
      const previous = rows.find((row) => row.id === job.id);
      return [previous && previous.version > job.version ? previous : job, ...rows.filter((row) => row.id !== job.id)];
    });
    setView(job.id);
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

  async function choosePackage(picked: PickedFile[]) {
    setError("");
    setImportPreview(null);
    if (!picked.length) return;
    const result = await resolvePetPackage(picked, (file) => file.text());
    if (!alive.current) return;
    setImportPackage(result);
    if (result.kind === "error") return;
    const fallback = result.kind === "files" ? result.sheet.name.replace(/\.(webp|png)$/i, "") : result.zip.name.replace(/\.zip$/i, "");
    setImportName(result.kind === "files" && result.info?.displayName
      ? result.info.displayName
      : /^spritesheet$/i.test(fallback) ? "" : fallback);
    if (result.kind !== "files") return;
    const url = URL.createObjectURL(result.sheet);
    const probe = new Image();
    probe.onload = () => {
      if (!alive.current) return URL.revokeObjectURL(url);
      setImportPreview({ url, version: atlasVersion(probe.naturalWidth, probe.naturalHeight), size: `${probe.naturalWidth}x${probe.naturalHeight}` });
    };
    probe.onerror = () => {
      URL.revokeObjectURL(url);
      if (alive.current) setImportPackage({ kind: "error", message: "The spritesheet could not be read as an image." });
    };
    probe.src = url;
  }

  function clearImport() {
    setImportPackage(null);
    setImportPreview(null);
    setImportName("");
    setError("");
    if (fileInput.current) fileInput.current.value = "";
    if (folderInput.current) folderInput.current.value = "";
  }

  async function importPet(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !importPackage || importPackage.kind === "error") return;
    if (importPackage.kind === "files" && importPreview?.version === null) return;
    // Reuse the id when retrying the same upload so an unconfirmed import is not duplicated.
    const signature = JSON.stringify([importName.trim(), importPackage.files.map((file) => [file.name, file.size, file.lastModified])]);
    if (pendingImport.current?.signature !== signature) pendingImport.current = { id: randomUuid(), signature };
    const form = new FormData();
    form.set("id", pendingImport.current.id);
    form.set("name", importName.trim());
    for (const file of importPackage.files) form.append("files", file, file.name);
    setBusy(true);
    setError("");
    try {
      const job = await api<PetJob>(`${path}/imports`, form);
      pendingImport.current = null;
      if (!alive.current) return;
      clearImport();
      update(job);
    } catch (e) {
      if (e instanceof ApiError && e.status < 500) pendingImport.current = null;
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

  async function regenerate(action: "revisions" | "doctor" = "revisions") {
    if (busy) return;
    let request = pendingRevision.current;
    if (!request) {
      if (!candidate || active(candidate) || hasActive) return;
      if (action === "revisions" && (!frame || !repairReason.trim())) return;
      if (action === "doctor" && (candidate.status !== "ready" || candidate.origin !== "generated")) return;
      request = {
        action,
        parentId: candidate.id, id: randomUuid(), expected_version: candidate.version,
        frame_names: action === "revisions" ? [frame!.name] : [],
        reason: action === "revisions" ? repairReason.trim() : "",
      };
      pendingRevision.current = request;
      sessionStorage.setItem(revisionKey, JSON.stringify(request));
      setRevisionPending(true);
    }
    setBusy(true);
    setError("");
    const { parentId, action: endpoint, frame_names, reason, ...identity } = request;
    const body = endpoint === "doctor" ? identity : { ...identity, frame_names, reason };
    try {
      update(await api<PetJob>(`${path}/${encodeURIComponent(parentId)}/${endpoint}`, body));
      pendingRevision.current = null;
      sessionStorage.removeItem(revisionKey);
      if (alive.current) {
        setRevisionPending(false);
        setRepairReason("");
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

  const importSheet: PetSheet | null = importPreview?.version
    ? { url: importPreview.url, version: importPreview.version }
    : null;
  const importReady = !!importPackage && importPackage.kind !== "error"
    && (importPackage.kind === "zip" || !!importSheet);

  const showsNewPanel = view === NEW || (view !== HATCHLING && !candidate && !loading);
  // The import form shows its own error beside the button, where a failed upload is noticed.
  const importErrorInline = showsNewPanel && newTab === "import";
  const newPanel = (
    <section className="pet-panel" aria-labelledby="pet-new-title">
      <h3 id="pet-new-title">New pet</h3>
      <div className="tabs" role="tablist" aria-label="How to add a pet">
        <button type="button" role="tab" aria-selected={newTab === "describe"} className={newTab === "describe" ? "active" : ""}
          onClick={() => { setNewTab("describe"); setError(""); }}><Sparkles size={14} /> Describe</button>
        <button type="button" role="tab" aria-selected={newTab === "import"} className={newTab === "import" ? "active" : ""}
          onClick={() => { setNewTab("import"); setError(""); }}><Upload size={14} /> Import</button>
      </div>
      {newTab === "describe" ? <form className="pet-form" onSubmit={create}>
        <label htmlFor="pet-prompt">Character and art style</label>
        <textarea id="pet-prompt" value={prompt} rows={4} maxLength={4000} required
          placeholder="Who is this pet, and how should it be drawn?"
          onChange={(event) => setPrompt(event.target.value)} />
        <div className="pet-ideas" aria-label="Ideas">
          {ideas.map((idea) => <button key={idea} type="button" className="chip" onClick={() => setPrompt(idea)}>{idea}</button>)}
        </div>
        <p className="field-hint">
          Kadan draws 73 animation and look frames in the background, which takes a while. You can close this window; your current pet stays in use until you choose the new one.
        </p>
        {hasActive && <p className="field-hint">Another pet is being created. Wait for it to finish or cancel it first.</p>}
        <div className="form-actions">
          <button className="primary" disabled={busy || loading || hasActive || revisionPending || !prompt.trim()}>
            {busy ? "Saving…" : "Create pet"}
          </button>
        </div>
      </form> : <form className="pet-form" onSubmit={importPet}>
        <p className="quiet">
          Bring over a pet made for ChatGPT or Codex. Use its folder (with <code>pet.json</code> and the spritesheet), a zip of that folder, or just the spritesheet.
        </p>
        <div className={`pet-drop ${dragging ? "dragging" : ""}`} hidden={!!importPackage && importPackage.kind !== "error"}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const files = droppedFiles(event.dataTransfer);
            void files.then(choosePackage, () => setError("Those files could not be read."));
          }}>
          <Upload size={22} aria-hidden="true" />
          <p>Drop a pet folder, zip or spritesheet here</p>
          <div className="pet-drop-actions">
            <button type="button" onClick={() => fileInput.current?.click()}><FileArchive size={16} /> Choose files</button>
            <button type="button" onClick={() => folderInput.current?.click()}><FolderOpen size={16} /> Choose folder</button>
          </div>
          <input ref={fileInput} type="file" multiple hidden
            accept=".zip,.json,.webp,.png,application/zip,application/json,image/webp,image/png"
            onChange={(event) => void choosePackage(Array.from(event.target.files ?? [], (file) => ({ file, path: file.name })))} />
          <input ref={folderInput} type="file" multiple hidden
            onChange={(event) => void choosePackage(Array.from(event.target.files ?? [], (file) => ({ file, path: file.webkitRelativePath || file.name })))} />
        </div>
        {importPackage?.kind === "error" && <p className="error" role="alert">{importPackage.message}</p>}
        {importPackage && importPackage.kind !== "error" && <>
          <div className="pet-import-files">
            <span>{importPackage.files.map((file) => file.name).join(" + ")}</span>
            <button type="button" className="icon" aria-label="Choose different files" title="Choose different files" onClick={clearImport}><X size={16} /></button>
          </div>
          {importPackage.kind === "files" && importPackage.info?.description && <p className="pet-description">{importPackage.info.description}</p>}
          <label htmlFor="pet-import-name">Name</label>
          <input id="pet-import-name" value={importName} maxLength={200}
            placeholder={importPackage.kind === "zip" ? "Uses the name in pet.json" : "Imported pet"}
            onChange={(event) => setImportName(event.target.value)} />
          {importPackage.kind === "zip"
            ? <p className="field-hint">The zip is checked when you import it. You can preview the pet before using it.</p>
            : importPreview && !importSheet
              ? <p className="error" role="alert">This spritesheet is {importPreview.size}. Codex pets are 1536x1872 (v1) or 1536x2288 (v2).</p>
              : importSheet && <PetStage key={importSheet.url} sheet={importSheet} />}
        </>}
        {error && <p className="error" role="alert">{error}</p>}
        <div className="form-actions">
          <button className="primary" disabled={busy || loading || !importReady}>{busy ? "Importing…" : "Import pet"}</button>
        </div>
      </form>}
    </section>
  );

  const hatchlingPanel = (
    <section className="pet-panel" aria-labelledby="pet-hatchling-title">
      <div className="pet-panel-head">
        <h3 id="pet-hatchling-title">Hatchling</h3>
        <span className={`pet-badge ${selected ? "" : "in-use"}`}>{selected ? "Default" : "In use"}</span>
      </div>
      <p className="pet-description">The pixel pet every agent starts with, coloured from {agentName}’s key.</p>
      <PetStage sheet={hatchling} />
      <div className="form-actions">
        <button className="primary" disabled={busy || !selected} onClick={() => void select(null)}>
          {selected ? "Use hatchling" : "In use"}
        </button>
      </div>
    </section>
  );

  const candidatePanel = candidate && (() => {
    const { title, detail } = petTitle(candidate);
    const inUse = selected?.id === candidate.id;
    const checked = candidate.steps.filter((step) => step.normalization);
    return (
      <section className="pet-panel" aria-labelledby="pet-candidate-title">
        <div className="pet-panel-head">
          <h3 id="pet-candidate-title">{title}</h3>
          <span className={`pet-badge ${inUse ? "in-use" : candidate.status}`} role="status">
            {inUse ? "In use" : statusLabel(candidate)}
          </span>
        </div>
        {detail && <p className="pet-description">{detail}</p>}
        <p className="field-hint">
          {candidate.origin === "imported"
            ? `Imported ${candidate.manifest?.sprite_version === 1 ? "v1 sheet · no look directions" : "v2 sheet"}`
            : candidate.doctor ? "Doctor · body size consistency"
            : candidate.parent_pet_id ? "Frame revision · other saved artwork retained" : "Generated"}
        </p>
        {active(candidate) && <div className="pet-progress">
          <progress value={candidate.completed_steps} max={candidate.total_steps} aria-label={candidate.doctor ? "Doctor progress" : "Pet creation progress"} />
          <div className="pet-progress-label">
            <span>{candidate.stage.replaceAll("_", " ")}</span>
            <span>{candidate.completed_steps} / {candidate.total_steps}</span>
          </div>
          <p className="field-hint">{candidate.doctor ? "Doctor continues" : "Creation continues"} when you close this window. Your current pet stays in use.</p>
        </div>}
        {candidate.error && <p className="error">{candidate.error}</p>}
        {candidate.status === "interrupted" && <p className="field-hint">Resume to continue from the saved artwork.</p>}
        {candidate.doctor && checked.length > 0 && <details className="pet-repair">
          <summary>Doctor report · {checked.filter((step) => step.normalization?.outcome === "corrected").length} corrected
            {" · "}{checked.filter((step) => step.normalization?.outcome === "consistent").length} consistent
            {" · "}{checked.filter((step) => step.normalization?.outcome === "skipped").length} skipped</summary>
          <p className="field-hint">Frames without reliable matches stay unchanged. Preview the result before choosing Use pet.</p>
          <ul className="pet-doctor-report">{checked.map((step) => <li key={step.name}>
            <strong>{frameLabel(step.name)}</strong>{" · "}
            {step.normalization!.outcome === "corrected" ? `Scaled to ${Math.round(step.normalization!.scale * 100)}%. ` : ""}
            {step.normalization!.reason}
          </li>)}</ul>
        </details>}
        {!active(candidate) && candidate.status !== "ready" && hasActive && <p className="field-hint">
          Another pet is being created. Resume this one after it finishes.
        </p>}
        {sheet && <PetStage key={candidate.id} sheet={sheet} ready={image.ready} failed={image.failed} />}
        <div className="form-actions">
          {active(candidate) && <button disabled={busy || revisionPending || candidate.status === "cancelling"} onClick={() => void changeJob("cancel")}>
            {candidate.status === "cancelling" ? "Cancelling…" : candidate.doctor ? "Cancel Doctor" : "Cancel creation"}
          </button>}
          {["failed", "interrupted", "cancelled"].includes(candidate.status) && <button disabled={busy || hasActive || revisionPending} onClick={() => void changeJob("resume")}>{candidate.doctor ? "Resume Doctor" : "Resume creation"}</button>}
          {sheet && <button className="primary" disabled={busy || !image.ready || inUse}
            onClick={() => void select(candidate.id)}>{inUse ? "In use" : "Use pet"}</button>}
          {sheet && <button disabled={busy || loading || hasActive || revisionPending || candidate.origin !== "generated"}
            title={candidate.origin === "imported" ? "Doctor needs the original character reference, which imported sheets do not include."
              : "Check character size and save a corrected candidate. Your current pet stays in use."}
            onClick={() => void regenerate("doctor")}><Stethoscope size={14} aria-hidden="true" /> Doctor</button>}
        </div>
        {!active(candidate) && frame && <details className="pet-repair" open={(revisionPending && pendingRevision.current?.action === "revisions") || undefined}>
          <summary><Wrench size={14} aria-hidden="true" /> Repair a frame</summary>
          <div className="pet-repair-body">
            <img className="pet-repair-image" width={144} height={156}
              src={`/api${agentPath(agentKey)}/assets/${encodeURIComponent(frame.output_asset_id)}/content`}
              alt={`${frameLabel(frame.name)} saved artwork`} />
            <div className="pet-form">
              <label htmlFor="pet-repair-frame">Frame</label>
              <select id="pet-repair-frame" value={frame.name} disabled={busy || revisionPending}
                onChange={(event) => setRepairFrame(event.target.value)}>
                {repairFrames.map((step) => <option key={step.name} value={step.name}>{frameLabel(step.name)}</option>)}
              </select>
              {["idle/0", "idle/5"].includes(frame.name) && <p className="field-hint">
                This neutral frame will reuse the original reference to keep its standing pose consistent.
              </p>}
              <label htmlFor="pet-repair-reason">What needs fixing?</label>
              <textarea id="pet-repair-reason" rows={2} maxLength={1000} value={repairReason}
                disabled={busy || revisionPending} placeholder="The feet float above the floor during the blink."
                onChange={(event) => setRepairReason(event.target.value)} />
            </div>
          </div>
          <p className="field-hint">Creates a revised candidate that keeps every other saved frame. Preview it before choosing Use pet.</p>
          <div className="form-actions">
            <button disabled={busy || hasActive || revisionPending || !repairReason.trim()}
              onClick={() => void regenerate()}>Regenerate frame</button>
          </div>
        </details>}
      </section>
    );
  })();

  return (
    <dialog ref={dialog} className="pet-editor" aria-labelledby="pet-editor-title"
      style={{ "--agent-h": agentHue(agentKey) } as React.CSSProperties}
      onCancel={(event) => { event.preventDefault(); onClose(); }}>
      <header className="pet-editor-head">
        <h2 id="pet-editor-title">{agentName}’s pet</h2>
        <button className="icon" aria-label="Close pet editor" onClick={onClose}><X size={18} /></button>
      </header>
      <div className="pet-editor-body">
        <nav className="pet-gallery" aria-label="Pets">
          <button className={`pet-card new ${view === NEW ? "current" : ""}`} aria-current={view === NEW || undefined}
            onClick={() => show(NEW)}>
            <span className="pet-thumb"><Plus size={18} aria-hidden="true" /></span>
            <span className="pet-card-text"><span className="pet-card-title">New pet</span>
              <span className="pet-card-meta">Describe or import</span></span>
          </button>
          <button className={`pet-card ${view === HATCHLING ? "current" : ""}`} aria-current={view === HATCHLING || undefined}
            onClick={() => show(HATCHLING)}>
            <span className="pet-thumb"><SheetThumb sheet={hatchling} /></span>
            <span className="pet-card-text"><span className="pet-card-title"><Egg size={12} aria-hidden="true" /> Hatchling</span>
              <span className={`pet-card-meta ${selected ? "" : "in-use"}`}>{selected ? "Default" : "In use"}</span></span>
          </button>
          {loading && <p className="quiet" role="status">Loading pets…</p>}
          {jobs.map((job) => {
            const inUse = selected?.id === job.id;
            return <button key={job.id} className={`pet-card ${view === job.id ? "current" : ""}`}
              aria-current={view === job.id || undefined} onClick={() => show(job.id)}>
              <span className="pet-thumb"><PetThumb job={job} agentKey={agentKey} />
                {active(job) && <span className="pet-thumb-progress" style={{ width: `${Math.max(6, (job.completed_steps / job.total_steps) * 100)}%` }} />}
              </span>
              <span className="pet-card-text">
                <span className="pet-card-title">{petTitle(job).title}</span>
                <span className={`pet-card-meta ${inUse ? "in-use" : job.status}`}>
                  {inUse ? "In use" : active(job) ? `${statusLabel(job)} · ${Math.round((job.completed_steps / job.total_steps) * 100)}%`
                    : job.status === "ready" ? job.doctor ? "Doctor · ready" : job.origin === "imported" ? "Imported" : "Ready" : statuses[job.status]}
                </span>
              </span>
            </button>;
          })}
        </nav>
        <div className="pet-main">
          {error && !importErrorInline && <p className="error" role="alert">{error}</p>}
          {revisionPending && <div className="banner" role="status">
            Checking whether the {pendingRevision.current?.action === "doctor" ? "Doctor request" : "frame repair"} was saved.
            <button disabled={busy} onClick={() => void regenerate()}>Retry confirmation</button>
          </div>}
          {selectionError && <div className="banner" role="alert">Could not confirm the selected pet: {selectionError}
            <button disabled={busy} onClick={() => void refreshSelection()}>Retry</button>
          </div>}
          {pollError && <div className="banner" role="alert">Updates unavailable: {pollError}
            <button onClick={() => void refresh()}>Retry</button>
          </div>}
          {view === NEW ? newPanel
            : view === HATCHLING ? hatchlingPanel
            : candidatePanel ?? (loading ? <p className="quiet" role="status">Loading pets…</p> : newPanel)}
        </div>
      </div>
    </dialog>
  );
}
