import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlarmClock,
  AlertCircle,
  Archive,
  ArchiveRestore,
  ArrowUp,
  Brain,
  Check,
  ChevronRight,
  FileCode2,
  Folder,
  FolderPlus,
  Paperclip,
  Square,
  Terminal,
  Trash2,
  X,
} from "lucide-react";
import ConfirmDialog from "./ConfirmDialog";
import FolderPickerDialog from "./FolderPickerDialog";
import Markdown from "./Markdown";
import ModelPicker from "./ModelPicker";
import ProjectPicker from "./ProjectPicker";
import { isDesktopApp, pickDesktopFolder } from "./desktop";
import {
  createWorkProject,
  createWorkThread,
  decodeModelSelection,
  deleteWorkThread,
  encodeModelSelection,
  formatSnoozeLabel,
  getWorkThreadDetail,
  interruptWorkThread,
  isThreadActive,
  listWorkHarnesses,
  listWorkProjects,
  organizeWorkThread,
  pendingRequestsFor,
  respondWorkPendingRequest,
  sendWorkMessage,
  updateWorkThread,
  workStatusLabel,
  type WorkActivityItem,
  type WorkHarness,
  type WorkPendingRequest,
  type WorkProject,
  type WorkThreadDetail,
} from "./work";

const LAST_PROJECT_KEY = "kadan:work:lastProjectId";
const LAST_MODEL_KEY = "kadan:work:lastModel";
const LAST_OPTIONS_KEY = "kadan:work:lastModelOptions";

interface Props {
  threadId: string | null;
  onThreadCreated?: (threadId: string) => void;
  onOpenSettings?: () => void;
  onChanged: () => void;
  onDeleted: () => void;
}

function readStorage(key: string): string {
  try {
    return typeof localStorage !== "undefined" ? (localStorage.getItem(key) ?? "") : "";
  } catch {
    return "";
  }
}

function writeStorage(key: string, value: string): void {
  try {
    if (typeof localStorage !== "undefined" && value) {
      localStorage.setItem(key, value);
    }
  } catch {
    // Ignore storage errors.
  }
}

function readOptions(): Record<string, unknown> {
  try {
    const parsed = JSON.parse(readStorage(LAST_OPTIONS_KEY) || "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function tomorrowMorningIso(): string {
  const next = new Date();
  next.setDate(next.getDate() + 1);
  next.setHours(9, 0, 0, 0);
  return next.toISOString();
}

function pickInitialModel(harnesses: WorkHarness[]): string {
  const saved = readStorage(LAST_MODEL_KEY);
  if (saved) {
    const { harness, model } = decodeModelSelection(saved);
    const match = harnesses.find((h) => h.id === harness && h.auth.status === "ready");
    if (match) {
      const validModel =
        model && match.models.some((m) => m.id === model)
          ? model
          : (match.default_model ?? match.models[0]?.id ?? null);
      return encodeModelSelection(match.id, validModel);
    }
  }
  const preferred = harnesses.find((h) => h.auth.status === "ready") ?? harnesses[0];
  if (!preferred) return "";
  return encodeModelSelection(
    preferred.id,
    preferred.default_model ?? preferred.models[0]?.id ?? null,
  );
}

function PendingCard({
  request,
  busy,
  onRespond,
}: {
  request: WorkPendingRequest;
  busy: boolean;
  onRespond: (requestId: string, response: Record<string, unknown>) => Promise<void>;
}) {
  const [answer, setAnswer] = useState("");
  const isApproval = request.kind === "approval";

  return (
    <section
      className={`work-pending-card ${isApproval ? "is-approval" : "is-question"}`}
      aria-live="polite"
    >
      <div className="work-pending-head">
        <strong>{request.title || (isApproval ? "Approval needed" : "Question")}</strong>
        {request.tool_name && <span className="work-pill">{request.tool_name}</span>}
      </div>
      <p className="work-pending-prompt">{request.prompt}</p>
      {isApproval ? (
        <div className="work-pending-actions">
          <button
            className="primary"
            disabled={busy}
            onClick={() => void onRespond(request.id, { decision: "approve" })}
          >
            <Check size={15} />
            Approve
          </button>
          <button
            className="button"
            disabled={busy}
            onClick={() => void onRespond(request.id, { decision: "approve_session" })}
          >
            Approve for session
          </button>
          <button
            className="button"
            disabled={busy}
            onClick={() => void onRespond(request.id, { decision: "deny" })}
          >
            <X size={15} />
            Decline
          </button>
        </div>
      ) : (
        <div className="work-pending-question-body">
          {request.options.length > 0 && (
            <div className="work-pending-options">
              {request.options.map((opt, idx) => {
                const label = String(opt.label ?? opt.name ?? opt.value ?? `Option ${idx + 1}`);
                const value = String(opt.value ?? opt.label ?? label);
                return (
                  <button
                    key={`${request.id}-opt-${idx}`}
                    className="button"
                    disabled={busy}
                    onClick={() => void onRespond(request.id, { answer: value })}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          )}
          <form
            className="work-pending-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!answer.trim()) return;
              void onRespond(request.id, { answer: answer.trim() }).then(() => setAnswer(""));
            }}
          >
            <input
              type="text"
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              placeholder="Type your answer…"
              disabled={busy}
            />
            <button type="submit" className="primary" disabled={busy || !answer.trim()}>
              Send answer
            </button>
          </form>
        </div>
      )}
    </section>
  );
}

function ActivityRow({ item }: { item: WorkActivityItem }) {
  if (item.kind === "user_message") {
    const steered = item.data?.mode === "steer";
    const imageCount = Number(item.data?.image_count ?? 0);
    return (
      <article className="work-item work-item-user">
        <div className="work-user-bubble">{item.text}</div>
        {(steered || imageCount > 0) && (
          <div className="work-item-meta">
            {steered && <span>Sent as a live update</span>}
            {imageCount > 0 && (
              <span className="work-pill">
                {imageCount} {imageCount === 1 ? "image" : "images"}
              </span>
            )}
          </div>
        )}
      </article>
    );
  }
  if (item.kind === "assistant_message") {
    return (
      <article className="work-item work-item-assistant">
        <Markdown>{item.text}</Markdown>
      </article>
    );
  }
  if (item.kind === "reasoning") {
    return (
      <details className="work-item work-item-step work-item-reasoning">
        <summary>
          <Brain size={13} aria-hidden="true" />
          <span className="work-step-label">Thinking</span>
          <ChevronRight size={13} className="work-step-chevron" aria-hidden="true" />
        </summary>
        <p>{item.text}</p>
      </details>
    );
  }
  if (item.kind === "tool_call" || item.kind === "tool_result") {
    const toolName = String(item.data?.tool_name ?? "Action");
    const isError = Boolean(item.data?.is_error);
    return (
      <details className={`work-item work-item-step work-item-tool ${isError ? "is-error" : ""}`}>
        <summary>
          <Terminal size={13} aria-hidden="true" />
          <span className="work-step-label">
            {item.kind === "tool_call" ? "Running" : "Finished"} <code>{toolName}</code>
          </span>
          {isError && <span className="work-pill error">Failed</span>}
          {item.text && <ChevronRight size={13} className="work-step-chevron" aria-hidden="true" />}
        </summary>
        {item.text && <pre className="work-code-block">{item.text}</pre>}
      </details>
    );
  }
  if (item.kind === "file_change") {
    const filePath = String(item.data?.file_path ?? "File updated");
    const changeType = String(item.data?.change_type ?? "modified");
    return (
      <div className="work-item work-item-step work-item-file">
        <FileCode2 size={13} aria-hidden="true" />
        <span className="work-file-path">{filePath}</span>
        <span className={`work-pill change-${changeType}`}>{changeType}</span>
      </div>
    );
  }
  if (item.kind === "error") {
    return (
      <div className="work-item work-item-error" role="alert">
        <AlertCircle size={15} />
        <span>{item.text}</span>
      </div>
    );
  }
  if (item.kind === "pending_request") {
    return (
      <div className="work-item work-item-status">
        <span>Paused: {item.text}</span>
      </div>
    );
  }
  return (
    <div className="work-item work-item-status">
      <span>{item.text}</span>
    </div>
  );
}

export default function WorkThreadView({
  threadId,
  onThreadCreated,
  onOpenSettings,
  onChanged,
  onDeleted,
}: Props) {
  const isNew = threadId === null;
  const [detail, setDetail] = useState<WorkThreadDetail | null>(null);
  const [harnesses, setHarnesses] = useState<WorkHarness[]>([]);
  const [projects, setProjects] = useState<WorkProject[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState("");
  const [snoozeOpen, setSnoozeOpen] = useState(false);
  const [customSnooze, setCustomSnooze] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [attachedImages, setAttachedImages] = useState<{ name: string; dataUrl: string }[]>([]);
  const [folderDialogOpen, setFolderDialogOpen] = useState(false);
  const [selectedOptions, setSelectedOptions] = useState<Record<string, unknown>>(() => readOptions());
  const feedEndRef = useRef<HTMLDivElement>(null);

  const load = useCallback(
    async (showSpinner = false) => {
      if (!threadId) return;
      if (showSpinner) setLoading(true);
      try {
        const next = await getWorkThreadDetail(threadId);
        setDetail(next);
        setError("");
      } catch (err) {
        setError((err as Error).message);
      } finally {
        if (showSpinner) setLoading(false);
      }
    },
    [threadId],
  );

  useEffect(() => {
    if (isNew) {
      setDetail(null);
      setLoading(true);
      void Promise.all([listWorkProjects(), listWorkHarnesses()])
        .then(([projList, harnessList]) => {
          setProjects(projList);
          setHarnesses(harnessList);
          const savedProject = readStorage(LAST_PROJECT_KEY);
          const initialProject =
            (savedProject && projList.some((p) => p.id === savedProject)
              ? savedProject
              : projList[0]?.id) ?? "";
          setSelectedProjectId((prev) =>
            prev && projList.some((p) => p.id === prev) ? prev : initialProject,
          );
          setSelectedModelKey((prev) => prev || pickInitialModel(harnessList));
          setError("");
        })
        .catch((err) => {
          setError((err as Error).message);
        })
        .finally(() => setLoading(false));
      return;
    }

    void load(true);
    void listWorkHarnesses()
      .then(setHarnesses)
      .catch(() => {});
  }, [isNew, load]);

  const active = detail ? isThreadActive(detail.thread.run_status) : false;

  useEffect(() => {
    if (isNew) return;
    const intervalMs = active ? 1200 : 6000;
    const timer = setInterval(() => void load(false), intervalMs);
    return () => clearInterval(timer);
  }, [isNew, active, load]);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [detail?.activity.length]);

  function handleSelectImages(files: FileList | null) {
    if (!files || files.length === 0) return;
    const allowed = Array.from(files).filter(
      (f) => f.type === "image/png" || f.type === "image/jpeg",
    );
    if (allowed.length < files.length) {
      setError("Only PNG and JPEG screenshots can be attached.");
    }
    for (const file of allowed.slice(0, Math.max(0, 8 - attachedImages.length))) {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          setAttachedImages((prev) => [
            ...prev.slice(0, 7),
            { name: file.name, dataUrl: reader.result as string },
          ]);
        }
      };
      reader.readAsDataURL(file);
    }
  }

  async function addFolderPath(path: string) {
    setBusy(true);
    setError("");
    try {
      const created = await createWorkProject({ root_path: path });
      const nextProjects = await listWorkProjects();
      setProjects(nextProjects);
      setSelectedProjectId(created.id);
      writeStorage(LAST_PROJECT_KEY, created.id);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleChooseFolder() {
    setError("");
    if (isDesktopApp()) {
      const picked = await pickDesktopFolder();
      if (picked.supported) {
        if (picked.path) await addFolderPath(picked.path);
        return;
      }
    }
    setFolderDialogOpen(true);
  }

  async function handleSend(mode: "send" | "steer" | "queue") {
    const text =
      draft.trim() || (attachedImages.length > 0 ? "Inspect the attached image(s)." : "");
    if (!text) return;

    if (isNew) {
      if (!selectedProjectId) {
        await handleChooseFolder();
        return;
      }
      const chosen = decodeModelSelection(selectedModelKey || pickInitialModel(harnesses));
      const harnessId = chosen.harness || harnesses[0]?.id || "inkling";
      const harnessInfo = harnesses.find((h) => h.id === harnessId) ?? null;
      const modelId = chosen.model || harnessInfo?.default_model || null;
      setBusy(true);
      setError("");
      try {
        const created = await createWorkThread({
          project_id: selectedProjectId,
          harness: harnessId,
          model: modelId,
          model_options: selectedOptions,
          prompt: text,
          images:
            harnessInfo?.supports_images && attachedImages.length > 0
              ? attachedImages.map((img) => img.dataUrl)
              : undefined,
        });
        writeStorage(LAST_PROJECT_KEY, selectedProjectId);
        writeStorage(LAST_MODEL_KEY, encodeModelSelection(harnessId, modelId));
        setDraft("");
        setAttachedImages([]);
        onChanged();
        onThreadCreated?.(created.id);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setBusy(false);
      }
      return;
    }

    if (!threadId) return;
    // Live steering is text-only; always queue image-bearing follow-ups so attachments are preserved.
    const effectiveMode = mode === "steer" && attachedImages.length > 0 ? "queue" : mode;
    setBusy(true);
    setError("");
    try {
      await sendWorkMessage(threadId, {
        prompt: text,
        mode: effectiveMode,
        images: attachedImages.length > 0 ? attachedImages.map((img) => img.dataUrl) : undefined,
      });
      setDraft("");
      setAttachedImages([]);
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleInterrupt() {
    if (!threadId) return;
    setBusy(true);
    setError("");
    try {
      await interruptWorkThread(threadId);
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleOrganize(
    action: "default" | "snooze" | "settle" | "unsettle",
    extra?: { snooze_minutes?: number; snoozed_until?: string },
  ) {
    if (!threadId) return;
    setBusy(true);
    setError("");
    setSnoozeOpen(false);
    try {
      await organizeWorkThread(threadId, { action, ...extra });
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleRespond(requestId: string, response: Record<string, unknown>) {
    if (!threadId) return;
    setBusy(true);
    setError("");
    try {
      await respondWorkPendingRequest(threadId, requestId, response);
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleHarnessOrModelChange(
    nextHarness: string,
    nextModel: string | null,
    nextOptions: Record<string, unknown>,
  ) {
    if (isNew) {
      const encoded = encodeModelSelection(nextHarness, nextModel);
      setSelectedModelKey(encoded);
      setSelectedOptions(nextOptions);
      writeStorage(LAST_MODEL_KEY, encoded);
      writeStorage(LAST_OPTIONS_KEY, JSON.stringify(nextOptions));
      return;
    }
    if (!detail || !threadId) return;
    setBusy(true);
    setError("");
    try {
      await updateWorkThread(threadId, {
        harness: nextHarness,
        model: nextModel,
        model_options: nextOptions,
        expected_version: detail.thread.version,
      });
      writeStorage(LAST_MODEL_KEY, encodeModelSelection(nextHarness, nextModel));
      writeStorage(LAST_OPTIONS_KEY, JSON.stringify(nextOptions));
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    if (!threadId) return;
    setBusy(true);
    setError("");
    try {
      await deleteWorkThread(threadId);
      setConfirmDelete(false);
      onDeleted();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  if (!isNew && loading && !detail) {
    return (
      <div className="center-empty" role="status">
        <span className="sr-only">Loading work thread</span>
        <span className="loader" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      </div>
    );
  }

  if (!isNew && !detail) {
    return (
      <div className="center-empty">
        <h2>Could not open this thread</h2>
        <p className="center-hint">{error || "It may have been deleted."}</p>
      </div>
    );
  }

  const thread = detail?.thread ?? null;
  const project = detail?.project ?? projects.find((p) => p.id === selectedProjectId) ?? null;
  const activity = detail?.activity ?? [];
  const pending = thread ? pendingRequestsFor(thread) : [];

  const decodedSelection = decodeModelSelection(selectedModelKey);
  const activeHarnessId = thread ? thread.harness : decodedSelection.harness;
  const currentHarness = harnesses.find((h) => h.id === activeHarnessId) ?? null;
  const activeModelId = thread
    ? (thread.model ?? currentHarness?.default_model ?? null)
    : decodedSelection.model;
  const activeOptions = thread ? (thread.model_options ?? {}) : selectedOptions;
  const canSend = !busy && (draft.trim().length > 0 || attachedImages.length > 0);
  const followUpMode =
    currentHarness?.supports_steer && attachedImages.length === 0 ? "steer" : "queue";

  const composer = (
    <div className={`work-composer-dock ${isNew ? "is-new" : ""}`}>
      {pending.length > 0 && (
        <div className="work-pending-stack">
          {pending.map((req) => (
            <PendingCard key={req.id} request={req} busy={busy} onRespond={handleRespond} />
          ))}
        </div>
      )}

      <div className={`work-composer ${active ? "is-running" : ""}`}>
        {attachedImages.length > 0 && (
          <div className="work-image-chips">
            {attachedImages.map((img, idx) => (
              <span key={`${img.name}-${idx}`} className="work-image-chip">
                <img src={img.dataUrl} alt="" />
                <span className="work-image-chip-name">{img.name}</span>
                <button
                  type="button"
                  className="icon"
                  aria-label={`Remove ${img.name}`}
                  onClick={() => setAttachedImages((prev) => prev.filter((_, i) => i !== idx))}
                >
                  <X size={12} />
                </button>
              </span>
            ))}
          </div>
        )}
        <textarea
          rows={isNew ? 3 : 2}
          autoFocus={isNew}
          value={draft}
          aria-label="Message"
          onChange={(event) => setDraft(event.target.value)}
          placeholder={
            active
              ? "Add a follow-up or update while this runs…"
              : isNew
                ? "Describe a task, ask a question, or paste an error…"
                : "Ask for follow-up changes…"
          }
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if ((draft.trim() || attachedImages.length > 0) && !busy) {
                void handleSend(active ? followUpMode : "send");
              }
            }
          }}
        />
        <div className="work-composer-bar">
          <div className="work-composer-left">
            {currentHarness?.supports_images && (
              <label className="work-icon-btn" title="Attach screenshots" aria-label="Attach screenshots">
                <Paperclip size={17} />
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  multiple
                  hidden
                  disabled={busy}
                  onChange={(event) => {
                    handleSelectImages(event.target.files);
                    event.target.value = "";
                  }}
                />
              </label>
            )}
          </div>
          <div className="work-composer-right">
            <ModelPicker
              harnesses={harnesses}
              harnessId={activeHarnessId}
              modelId={activeModelId}
              options={activeOptions}
              disabled={busy || active}
              onChange={(h, m, o) => void handleHarnessOrModelChange(h, m, o)}
              onOpenSettings={onOpenSettings}
            />
            {active ? (
              <>
                {currentHarness?.supports_steer && (
                  <button
                    type="button"
                    className="work-text-btn"
                    disabled={busy || !draft.trim() || attachedImages.length > 0}
                    title={
                      attachedImages.length > 0
                        ? "Follow-ups with images are queued as the next turn"
                        : "Send this as a live update to the running turn"
                    }
                    onClick={() => void handleSend("steer")}
                  >
                    Steer
                  </button>
                )}
                <button
                  type="button"
                  className="work-text-btn"
                  disabled={!canSend}
                  title="Run this after the current turn finishes"
                  onClick={() => void handleSend("queue")}
                >
                  Queue
                </button>
                <button
                  type="button"
                  className="work-send is-stop"
                  disabled={busy}
                  title="Stop"
                  onClick={() => void handleInterrupt()}
                >
                  <Square size={12} fill="currentColor" aria-hidden="true" />
                  <span className="sr-only">Stop</span>
                </button>
              </>
            ) : (
              <button
                type="button"
                className="work-send"
                disabled={!canSend}
                title="Send"
                onClick={() => void handleSend("send")}
              >
                <ArrowUp size={17} aria-hidden="true" />
                <span className="sr-only">Send</span>
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="work-composer-meta">
        {isNew ? (
          <ProjectPicker
            projects={projects}
            value={selectedProjectId}
            disabled={busy}
            onChange={(id) => {
              setSelectedProjectId(id);
              writeStorage(LAST_PROJECT_KEY, id);
            }}
            onAddFolder={() => void handleChooseFolder()}
            onManage={onOpenSettings}
          />
        ) : (
          project && (
            <span className="work-meta-item" title={project.root_path}>
              <Folder size={14} />
              {project.title}
            </span>
          )
        )}
        <span className="work-meta-spacer" />
        {currentHarness?.auth.login_method === "local" && currentHarness.auth.status !== "ready" ? (
          <span className="work-meta-item is-warn" role="status">
            {currentHarness.auth.message}
          </span>
        ) : (
          currentHarness && (
            <span className="work-meta-item">
              {currentHarness.auth.login_method === "local" ? "Runs on Kadan Server" : "Runs on this computer"}
            </span>
          )
        )}
      </div>
    </div>
  );

  const dialogs = (
    <>
      {folderDialogOpen && (
        <FolderPickerDialog
          onSelect={(path) => {
            setFolderDialogOpen(false);
            void addFolderPath(path);
          }}
          onClose={() => setFolderDialogOpen(false)}
        />
      )}
      {confirmDelete && (
        <ConfirmDialog
          title="Delete this work thread?"
          message="This deletes the thread transcript. Your project folder is untouched."
          confirm="Delete thread"
          onConfirm={handleDelete}
          onClose={() => setConfirmDelete(false)}
        />
      )}
    </>
  );

  if (isNew) {
    return (
      <div className="work-thread-view is-new">
        {error && (
          <div className="banner" role="alert">
            {error}
          </div>
        )}
        <div className="work-new-stage">
          <h1 className="work-new-title">
            {project ? (
              <>
                What should we build in <span className="work-new-project">{project.title}</span>?
              </>
            ) : (
              "What should we build?"
            )}
          </h1>
          {!project && !loading ? (
            <button
              type="button"
              className="work-new-sub work-new-add"
              disabled={busy}
              onClick={() => void handleChooseFolder()}
            >
              <FolderPlus size={15} />
              Choose a project folder to get started
            </button>
          ) : (
            <p className="work-new-sub">Describe the task and Kadan will start a coding thread.</p>
          )}
          {composer}
        </div>
        {dialogs}
      </div>
    );
  }

  return (
    <div className="work-thread-view">
      {thread && (
        <header className="work-thread-header">
          <div className="work-thread-heading">
            <h2 title={thread.title}>{thread.title}</h2>
            {project && (
              <span className="work-header-project" title={project.root_path}>
                {project.title}
              </span>
            )}
            <span className={`work-status-badge status-${thread.run_status}`}>
              {isThreadActive(thread.run_status) && <span className="work-status-pulse" aria-hidden="true" />}
              {workStatusLabel(thread.run_status)}
            </span>
            {thread.bucket === "snoozed" && thread.snoozed_until && (
              <span className="work-status-badge">
                <AlarmClock size={12} />
                Snoozed · {formatSnoozeLabel(thread.snoozed_until)}
              </span>
            )}
          </div>

          <div className="work-thread-controls">
            <div className="work-snooze-wrap">
              <button
                className="icon work-header-btn"
                disabled={busy}
                aria-label={thread.bucket === "snoozed" ? "Resnooze thread" : "Snooze thread"}
                aria-expanded={snoozeOpen}
                title={thread.bucket === "snoozed" ? "Resnooze" : "Snooze"}
                onClick={() => setSnoozeOpen((open) => !open)}
              >
                <AlarmClock size={16} />
              </button>
              {snoozeOpen && (
                <div className="menu work-snooze-menu" role="menu">
                  <p className="menu-label">Snooze until…</p>
                  <button onClick={() => void handleOrganize("snooze", { snooze_minutes: 30 })}>
                    30 minutes
                  </button>
                  <button onClick={() => void handleOrganize("snooze", { snooze_minutes: 120 })}>
                    2 hours
                  </button>
                  <button
                    onClick={() =>
                      void handleOrganize("snooze", { snoozed_until: tomorrowMorningIso() })
                    }
                  >
                    Tomorrow morning
                  </button>
                  <div className="work-snooze-custom">
                    <input
                      type="datetime-local"
                      aria-label="Snooze until date and time"
                      value={customSnooze}
                      onChange={(event) => setCustomSnooze(event.target.value)}
                    />
                    <button
                      className="primary"
                      disabled={!customSnooze}
                      onClick={() => {
                        const parsed = new Date(customSnooze);
                        if (!Number.isNaN(parsed.getTime())) {
                          void handleOrganize("snooze", { snoozed_until: parsed.toISOString() });
                        }
                      }}
                    >
                      Set
                    </button>
                  </div>
                  {thread.bucket === "snoozed" && (
                    <>
                      <hr />
                      <button onClick={() => void handleOrganize("default")}>Wake now</button>
                    </>
                  )}
                </div>
              )}
            </div>

            {thread.bucket === "settled" ? (
              <button
                className="work-header-pill"
                disabled={busy}
                title="Move back to your active threads"
                onClick={() => void handleOrganize("unsettle")}
              >
                <ArchiveRestore size={14} />
                Unsettle
              </button>
            ) : (
              <button
                className="icon work-header-btn"
                disabled={busy}
                aria-label="Settle thread"
                title="Settle"
                onClick={() => void handleOrganize("settle")}
              >
                <Archive size={16} />
              </button>
            )}

            <button
              className="icon work-header-btn is-danger"
              aria-label="Delete work thread"
              title="Delete thread"
              disabled={busy}
              onClick={() => setConfirmDelete(true)}
            >
              <Trash2 size={16} />
            </button>
          </div>
        </header>
      )}

      {error && (
        <div className="banner" role="alert">
          {error}
        </div>
      )}

      <div className="work-activity-feed">
        <div className="work-activity-inner">
          {activity.length === 0 ? (
            <p className="work-empty-feed">Send a message below to get started.</p>
          ) : (
            activity.map((item) => <ActivityRow key={item.id} item={item} />)
          )}
          {active && thread?.run_status !== "waiting" && (
            <div className="work-working-row" role="status">
              <span className="work-working-dots" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
              {workStatusLabel(thread?.run_status ?? "running")}…
            </div>
          )}
          <div ref={feedEndRef} />
        </div>
      </div>

      {composer}
      {dialogs}
    </div>
  );
}
