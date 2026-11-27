import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Archive,
  ArchiveRestore,
  Check,
  Clock,
  FileCode2,
  FolderGit2,
  GitBranch,
  Send,
  Square,
  Terminal,
  Trash2,
  X,
} from "lucide-react";
import ConfirmDialog from "./ConfirmDialog";
import Markdown from "./Markdown";
import {
  deleteWorkThread,
  formatSnoozeLabel,
  getWorkThreadDetail,
  interruptWorkThread,
  isThreadActive,
  listWorkHarnesses,
  organizeWorkThread,
  pendingRequestsFor,
  respondWorkPendingRequest,
  sendWorkMessage,
  updateWorkThread,
  workStatusLabel,
  type WorkActivityItem,
  type WorkHarness,
  type WorkPendingRequest,
  type WorkThreadDetail,
} from "./work";

interface Props {
  threadId: string;
  onChanged: () => void;
  onDeleted: () => void;
}

function tomorrowMorningIso(): string {
  const next = new Date();
  next.setDate(next.getDate() + 1);
  next.setHours(9, 0, 0, 0);
  return next.toISOString();
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
    <section className={`work-pending-card ${isApproval ? "is-approval" : "is-question"}`} aria-live="polite">
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
    return (
      <article className="work-item work-item-user">
        <div className="work-item-meta">{steered ? "You (update)" : "You"}</div>
        <div className="work-item-body">{item.text}</div>
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
      <details className="work-item work-item-reasoning">
        <summary>Thinking progress</summary>
        <p>{item.text}</p>
      </details>
    );
  }
  if (item.kind === "tool_call" || item.kind === "tool_result") {
    const toolName = String(item.data?.tool_name ?? "Action");
    const isError = Boolean(item.data?.is_error);
    return (
      <details className={`work-item work-item-tool ${isError ? "is-error" : ""}`}>
        <summary>
          <Terminal size={13} />
          <span>{item.kind === "tool_call" ? `Running ${toolName}` : `Finished ${toolName}`}</span>
          {isError && <span className="work-pill error">Failed</span>}
        </summary>
        {item.text && <pre className="work-code-block">{item.text}</pre>}
      </details>
    );
  }
  if (item.kind === "file_change") {
    const filePath = String(item.data?.file_path ?? "File updated");
    const changeType = String(item.data?.change_type ?? "modified");
    return (
      <div className="work-item work-item-file">
        <FileCode2 size={14} />
        <span className="work-file-path">{filePath}</span>
        <span className="work-pill">{changeType}</span>
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

export default function WorkThreadView({ threadId, onChanged, onDeleted }: Props) {
  const [detail, setDetail] = useState<WorkThreadDetail | null>(null);
  const [harnesses, setHarnesses] = useState<WorkHarness[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState("");
  const [snoozeOpen, setSnoozeOpen] = useState(false);
  const [customSnooze, setCustomSnooze] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const feedEndRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async (showSpinner = false) => {
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
  }, [threadId]);

  useEffect(() => {
    void load(true);
    void listWorkHarnesses()
      .then(setHarnesses)
      .catch(() => {});
  }, [load]);

  const active = detail ? isThreadActive(detail.thread.run_status) : false;

  useEffect(() => {
    const intervalMs = active ? 1200 : 6000;
    const timer = setInterval(() => void load(false), intervalMs);
    return () => clearInterval(timer);
  }, [active, load]);

  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [detail?.activity.length]);

  async function handleSend(mode: "send" | "steer" | "queue") {
    const text = draft.trim();
    if (!text) return;
    setBusy(true);
    setError("");
    try {
      await sendWorkMessage(threadId, { prompt: text, mode });
      setDraft("");
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleInterrupt() {
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

  async function handleHarnessOrModelChange(nextHarness: string, nextModel: string | null) {
    if (!detail) return;
    setBusy(true);
    setError("");
    try {
      await updateWorkThread(threadId, {
        harness: nextHarness,
        model: nextModel,
        expected_version: detail.thread.version,
      });
      await load(false);
      onChanged();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
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

  if (loading && !detail) {
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

  if (!detail) {
    return (
      <div className="center-empty">
        <h2>Could not open this work thread</h2>
        <p className="center-hint">{error || "It may have been deleted."}</p>
      </div>
    );
  }

  const { thread, project, activity } = detail;
  const pending = pendingRequestsFor(thread);
  const currentHarness = harnesses.find((h) => h.id === thread.harness) ?? null;

  return (
    <div className="work-thread-view">
      <header className="work-thread-header">
        <div className="work-thread-title-group">
          <div className="work-thread-title-row">
            <h2>{thread.title}</h2>
            <span className={`work-status-badge status-${thread.run_status}`}>
              {workStatusLabel(thread.run_status)}
            </span>
          </div>
          <div className="work-thread-meta">
            {project && (
              <span className="work-meta-chip">
                <FolderGit2 size={13} />
                {project.title}
              </span>
            )}
            <span className="work-meta-chip">
              <GitBranch size={13} />
              {thread.workspace_mode === "worktree"
                ? thread.branch || "Isolated branch"
                : "Project folder"}
            </span>
            {thread.bucket === "snoozed" && thread.snoozed_until && (
              <span className="work-meta-chip">
                <Clock size={13} />
                Snoozed ({formatSnoozeLabel(thread.snoozed_until)})
              </span>
            )}
          </div>
        </div>

        <div className="work-thread-controls">
          {harnesses.length > 0 && (
            <div className="work-picker-group">
              <select
                aria-label="Coding harness"
                value={thread.harness}
                disabled={busy || active}
                onChange={(event) => {
                  const chosen = harnesses.find((h) => h.id === event.target.value);
                  void handleHarnessOrModelChange(
                    event.target.value,
                    chosen?.default_model ?? null,
                  );
                }}
              >
                {harnesses.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.display_name}
                  </option>
                ))}
              </select>
              {currentHarness && currentHarness.models.length > 0 && (
                <select
                  aria-label="Model"
                  value={thread.model ?? currentHarness.default_model ?? ""}
                  disabled={busy || active}
                  onChange={(event) =>
                    void handleHarnessOrModelChange(thread.harness, event.target.value || null)
                  }
                >
                  {currentHarness.models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          <div className="work-snooze-wrap">
            <button
              className="button"
              disabled={busy}
              onClick={() => setSnoozeOpen((open) => !open)}
              title="Snooze this thread"
            >
              <Clock size={14} />
              {thread.bucket === "snoozed" ? "Resnooze" : "Snooze"}
            </button>
            {snoozeOpen && (
              <div className="menu work-snooze-menu" role="menu">
                <button onClick={() => void handleOrganize("snooze", { snooze_minutes: 30 })}>
                  For 30 minutes
                </button>
                <button onClick={() => void handleOrganize("snooze", { snooze_minutes: 120 })}>
                  For 2 hours
                </button>
                <button
                  onClick={() =>
                    void handleOrganize("snooze", { snoozed_until: tomorrowMorningIso() })
                  }
                >
                  Until tomorrow morning
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
                    Set time
                  </button>
                </div>
                {thread.bucket === "snoozed" && (
                  <button onClick={() => void handleOrganize("default")}>Wake now</button>
                )}
              </div>
            )}
          </div>

          {thread.bucket === "settled" ? (
            <button
              className="button"
              disabled={busy}
              onClick={() => void handleOrganize("unsettle")}
            >
              <ArchiveRestore size={14} />
              Unsettle
            </button>
          ) : (
            <button
              className="button"
              disabled={busy}
              onClick={() => void handleOrganize("settle")}
            >
              <Archive size={14} />
              Settle
            </button>
          )}

          <button
            className="icon"
            aria-label="Delete work thread"
            title="Delete thread"
            disabled={busy}
            onClick={() => setConfirmDelete(true)}
          >
            <Trash2 size={15} />
          </button>
        </div>
      </header>

      {error && (
        <div className="banner" role="alert">
          {error}
        </div>
      )}

      <div className="work-activity-feed">
        {activity.length === 0 ? (
          <div className="work-empty-feed">
            <p className="quiet">Send a message below to start working in this project.</p>
          </div>
        ) : (
          activity.map((item) => <ActivityRow key={item.id} item={item} />)
        )}
        <div ref={feedEndRef} />
      </div>

      {pending.length > 0 && (
        <div className="work-pending-stack">
          {pending.map((req) => (
            <PendingCard
              key={req.id}
              request={req}
              busy={busy}
              onRespond={handleRespond}
            />
          ))}
        </div>
      )}

      <footer className="work-composer">
        <textarea
          rows={2}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={
            active
              ? "Add a follow-up or update while this runs…"
              : "Describe what you want done next…"
          }
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (draft.trim() && !busy) {
                void handleSend(
                  active ? (currentHarness?.supports_steer ? "steer" : "queue") : "send",
                );
              }
            }
          }}
        />
        <div className="work-composer-actions">
          {active ? (
            <>
              <button
                className="button"
                disabled={busy}
                onClick={() => void handleInterrupt()}
              >
                <Square size={14} />
                Stop
              </button>
              {currentHarness?.supports_steer && (
                <button
                  className="button"
                  disabled={busy || !draft.trim()}
                  onClick={() => void handleSend("steer")}
                >
                  Steer
                </button>
              )}
              <button
                className="primary"
                disabled={busy || !draft.trim()}
                onClick={() => void handleSend("queue")}
              >
                <Send size={14} />
                Queue
              </button>
            </>
          ) : (
            <button
              className="primary"
              disabled={busy || !draft.trim()}
              onClick={() => void handleSend("send")}
            >
              <Send size={14} />
              Send
            </button>
          )}
        </div>
      </footer>

      {confirmDelete && (
        <ConfirmDialog
          title="Delete this work thread?"
          message={
            thread.workspace_mode === "worktree"
              ? "This deletes the thread transcript and removes its isolated branch copy."
              : "This deletes the thread transcript. Your project folder is untouched."
          }
          confirm="Delete thread"
          onConfirm={handleDelete}
          onClose={() => setConfirmDelete(false)}
        />
      )}
    </div>
  );
}
