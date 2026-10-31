import { useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import type { Resource } from "./types";
import { chatStorageKeys, queueSend, readPending, type PendingMessage } from "./outbox.ts";
import { randomUuid } from "./uuid.ts";

import { codingBusy, codingLabels as labels, type CodingJob, type CodingStatus } from "./coding.ts";

interface LoginStatus {
  state: "signed_in" | "signed_out" | "signing_in" | "missing" | "unavailable";
  message: string;
  login?: { url: string; code: string } | null;
}

interface Props {
  agentKey: string;
  conversation: Resource | null;
  onQueued: (value: Resource) => void;
  onClose: () => void;
}

function readFollowups(key: string): Record<string, string> {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || "{}");
    return value && typeof value === "object" && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).filter(([, text]) => typeof text === "string")) as Record<string, string>
      : {};
  } catch { return {}; }
}

export default function Coding({ agentKey, conversation, onQueued, onClose }: Props) {
  const pendingKey = conversation ? chatStorageKeys(agentKey, conversation.id).pending : "";
  const draftsKey = `${pendingKey}.coding-drafts`;
  const submission = useRef<PendingMessage | null>(pendingKey ? readPending(sessionStorage.getItem(pendingKey)) : null);
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const reading = useRef(false);
  const writing = useRef(false);
  const revision = useRef(0);
  const [status, setStatus] = useState<CodingStatus | null>(null);
  const [login, setLogin] = useState<LoginStatus | null>(null);
  const [authBusy, setAuthBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [stopping, setStopping] = useState("");
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(submission.current?.resume_coding_job_id ?? "");
  const [followups, setFollowups] = useState(() => readFollowups(draftsKey));
  const [continuing, setContinuing] = useState(false);
  const job = status?.jobs.find(row => row.id === selected) ?? status?.jobs[0];
  const resumable = job?.session_id && !["starting", "running", "stopping", "cleanup_required"].includes(job.status);
  const retry = !!job && submission.current?.resume_coding_job_id === job.id;
  const followup = retry ? submission.current!.content : followups[job?.id ?? ""] ?? "";
  const computerBusy = status?.jobs.some(codingBusy);

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => { active.current = false; clearInterval(timer); };
  }, []);

  async function refresh() {
    if (reading.current || writing.current) return;
    reading.current = true;
    const read = revision.current;
    try {
      const [result, auth] = await Promise.all([api<CodingStatus>("/computer/coding"), api<LoginStatus>("/computer/coding/login/status")]);
      if (active.current && read === revision.current) { setStatus(result); setLogin(auth); setError(""); }
    } catch (e) {
      if (active.current && read === revision.current) setError(`Could not refresh coding jobs: ${(e as Error).message}`);
    } finally {
      reading.current = false;
      if (active.current) setLoading(false);
    }
  }

  async function authenticate(operation: "login" | "logout") {
    if (writing.current) return;
    writing.current = true;
    ++revision.current;
    setAuthBusy(true);
    setError("");
    try {
      const result = await api<LoginStatus>(`/computer/coding/${operation}`, {});
      if (active.current) setLogin(result);
    } catch (e) {
      if (active.current) setError(`Sign-in could not be confirmed: ${(e as Error).message}`);
    } finally {
      writing.current = false;
      if (active.current) setAuthBusy(false);
      void refresh();
    }
  }

  async function cancel(id: string) {
    if (writing.current) return;
    writing.current = true;
    ++revision.current;
    setStopping(id);
    setError("");
    try {
      const saved = await api<CodingJob>(`/computer/coding/${encodeURIComponent(id)}/cancel`, {});
      if (active.current) setStatus(current => current && ({ ...current,
        jobs: current.jobs.map(row => row.id === id ? saved : row),
      }));
    } catch (e) {
      if (active.current) setError(`Stop coding is unconfirmed. Refresh to check cleanup: ${(e as Error).message}`);
    } finally {
      writing.current = false;
      if (active.current) setStopping("");
    }
  }

  function close() { active.current = false; dialog.current?.close(); onClose(); }

  function saveFollowup(id: string, text: string) {
    const saved = { ...followups };
    if (text) saved[id] = text; else delete saved[id];
    sessionStorage.setItem(draftsKey, JSON.stringify(saved));
    setFollowups(saved);
  }

  async function continueSession() {
    if (!conversation || !job || !resumable || writing.current) return;
    writing.current = true;
    ++revision.current;
    setContinuing(true);
    setError("");
    try {
      const saved = readPending(sessionStorage.getItem(pendingKey));
      const message = queueSend(saved, retry ? submission.current! : {
        id: randomUuid(), content: followup.trim(), generate: true, persona: "",
        resume_coding_job_id: job.id,
      });
      if (message.resume_coding_job_id !== job.id) throw new Error("Retry or discard the unsent message in Messages first.");
      submission.current = message;
      sessionStorage.setItem(pendingKey, JSON.stringify(message));
      saveFollowup(job.id, "");
      const path = `${agentPath(agentKey)}/conversations/${conversation.id}/runs`;
      const current = await api<Resource>(`${agentPath(agentKey)}/resources/${conversation.id}`);
      if (!active.current) return;
      if (current.id !== conversation.id || current.agent_key !== agentKey) throw new Error("The conversation changed. Reopen Coding before continuing.");
      const result = await api<{ conversation: Resource }>(path, {
        content: message.content, expected_version: current.version, persona_id: null,
        message_id: message.id, resume_coding_job_id: message.resume_coding_job_id,
      });
      if (active.current) {
        if (result.conversation.id !== conversation.id || result.conversation.agent_key !== agentKey) throw new Error("The saved conversation did not match this request. Check Messages before retrying.");
        if (readPending(sessionStorage.getItem(pendingKey))?.id === message.id) sessionStorage.removeItem(pendingKey);
        onQueued(result.conversation);
        close();
      }
    } catch (e) {
      if (active.current) setError(`Continuation is unconfirmed. Check Messages or retry this same request; do not submit a new one. ${(e as Error).message}`);
    } finally {
      writing.current = false;
      if (active.current) setContinuing(false);
    }
  }

  return <dialog ref={dialog} className="coding-dialog" aria-labelledby="coding-title"
    onCancel={event => { event.preventDefault(); close(); }}>
    <div className="coding-content">
      <h2 id="coding-title">Coding</h2>
      {!status ? <p role="status">{loading ? "Checking computer and Codex…" : "Coding status unavailable"}</p> : <>
        <p role="status">{error ? "Last checked: " : ""}{status.setup.message}</p>
        {login && <div className="coding-setup">
          <p role="status">{login.message}</p>
          {login.state === "signed_out" && <button disabled={authBusy} onClick={() => void authenticate("login")}>Sign in with ChatGPT</button>}
          {login.state === "signing_in" && login.login && <p>Open <a href={login.login.url} target="_blank" rel="noreferrer">the sign-in page</a> and enter <strong>{login.login.code}</strong>.</p>}
          {login.state === "signed_in" && <button disabled={authBusy} onClick={() => void authenticate("logout")}>Sign out</button>}
        </div>}
        {status.jobs.length === 0 ? <div className="coding-setup">
          <p>No coding jobs yet.</p>
          {status.setup.state === "ready" ? <p>Choose an existing project folder on the computer. For a first try, ask in Messages: “Ask Codex to look through my project in [folder] and explain it without changing any files.” Replace [folder] with its path.</p>
            : <p>Finish the setup above before asking your agent to code.</p>}
        </div> : <>
          <label>Recent jobs
            <select value={job?.id ?? ""} disabled={continuing || !!submission.current?.resume_coding_job_id} onChange={event => setSelected(event.target.value)}>
              {status.jobs.map(row => <option key={row.id} value={row.id}>
                {labels[row.status]} · {new Date(row.created_at).toLocaleString()}
              </option>)}
            </select>
          </label>
          {job && <>
            <p className="coding-state" role="status">{error ? "Last known state: " : ""}{labels[job.status]}</p>
            <p className="field-hint">Project folder: <code>{job.cwd}</code></p>
            <p className="coding-message">{job.message}</p>
            {job.events.length > 0 && <ol className="coding-events" aria-label="Coding progress">
              {job.events.map(event => <li key={event.sequence}>{event.message}</li>)}
            </ol>}
            {job.events_truncated && <p className="field-hint">Showing recent progress. Earlier events have been omitted.</p>}
            <details><summary>Job details</summary><dl>
              <dt>Job</dt><dd><code>{job.id}</code></dd>
              <dt>Codex session</dt><dd><code>{job.session_id ?? "Not recorded yet"}</code></dd>
            </dl></details>
            {resumable && <form className="coding-setup" onSubmit={event => { event.preventDefault(); void continueSession(); }}>
              <label>Follow-up for this coding session
                <textarea value={followup} maxLength={16000} rows={3} disabled={continuing || !!submission.current?.resume_coding_job_id}
                  placeholder="Describe the next change or check in this project" onChange={event => saveFollowup(job.id, event.target.value)} />
              </label>
              <p className="field-hint">Continue with Codex queues this task directly in Messages and reports the saved job result. It uses your Codex allowance and can change files. A chat claim alone does not confirm a continuation ran.</p>
              <button type="submit" disabled={continuing || !conversation || (!retry && (status.setup.state !== "ready" || computerBusy || !!error || !followup.trim()))}>
                {continuing ? "Queuing continuation…" : retry ? "Retry same continuation" : "Continue with Codex"}
              </button>
            </form>}
            {(["starting", "running", "stopping"] as string[]).includes(job.status) && <button
              disabled={!!stopping || job.status === "stopping"} onClick={() => void cancel(job.id)}>
              {stopping ? "Requesting stop…" : job.status === "stopping" ? "Waiting for cleanup…" : "Stop coding"}
            </button>}
          </>}
        </>}
        <p className="field-hint">Your agent continues in the original conversation when coding finishes. Stop response in Messages prevents that reply and leaves coding running. Stop coding ends the coding task; file changes remain.</p>
      </>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="form-actions">
        <button onClick={() => void refresh()}>Refresh coding jobs</button>
        <button onClick={close}>Close</button>
      </div>
    </div>
  </dialog>;
}
