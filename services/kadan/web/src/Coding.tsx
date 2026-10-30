import { useEffect, useRef, useState } from "react";
import { api } from "./api";

export interface CodingJob {
  id: string;
  status: "starting" | "running" | "stopping" | "completed" | "failed" | "cancelled" | "timed_out" | "interrupted" | "cleanup_required";
  cwd: string;
  session_id: string | null;
  created_at: string;
  message: string;
  events: { sequence: number; kind: string; message: string }[];
  events_truncated?: boolean;
}

interface CodingStatus {
  setup: { state: string; message: string; version?: string | null };
  jobs: CodingJob[];
}

const labels: Record<CodingJob["status"], string> = {
  starting: "Starting", running: "Working", stopping: "Stopping and cleaning up…",
  completed: "Completed", failed: "Failed", cancelled: "Cancelled", timed_out: "Timed out",
  interrupted: "Interrupted", cleanup_required: "Cleanup needs attention",
};

export default function Coding({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const reading = useRef(false);
  const writing = useRef(false);
  const revision = useRef(0);
  const [status, setStatus] = useState<CodingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [stopping, setStopping] = useState("");
  const [error, setError] = useState("");
  const [selected, setSelected] = useState("");
  const job = status?.jobs.find(row => row.id === selected) ?? status?.jobs[0];

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
      const result = await api<CodingStatus>("/computer/coding");
      if (active.current && read === revision.current) { setStatus(result); setError(""); }
    } catch (e) {
      if (active.current && read === revision.current) setError(`Could not refresh coding jobs: ${(e as Error).message}`);
    } finally {
      reading.current = false;
      if (active.current) setLoading(false);
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

  function close() { active.current = false; onClose(); }

  return <dialog ref={dialog} className="coding-dialog" aria-labelledby="coding-title"
    onCancel={event => { event.preventDefault(); close(); }}>
    <div className="coding-content">
      <h2 id="coding-title">Coding</h2>
      {!status ? <p role="status">{loading ? "Checking computer and Codex…" : "Coding status unavailable"}</p> : <>
        <p role="status">{status.setup.message}</p>
        {status.setup.state === "auth_required" && <div className="coding-setup">
          <p>In the computer’s terminal, sign in with <code>CODEX_HOME=/home/gem/.codex codex login --device-auth</code>, then refresh. Device login must be enabled for your account.</p>
          <p>You can also provision your own API key through Codex’s terminal login. Keep credentials out of chat. A ChatGPT browser session does not sign this computer in.</p>
        </div>}
        {status.setup.version && <p className="field-hint">Codex {status.setup.version} · GPT-6.1 Sol · High effort</p>}
        {status.jobs.length === 0 ? <p>No coding jobs yet. Ask your agent to work on a project in the computer.</p> : <>
          <label>Recent jobs
            <select value={job?.id ?? ""} onChange={event => setSelected(event.target.value)}>
              {status.jobs.map(row => <option key={row.id} value={row.id}>
                {labels[row.status]} · {new Date(row.created_at).toLocaleString()}
              </option>)}
            </select>
          </label>
          {job && <>
            <p className="coding-state">{error ? "Last known state: " : ""}{labels[job.status]}</p>
            <p className="coding-message">{job.message}</p>
            {job.events.length > 0 && <ol className="coding-events" aria-label="Coding progress">
              {job.events.map(event => <li key={event.sequence}>{event.message}</li>)}
            </ol>}
            {job.events_truncated && <p className="field-hint">Showing recent progress. Earlier events have been omitted.</p>}
            <details><summary>Job details</summary><dl>
              <dt>Job</dt><dd><code>{job.id}</code></dd>
              <dt>Directory</dt><dd><code>{job.cwd}</code></dd>
              <dt>Codex session</dt><dd><code>{job.session_id ?? "Not recorded yet"}</code></dd>
            </dl></details>
            {(["starting", "running", "stopping"] as string[]).includes(job.status) && <button
              disabled={!!stopping || job.status === "stopping"} onClick={() => void cancel(job.id)}>
              {stopping ? "Requesting stop…" : job.status === "stopping" ? "Waiting for cleanup…" : "Stop coding"}
            </button>}
          </>}
        </>}
        <p className="field-hint">The agent continues in the original conversation when coding finishes. Chat Stop suppresses that continuation and leaves coding running. Stop coding ends its processes; file changes remain.</p>
      </>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="form-actions">
        <button onClick={() => void refresh()}>Refresh coding jobs</button>
        <button onClick={close}>Close</button>
      </div>
    </div>
  </dialog>;
}
