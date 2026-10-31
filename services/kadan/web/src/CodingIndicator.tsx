import { useEffect, useState } from "react";
import { api } from "./api";
import { codingBusy, codingLabels, currentCodingJob, type CodingStatus } from "./coding.ts";

export default function CodingIndicator({ onOpen }: { onOpen: () => void }) {
  const [status, setStatus] = useState<CodingStatus | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let closed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const result = await api<CodingStatus>("/computer/coding");
        if (!closed) {
          const unavailable = result.setup.state === "unavailable";
          setStatus(current => unavailable && !result.jobs.length ? current ?? result : result);
          setError(unavailable);
        }
      } catch {
        if (!closed) setError(true);
      } finally {
        if (!closed) timer = setTimeout(refresh, 3000);
      }
    }
    void refresh();
    return () => { closed = true; clearTimeout(timer); };
  }, []);

  const job = status && currentCodingJob(status.jobs);
  const unavailable = error || status?.setup.state === "unavailable";
  const state = !status ? error ? "Status unavailable" : "Checking status…"
    : unavailable ? "Status unavailable"
      : job && codingBusy(job) ? codingLabels[job.status] : "No active job";

  return <div className="chat-coding">
    <div>
      <p role="status">Codex on shared computer · {state}</p>
      {job && <p className="quiet">
        {unavailable ? `Last known: ${codingLabels[job.status]} · ` : codingBusy(job) ? "Project folder: " : `Latest job: ${codingLabels[job.status]} · `}
        {codingBusy(job) ? <code>{job.cwd}</code> : <time dateTime={job.created_at}>started {new Date(job.created_at).toLocaleString()}</time>}
      </p>}
      {unavailable && <p className="quiet">Open Coding to refresh; a job may still be running.</p>}
    </div>
    <button type="button" onClick={onOpen}>Open Coding</button>
  </div>;
}
