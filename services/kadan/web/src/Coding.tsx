import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { CodexStatus, CodexThread } from "./coding.ts";

interface Props {
  onClose: () => void;
}

export default function Coding({ onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const selectedRef = useRef("");
  const [status, setStatus] = useState<CodexStatus | null>(null);
  const [thread, setThread] = useState<CodexThread | null>(null);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => { active.current = false; clearInterval(timer); };
  }, []);

  async function refresh() {
    try {
      const result = await api<CodexStatus>("/computer/coding");
      if (!active.current) return;
      setStatus(result);
      setError("");
      const id = selectedRef.current;
      if (id) {
        const current = await api<CodexThread>(`/computer/coding/threads/${encodeURIComponent(id)}`);
        if (active.current && selectedRef.current === id) setThread(current);
      }
    } catch (e) {
      if (active.current) setError((e as Error).message);
    }
  }

  function open(id: string) {
    selectedRef.current = id;
    setSelected(id);
    setThread(null);
    void refresh();
  }

  async function act(name: string, path: string) {
    setBusy(name);
    setError("");
    try {
      await api(path, {});
      await refresh();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy("");
    }
  }

  function close() { active.current = false; dialog.current?.close(); onClose(); }

  const setup = status?.setup;
  const summary = status?.threads.find(row => row.id === selected);

  return <dialog ref={dialog} className="coding-dialog" aria-labelledby="coding-title"
    onCancel={event => { event.preventDefault(); close(); }}>
    <div className="coding-content">
      <h2 id="coding-title">Coding</h2>
      {!setup ? <p role="status">{error ? "Codex status unavailable" : "Checking Codex…"}</p> : <>
        <div className="coding-setup">
          <p role="status">{setup.message}</p>
          {setup.state === "signed_out" && <button disabled={!!busy} onClick={() => void act("login", "/computer/coding/login")}>
            {busy === "login" ? "Starting sign-in…" : "Sign in with ChatGPT"}
          </button>}
          {setup.state === "signing_in" && setup.login && <>
            <a href={setup.login.url} target="_blank" rel="noreferrer">Open sign-in page<span className="sr-only"> (opens in a new tab)</span></a>
            <p>Code: <code className="coding-code">{setup.login.code}</code></p>
          </>}
          {setup.state === "signed_in" && <button disabled={!!busy} onClick={() => void act("logout", "/computer/coding/logout")}>
            {busy === "logout" ? "Signing out…" : "Sign out"}
          </button>}
        </div>
        {selected ? <>
          <button onClick={() => open("")}>All threads</button>
          {summary && <>
            <h3>{summary.title}</h3>
            <p className="field-hint"><code>{summary.cwd}</code> · {summary.running ? "Working" : "Idle"}</p>
            {summary.running && <button disabled={!!busy}
              onClick={() => void act("stop", `/computer/coding/threads/${encodeURIComponent(summary.id)}/stop`)}>
              {busy === "stop" ? "Stopping…" : "Stop"}
            </button>}
          </>}
          {!thread ? <p role="status">Loading thread…</p> : <ol className="coding-thread" aria-label="Thread">
            {thread.items_truncated && <li className="field-hint">Earlier messages are hidden.</li>}
            {thread.items.map((item, index) => <li key={index} className={`coding-item ${item.kind}`}>{item.text}</li>)}
          </ol>}
        </> : status.threads.length === 0 ? <p>No Codex threads yet. Ask your agent to use Codex on a project and its threads will show up here.</p>
          : <ul className="coding-threads" aria-label="Codex threads">
            {status.threads.map(row => <li key={row.id}>
              <button onClick={() => open(row.id)}>
                <strong>{row.title}</strong>
                <span className="field-hint">{row.running ? "Working · " : ""}<code>{row.cwd}</code> · {new Date(row.updated_at).toLocaleString()}</span>
              </button>
            </li>)}
          </ul>}
      </>}
      {error && <p className="error" role="alert">{error}</p>}
      <div className="form-actions">
        <button onClick={close}>Close</button>
      </div>
    </div>
  </dialog>;
}
