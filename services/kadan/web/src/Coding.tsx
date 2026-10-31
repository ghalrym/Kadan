import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { CodexSetup, CodexStatus, CodexThread } from "./coding.ts";

interface Props {
  onClose: () => void;
}

export default function Coding({ onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const reading = useRef(false);
  const writing = useRef(false);
  const revision = useRef(0);
  const selectedRef = useRef("");
  const [setup, setSetup] = useState<CodexSetup | null>(null);
  const [authError, setAuthError] = useState("");
  const [status, setStatus] = useState<CodexStatus | null>(null);
  const [thread, setThread] = useState<CodexThread | null>(null);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [threadError, setThreadError] = useState("");
  const [actionError, setActionError] = useState("");

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
    const id = selectedRef.current;
    try {
      await Promise.all([
        api<CodexStatus>("/computer/coding").then(result => {
          if (active.current && read === revision.current) { setStatus(result); setError(""); }
        }).catch(e => {
          if (active.current && read === revision.current) setError(`Could not refresh Codex threads: ${(e as Error).message}`);
        }),
        api<CodexSetup>("/computer/coding/login/status").then(result => {
          if (active.current && read === revision.current) { setSetup(result); setAuthError(""); }
        }).catch(() => {
          if (active.current && read === revision.current) {
            setSetup(null);
            setAuthError("Sign-in status unavailable. Refresh or ask the person running Kadan to update the computer.");
          }
        }),
        ...(id ? [api<CodexThread>(`/computer/coding/threads/${encodeURIComponent(id)}`).then(result => {
          if (active.current && read === revision.current && selectedRef.current === id) { setThread(result); setThreadError(""); }
        }).catch(e => {
          if (active.current && read === revision.current && selectedRef.current === id) setThreadError(`Could not refresh this thread: ${(e as Error).message}`);
        })] : []),
      ]);
    } finally { reading.current = false; }
  }

  function open(id: string) {
    selectedRef.current = id;
    setSelected(id);
    setThread(null);
    setThreadError("");
    ++revision.current;
    void refresh();
  }

  async function act(name: string, path: string) {
    if (writing.current) return;
    writing.current = true;
    ++revision.current;
    setBusy(name);
    setActionError("");
    setAuthError("");
    setError("");
    try {
      const result = await api<CodexSetup>(path, {});
      if (active.current && name !== "stop") setSetup(result);
    } catch (e) {
      if (active.current) {
        if (name === "stop") setActionError(`Stop is unconfirmed: ${(e as Error).message}`);
        else setAuthError(`Sign-in could not be confirmed: ${(e as Error).message}`);
      }
    } finally {
      writing.current = false;
      if (active.current) setBusy("");
      void refresh();
    }
  }

  function close() { active.current = false; dialog.current?.close(); onClose(); }

  const summary = status?.threads.find(row => row.id === selected) ?? thread;

  return <dialog ref={dialog} className="coding-dialog" aria-labelledby="coding-title"
    onCancel={event => { event.preventDefault(); close(); }}>
    <div className="coding-content">
      <h2 id="coding-title">Coding</h2>
      {authError && <p className="error" role="alert">{authError}</p>}
      {setup && <>
        <div className="coding-setup">
          <p role="status">{setup.message}</p>
          {setup.state === "signed_out" && <button disabled={!!busy} onClick={() => void act("login", "/computer/coding/login")}>
            {busy === "login" ? "Starting sign-in…" : "Sign in with ChatGPT"}
          </button>}
          {setup.state === "signing_in" && setup.login && <>
            <a href={setup.login.url} target="_blank" rel="noreferrer">Open sign-in page<span className="sr-only"> (opens in a new tab)</span></a>
            <p>Code: <code className="coding-code">{setup.login.code}</code></p>
          </>}
          {["signed_in", "signing_in"].includes(setup.state) && <button disabled={!!busy} onClick={() => void act("logout", "/computer/coding/logout")}>
            {busy === "logout" ? "Signing out…" : setup.state === "signing_in" ? "Cancel sign-in" : "Sign out"}
          </button>}
        </div>
      </>}
      <p className="field-hint">Sign-in does not confirm model access or remaining usage. Keep passwords, codes and API keys out of chat.</p>
      {!status ? <p role="status">{error ? "Codex status unavailable" : "Checking Codex…"}</p> : <>
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
      {threadError && <p className="error" role="alert">{threadError}</p>}
      {actionError && <p className="error" role="alert">{actionError}</p>}
      <div className="form-actions">
        <button onClick={() => void refresh()}>Refresh</button>
        <button onClick={close}>Close</button>
      </div>
    </div>
  </dialog>;
}
