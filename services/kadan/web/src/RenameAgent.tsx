import { useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import type { Agent } from "./types";

interface Props {
  agent: Agent;
  onUpdated: (agent: Agent) => void;
  onClose: () => void;
}

export default function RenameAgent({ agent, onUpdated, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const pending = useRef(false);
  const [name, setName] = useState(agent.name);
  const [expectedName, setExpectedName] = useState(agent.name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [needsReload, setNeedsReload] = useState(false);

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    return () => { active.current = false; };
  }, []);

  function close() {
    if (pending.current) return;
    active.current = false;
    onClose();
  }

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (pending.current || needsReload || !name.trim()) return;
    pending.current = true;
    setBusy(true);
    setError("");
    try {
      const saved = await api<Agent>(agentPath(agent.key), {
        name: name.trim(), expected_name: expectedName,
      }, "PUT");
      if (!active.current) return;
      onUpdated(saved);
      onClose();
    } catch (e) {
      if (!active.current) return;
      setError(`${(e as Error).message} Reload the current name before trying again.`);
      setNeedsReload(true);
    } finally {
      pending.current = false;
      if (active.current) setBusy(false);
    }
  }

  async function reload() {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    try {
      const saved = await api<Agent>(agentPath(agent.key));
      if (!active.current) return;
      setExpectedName(saved.name);
      onUpdated(saved);
      setNeedsReload(false);
      setError("");
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      pending.current = false;
      if (active.current) setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} className="rename-agent" aria-labelledby="rename-agent-title"
      onCancel={(event) => { event.preventDefault(); close(); }}>
      <form onSubmit={save}>
        <h2 id="rename-agent-title">Rename agent</h2>
        <label>
          Name
          <input autoFocus required maxLength={500} value={name} disabled={busy}
            onChange={(event) => setName(event.target.value)} />
        </label>
        <label>
          Permanent key
          <input readOnly value={agent.key} />
        </label>
        {expectedName !== name.trim() && <p className="field-hint">Saved name: {expectedName}</p>}
        {error && <p className="error" role="alert">{error}</p>}
        {needsReload && <button type="button" disabled={busy} onClick={() => void reload()}>
          Reload current name
        </button>}
        <div className="form-actions">
          <button type="button" disabled={busy} onClick={close}>Cancel</button>
          <button className="primary" disabled={busy || needsReload || !name.trim()}>
            {busy ? "Waiting…" : "Save"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
