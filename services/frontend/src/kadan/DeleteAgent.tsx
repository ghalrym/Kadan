import { useEffect, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import { agentPath, api } from "./api";
import type { Agent } from "./types";

interface Props {
  agent: Agent;
  onDeleted: () => void;
  onClose: () => void;
}

export function clearAgentStorage(agentKey: string) {
  const prefixes = [
    `kadan.draft.${agentKey}`,
    `kadan.memory-draft.${agentKey}.`,
    `kadan.attachments.${agentKey}.`,
  ];
  const keys: string[] = [];
  for (let i = 0; i < sessionStorage.length; i++) {
    const key = sessionStorage.key(i);
    if (key && prefixes.some((prefix) => key === prefix || key.startsWith(prefix))) {
      keys.push(key);
    }
  }
  for (const key of keys) sessionStorage.removeItem(key);
}

/** Requires typing "<agent name> Delete" before permanently deleting the agent and all its data. */
export default function DeleteAgent({ agent, onDeleted, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const pending = useRef(false);
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const expected = `${agent.name} Delete`;
  const matched = confirmation.trim() === expected;

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

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (pending.current || !matched) return;
    pending.current = true;
    setBusy(true);
    setError("");
    try {
      await api<void>(agentPath(agent.key), undefined, "DELETE");
      clearAgentStorage(agent.key);
      if (!active.current) return;
      onDeleted();
      onClose();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      pending.current = false;
      if (active.current) setBusy(false);
    }
  }

  return (
    <dialog
      ref={dialog}
      className="delete-agent"
      aria-labelledby="delete-agent-title"
      onCancel={(event) => { event.preventDefault(); close(); }}
    >
      <form onSubmit={submit}>
        <h2 id="delete-agent-title">Delete {agent.name}?</h2>
        <p className="field-hint">
          This permanently deletes <strong>{agent.name}</strong>, all saved memories, chat history,
          persona, Space items, and custom pets so you can start completely fresh. This cannot be undone.
        </p>
        <label>
          <span>
            Type <strong>{expected}</strong> to confirm
          </span>
          <input
            autoFocus
            required
            autoComplete="off"
            spellCheck={false}
            placeholder={expected}
            value={confirmation}
            disabled={busy}
            onChange={(event) => setConfirmation(event.target.value)}
          />
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="form-actions">
          <button type="button" disabled={busy} onClick={close}>Cancel</button>
          <button className="danger-primary" disabled={busy || !matched}>
            <Trash2 size={15} />
            {busy ? "Deleting…" : "Delete agent"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
