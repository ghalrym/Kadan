import { useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import type { Space } from "./types";

interface Props {
  agentKey: string;
  /** Editing an existing Space; omitted to create one. */
  space?: Space;
  onSaved: (space: Space) => void;
  onClose: () => void;
}

/** Create a Space or change its name and instructions. */
export default function SpaceDialog({ agentKey, space, onSaved, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const [title, setTitle] = useState(space?.title ?? "");
  const [instructions, setInstructions] = useState(space?.instructions ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    return () => { active.current = false; };
  }, []);

  function close() {
    if (!busy) onClose();
  }

  async function save(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !title.trim()) return;
    setBusy(true);
    setError("");
    try {
      const base = `${agentPath(agentKey)}/spaces`;
      const saved = space
        ? await api<Space>(`${base}/${space.id}`, {
          title: title.trim(), instructions, expected_version: space.version,
        }, "PUT")
        : await api<Space>(base, { title: title.trim(), instructions });
      if (!active.current) return;
      onSaved(saved);
      onClose();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} className="space-dialog" aria-labelledby="space-dialog-title"
      onCancel={(event) => { event.preventDefault(); close(); }}
      onClick={(event) => { if (event.target === dialog.current) close(); }}>
      <form onSubmit={save}>
        <h2 id="space-dialog-title">{space ? "Space settings" : "New Space"}</h2>
        {!space && (
          <p className="field-hint">
            Spaces keep related chats together and share instructions across them.
          </p>
        )}
        <label>
          Name
          <input autoFocus required maxLength={200} value={title} disabled={busy}
            placeholder="Trip planning, Garden, Side project…"
            onChange={(event) => setTitle(event.target.value)} />
        </label>
        <label>
          Instructions
          <textarea rows={6} maxLength={48000} value={instructions} disabled={busy}
            placeholder="How should the agent respond in this Space? Share goals, tone or background it should keep in mind."
            onChange={(event) => setInstructions(event.target.value)} />
        </label>
        <p className="field-hint">Instructions apply to every chat in this Space.</p>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="form-actions">
          <button type="button" disabled={busy} onClick={close}>Cancel</button>
          <button className="primary" disabled={busy || !title.trim()}>
            {busy ? "Saving…" : space ? "Save" : "Create Space"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
