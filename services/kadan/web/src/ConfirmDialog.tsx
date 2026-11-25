import { useEffect, useRef, useState } from "react";

interface Props {
  title: string;
  message: string;
  confirm: string;
  onConfirm: () => Promise<void>;
  onClose: () => void;
}

/** Ask before an action that cannot be undone from Kadan. */
export default function ConfirmDialog({ title, message, confirm, onConfirm, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const active = useRef(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    active.current = true;
    dialog.current?.showModal();
    return () => { active.current = false; };
  }, []);
  async function run() {
    setBusy(true);
    setError("");
    try {
      await onConfirm();
      if (active.current) onClose();
    } catch (e) {
      if (active.current) setError((e as Error).message);
    } finally {
      if (active.current) setBusy(false);
    }
  }
  return (
    <dialog ref={dialog} className="confirm-dialog" aria-labelledby="confirm-title"
      onCancel={(event) => { event.preventDefault(); if (!busy) onClose(); }}>
      <form onSubmit={(event) => { event.preventDefault(); void run(); }}>
        <h2 id="confirm-title">{title}</h2>
        <p>{message}</p>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="form-actions">
          <button type="button" autoFocus disabled={busy} onClick={onClose}>Cancel</button>
          <button className="primary danger" disabled={busy}>{busy ? "Working…" : confirm}</button>
        </div>
      </form>
    </dialog>
  );
}
