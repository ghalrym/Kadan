import { useEffect, useRef, useState } from "react";
import { Egg, X } from "lucide-react";
import { api } from "./api";
import Pet, { agentPet, useMood } from "./Pet";
import { agentHue } from "./pet";
import type { Agent } from "./types";

const slug = (name: string) =>
  name
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^[-_]+|-+$/g, "")
    .slice(0, 64);

interface Props {
  open: boolean;
  onClose: () => void;
  onHatched: (agent: Agent) => void;
}

export default function Hatch({ open, onClose, onHatched }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState("");
  const [key, setKey] = useState("");
  const [keyEdited, setKeyEdited] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const preview = key || "kadan";
  const hue = agentHue(preview);
  const [mood, react] = useMood(saving ? "running" : error ? "failed" : name ? "idle" : "waiting");

  useEffect(() => {
    if (open) {
      setName("");
      setKey("");
      setKeyEdited(false);
      setError("");
      dialog.current?.showModal();
    } else dialog.current?.close();
  }, [open]);
  useEffect(() => {
    if (open) react("waving");
  }, [hue, open, react]);

  async function hatch(event: React.FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError("");
    try {
      onHatched(await api<Agent>("/agents", { name: name.trim(), key }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <dialog
      ref={dialog}
      className="hatch"
      style={{ "--agent-h": hue } as React.CSSProperties}
      onCancel={(event) => {
        event.preventDefault();
        if (!saving) onClose();
      }}
      aria-labelledby="hatch-title"
    >
      <form onSubmit={hatch}>
        <button
          type="button"
          className="icon close"
          aria-label="Close"
          disabled={saving}
          onClick={onClose}
        >
          <X size={18} />
        </button>
        <div className="hatch-stage">
          {open && <Pet sheet={agentPet(preview)} state={mood} size={144} look grounded />}
        </div>
        <h2 id="hatch-title">Hatch an agent</h2>
        <label>
          Name
          <input
            name="name"
            autoFocus
            required
            maxLength={120}
            placeholder="Research"
            value={name}
            onChange={(event) => {
              setName(event.target.value);
              if (!keyEdited) setKey(slug(event.target.value));
            }}
          />
        </label>
        <label>
          Key
          <input
            name="key"
            required
            pattern="[a-z0-9][a-z0-9_-]{0,63}"
            maxLength={64}
            placeholder="research"
            autoCapitalize="none"
            autoComplete="off"
            spellCheck={false}
            value={key}
            onChange={(event) => {
              setKey(event.target.value);
              setKeyEdited(true);
            }}
            aria-describedby="key-hint"
          />
        </label>
        <p className="field-hint" id="key-hint">
          a–z, 0–9, - and _ · permanent
        </p>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <div className="form-actions">
          <button type="button" disabled={saving} onClick={onClose}>
            Cancel
          </button>
          <button className="primary" disabled={saving || !name.trim() || !key}>
            <Egg size={16} />
            {saving ? "Hatching…" : "Hatch"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
