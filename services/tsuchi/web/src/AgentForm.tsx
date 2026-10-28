import { useEffect, useRef, useState } from "react";
import { Plus } from "lucide-react";
import { json, request } from "./api";
import { ErrorNotice } from "./components";
import type { Agent } from "./types";

export function AgentForm({
  onCreated,
}: {
  onCreated: (agent: Agent) => void;
}) {
  const [name, setName] = useState("");
  const [key, setKey] = useState("");
  const [keyEdited, setKeyEdited] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const active = useRef(false);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  return (
    <form
      className="agent-form"
      onSubmit={async (event) => {
        event.preventDefault();
        setBusy(true);
        setError("");
        try {
          const agent = await request<Agent>(
            "/api/agents",
            json("POST", { name: name.trim(), key }),
          );
          if (active.current) onCreated(agent);
        } catch (error) {
          if (!active.current) return;
          setError(
            error instanceof Error
              ? error.message
              : "Unable to create namespace.",
          );
        } finally {
          if (active.current) setBusy(false);
        }
      }}
    >
      <ErrorNotice message={error} />
      <label>
        Name
        <input
          required
          value={name}
          maxLength={200}
          autoFocus
          onChange={(event) => {
            setName(event.target.value);
            if (!keyEdited)
              setKey(
                event.target.value
                  .toLowerCase()
                  .replace(/[^a-z0-9_-]+/g, "-")
                  .replace(/^[^a-z0-9]+/, "")
                  .slice(0, 64),
              );
          }}
        />
      </label>
      <label>
        Namespace key
        <input
          className="mono"
          required
          value={key}
          maxLength={64}
          pattern="[a-z0-9][a-z0-9_-]{0,63}"
          title="Use 1–64 lowercase letters, digits, underscores, or hyphens. Start with a letter or digit."
          onChange={(event) => {
            setKeyEdited(true);
            setKey(event.target.value);
          }}
        />
      </label>
      <button className="primary" disabled={busy || !name.trim() || !key}>
        <Plus size={18} />
        {busy ? "Creating…" : "Create namespace"}
      </button>
    </form>
  );
}
