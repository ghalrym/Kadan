import { useEffect, useRef, useState } from "react";
import { api, agentPath } from "./api";
import type { Resource } from "./types";

export interface SelectedReference {
  resource_id: string;
  expected_version: number;
  title: string;
}

export function eligibleReferences(resources: Resource[], agentKey: string): Resource[] {
  return resources.filter((item) => item.agent_key === agentKey && !item.deleted_at &&
    (item.kind === "space" || item.kind === "memory"));
}

export default function RequestReferences({ agentKey, selected, onChange, disabled }: {
  agentKey: string;
  selected: SelectedReference[];
  onChange: (value: SelectedReference[]) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [resources, setResources] = useState<Resource[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const request = useRef(0);
  const agent = useRef(agentKey);
  agent.current = agentKey;
  useEffect(() => {
    setOpen(false);
    setResources([]);
    setQuery("");
    setError("");
    setLoading(false);
    setTruncated(false);
    return () => { request.current++; };
  }, [agentKey]);
  async function load() {
    const revision = ++request.current;
    setOpen(true);
    setError("");
    setResources([]);
    setTruncated(false);
    setLoading(true);
    try {
      const values = await Promise.all(["workspace/documents", "memories"].map((path) =>
        api<Resource[]>(`${agentPath(agentKey)}/${path}?limit=200&q=${encodeURIComponent(query)}`)));
      if (revision !== request.current || agent.current !== agentKey) return;
      setResources(eligibleReferences(values.flat(), agentKey));
      setTruncated(values.some((items) => items.length >= 200));
    } catch (e) {
      if (revision === request.current && agent.current === agentKey) setError((e as Error).message);
    } finally {
      if (revision === request.current && agent.current === agentKey) setLoading(false);
    }
  }
  return <div className="request-references">
    <button disabled={disabled} onClick={() => {
      if (open) { request.current++; setOpen(false); }
      else void load();
    }}>
      References{selected.length ? ` (${selected.length}/3)` : ""}
    </button>
    {selected.map((item) => <button key={item.resource_id} disabled={disabled}
      onClick={() => onChange(selected.filter((value) => value.resource_id !== item.resource_id))}>
      {item.title} · v{item.expected_version} ×
    </button>)}
    {open && <div>
      <p>Choose up to three saved Markdown spaces or memories for your next reply, with at most 12,000 Markdown characters combined. The agent can read these versions; use is not guaranteed.</p>
      <form onSubmit={(event) => { event.preventDefault(); if (!disabled) void load(); }}>
        <input aria-label="Search references" value={query} onChange={(event) => setQuery(event.target.value)} />
        <button disabled={disabled}>Search</button>
      </form>
      {loading && <p role="status">Loading references…</p>}
      {error && <p role="alert">{error}</p>}
      {!loading && !error && resources.length === 0 && <p>No matching references.</p>}
      {resources.map((item) => {
        const checked = selected.some((value) => value.resource_id === item.id);
        return <label key={item.id} style={{ display: "block" }}><input type="checkbox" checked={checked}
          disabled={disabled || loading || (!checked && selected.length >= 3)}
          onChange={() => onChange(checked ? selected.filter((value) => value.resource_id !== item.id) :
            [...selected, { resource_id: item.id, expected_version: item.version, title: item.title }])} />
          {item.title} · {item.kind} · v{item.version}
        </label>;
      })}
      {truncated && <p>Showing at most 200 records from each collection. Search to narrow these results.</p>}
    </div>}
  </div>;
}
