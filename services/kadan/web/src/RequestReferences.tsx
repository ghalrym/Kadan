import { useEffect, useRef, useState } from "react";
import { BookOpen, Brain, Check, FileText, Search, X } from "lucide-react";
import { api, agentPath } from "./api";
import type { Resource } from "./types";

export interface SelectedReference {
  resource_id: string;
  expected_version: number;
  title: string;
  characters?: number;
}

export const REFERENCE_LIMIT = 3;
export const REFERENCE_CHARACTERS = 12_000;

export function eligibleReferences(resources: Resource[], agentKey: string): Resource[] {
  return resources.filter((item) => item.agent_key === agentKey && !item.deleted_at &&
    (item.kind === "space" || item.kind === "memory"));
}

const count = (value: number) => value.toLocaleString("en-US");

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
  const searched = useRef<string | null>("");
  const container = useRef<HTMLDivElement>(null);
  const agent = useRef(agentKey);
  agent.current = agentKey;
  useEffect(() => {
    setOpen(false);
    setResources([]);
    setQuery("");
    setError("");
    setLoading(false);
    setTruncated(false);
    searched.current = "";
    return () => { request.current++; };
  }, [agentKey]);
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) close();
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);
  useEffect(() => {
    if (!open || disabled || query === searched.current) return;
    const timer = setTimeout(() => { if (query !== searched.current) void load(); }, 250);
    return () => clearTimeout(timer);
  }, [open, query, disabled]);
  async function load() {
    const revision = ++request.current;
    searched.current = query;
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
  function close() {
    request.current++;
    setOpen(false);
    setLoading(false);
  }
  function changeQuery(value: string) {
    request.current++;
    searched.current = null;
    setQuery(value);
    setResources([]);
    setError("");
    setTruncated(false);
    setLoading(true);
  }
  const remove = (id: string) => onChange(selected.filter((value) => value.resource_id !== id));
  const full = selected.length >= REFERENCE_LIMIT;
  const measured = selected.every((item) => item.characters !== undefined);
  const total = selected.reduce((sum, item) => sum + (item.characters ?? 0), 0);
  const over = total > REFERENCE_CHARACTERS;
  return <div className="request-references" ref={container}
    onKeyDown={(event) => { if (open && event.key === "Escape") { event.stopPropagation(); close(); } }}
    onBlur={(event) => {
      if (open && event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) close();
    }}>
    <div className="reference-row">
      <button type="button" className={`chip${open || selected.length ? " active" : ""}`} disabled={disabled}
        aria-expanded={open} aria-haspopup="dialog" onClick={() => { if (open) close(); else void load(); }}>
        <BookOpen size={14} />
        References{selected.length ? ` ${selected.length}/${REFERENCE_LIMIT}` : ""}
      </button>
      {selected.map((item) => <span key={item.resource_id} className="reference-pill" title={item.title}>
        <span className="reference-pill-title">{item.title}</span>
        <span className="reference-pill-version">v{item.expected_version}</span>
        <button type="button" disabled={disabled} aria-label={`Remove ${item.title}`}
          onClick={() => remove(item.resource_id)}>
          <X size={12} />
        </button>
      </span>)}
      {selected.length > 0 && measured &&
        <span className={`reference-budget${over ? " over" : ""}`}>
          {count(total)} / {count(REFERENCE_CHARACTERS)}
        </span>}
    </div>
    {open && <div className="reference-panel" role="dialog" aria-label="Choose references">
      <form className="reference-search" onSubmit={(event) => { event.preventDefault(); if (!disabled) void load(); }}>
        <Search size={14} />
        <input aria-label="Search references" placeholder="Search" autoFocus
          value={query} onChange={(event) => changeQuery(event.target.value)} />
        <button type="button" className="icon" aria-label="Close references" onClick={close}><X size={14} /></button>
      </form>
      <p className="reference-hint">Up to {REFERENCE_LIMIT}, {count(REFERENCE_CHARACTERS)} characters total.</p>
      <div className="reference-list">
        {loading && <p className="reference-empty" role="status">Loading references…</p>}
        {error && <p className="reference-empty error" role="alert">{error}</p>}
        {!loading && !error && resources.length === 0 &&
          <p className="reference-empty">{query ? "No matches." : "Nothing saved yet."}</p>}
        {resources.map((item) => {
          const checked = selected.some((value) => value.resource_id === item.id);
          const blocked = !checked && full;
          const Icon = item.kind === "memory" ? Brain : FileText;
          return <label key={item.id} className={`reference-option${checked ? " checked" : ""}${blocked ? " blocked" : ""}`}>
            <input type="checkbox" checked={checked}
              disabled={disabled || loading || blocked}
              onChange={() => onChange(checked ? selected.filter((value) => value.resource_id !== item.id) :
                [...selected, { resource_id: item.id, expected_version: item.version, title: item.title,
                  characters: item.markdown.length }])} />
            <span className="reference-check">{checked && <Check size={12} />}</span>
            <Icon size={14} className="reference-kind" />
            <span className="reference-option-title">{item.title || "Untitled"}</span>
            <span className="reference-option-meta">
              {item.kind === "memory" ? "Memory" : "Space"} · v{item.version} · {count(item.markdown.length)} chars
            </span>
          </label>;
        })}
      </div>
      {truncated && <p className="reference-hint">Showing the first 200 of each. Search to narrow.</p>}
    </div>}
  </div>;
}
