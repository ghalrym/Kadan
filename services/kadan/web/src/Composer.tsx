import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, ArrowUpRight, ChevronDown, NotebookPen, Plug, Square, UserRound } from "lucide-react";
import { api } from "./api";
import type { Plugin, Resource, Status } from "./types";

export function readiness(status: Status | null) {
  const storage = status?.tsuchi.state === "ready";
  return { storage, chat: storage && status?.hikari.state === "ready" };
}

function attention(status: Status | null) {
  if (!status) return null;
  if (status.tsuchi.state !== "ready")
    return { label: "Storage offline", detail: status.tsuchi.message, href: status.tsuchi.browser_url };
  const hikari = status.hikari;
  if (hikari.state === "ready") return null;
  return {
    label:
      hikari.state === "tier_required"
        ? "Choose a tier · notes only"
        : hikari.state === "model_required"
          ? "Download a model · notes only"
          : "Hikari offline · notes only",
    detail: hikari.message,
    href: hikari.settings_url,
  };
}

function storedDisabledPlugins(key: string): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(value) && value.every((id) => typeof id === "string") ? value : [];
  } catch {
    return [];
  }
}

export function usePlugins(key: string, initialIds?: string[]) {
  const [items, setItems] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [retry, setRetry] = useState(0);
  const [disabledIds, setDisabledIds] = useState(() => storedDisabledPlugins(key));
  const initial = useRef(initialIds);
  function remember(ids: string[]) {
    setDisabledIds(ids);
    try {
      localStorage.setItem(key, JSON.stringify(ids));
      setNotice("");
    } catch {
      setNotice("Plugin choice applies now, but this browser could not save it for later.");
    }
  }
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    api<Plugin[]>("/plugins")
      .then((value) => {
        if (cancelled) return;
        setItems(value);
        if (initial.current) {
          remember(value.filter((item) => !initial.current!.includes(item.id)).map((item) => item.id));
          initial.current = undefined;
        }
      })
      .catch((e: Error) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [key, retry]);
  return {
    items, loading, error, notice,
    ready: !loading && !error,
    ids: items.filter((item) => item.enabled && !disabledIds.includes(item.id)).map((item) => item.id),
    reload: () => setRetry((value) => value + 1),
    toggle: (id: string) => remember(disabledIds.includes(id) ? disabledIds.filter((value) => value !== id) : [...disabledIds, id]),
  };
}

interface Props {
  agentName: string;
  value: string;
  onChange: (value: string) => void;
  onSend: (generate: boolean) => void;
  onStop?: () => void;
  sending: boolean;
  blocked: boolean;
  status: Status | null;
  personas: Resource[];
  persona: string;
  onPersona: (id: string) => void;
  pet: ReactNode;
  plugins: ReturnType<typeof usePlugins>;
  autoFocus?: boolean;
}

export default function Composer({
  agentName,
  value,
  onChange,
  onSend,
  onStop,
  sending,
  blocked,
  status,
  personas,
  persona,
  onPersona,
  pet,
  plugins,
  autoFocus,
}: Props) {
  const field = useRef<HTMLTextAreaElement>(null);
  const [pluginsOpen, setPluginsOpen] = useState(false);
  const pluginOptionsId = useId();
  const ready = readiness(status);
  const issue = attention(status);
  const empty = !value.trim();
  const locked = sending || blocked;
  useLayoutEffect(() => {
    const node = field.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 240)}px`;
  }, [value]);

  return (
    <div className="composer-wrap">
      {issue && (
        <a
          className="attention"
          href={issue.href ?? undefined}
          aria-disabled={!issue.href}
          target="_blank"
          rel="noreferrer"
          title={issue.detail}
        >
          <span className="dot warn" />
          {issue.label}
          {issue.href && <ArrowUpRight size={13} />}
        </a>
      )}
      <form
        className={`composer ${sending ? "is-sending" : ""}`}
        onSubmit={(event) => {
          event.preventDefault();
          if (!ready.chat || plugins.ready) onSend(ready.chat);
        }}
      >
        <span className="perch">{pet}</span>
        <label className="sr-only" htmlFor="message">
          Message
        </label>
        <textarea
          id="message"
          ref={field}
          autoFocus={autoFocus}
          placeholder={ready.chat ? `Message ${agentName}` : "Write a note"}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          maxLength={16000}
          rows={1}
          disabled={locked || !ready.storage}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (!ready.chat || plugins.ready) onSend(ready.chat);
            }
          }}
        />
        {ready.chat && (
          <div className="plugin-controls">
            <button
              className="chip"
              type="button"
              aria-expanded={pluginsOpen}
              aria-controls={pluginOptionsId}
              onClick={() => setPluginsOpen((open) => !open)}
            >
              <Plug size={14} />
              {plugins.loading ? "Loading plugins…" : plugins.error ? "Plugins unavailable" : `Plugins · ${plugins.ids.length}`}
              <ChevronDown size={13} />
            </button>
            {plugins.error && (
              <div className="plugin-error" role="alert">
                <span>{plugins.error} Reload plugins to send a message.</span>
                <button type="button" disabled={locked} onClick={plugins.reload}>Reload plugins</button>
              </div>
            )}
            {plugins.notice && <p className="field-hint" role="status">{plugins.notice}</p>}
            {pluginsOpen && (
              <div id={pluginOptionsId} className="plugin-options">
                <p className="field-hint">Selected plugins can run tools without confirmation.</p>
                {plugins.loading && <p className="field-hint" role="status">Loading plugins…</p>}
                {!plugins.loading && !plugins.error && !plugins.items.length && <p className="field-hint">No plugins configured.</p>}
                {plugins.items.map((plugin) => (
                  <div className="plugin-option" key={plugin.id}>
                    <label>
                      <input type="checkbox" checked={plugins.ids.includes(plugin.id)} disabled={locked || !plugin.enabled || !plugins.ready} onChange={() => plugins.toggle(plugin.id)} />
                      <span><strong>{plugin.name}</strong><span className="field-hint">{plugin.description}</span></span>
                      {!plugin.enabled && <span className="field-hint">Disabled on server</span>}
                    </label>
                    {(plugin.skills.length > 0 || plugin.setup.instructions) && (
                      <details className="plugin-details">
                        <summary>Plugin details</summary>
                        {plugin.setup.instructions && <p>{plugin.setup.instructions}</p>}
                        {plugin.skills.map((skill) => (
                          <details key={skill.name}>
                            <summary>{skill.name}</summary>
                            <p>{skill.description}</p>
                            <pre>{skill.instructions}</pre>
                          </details>
                        ))}
                      </details>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="composer-bar">
          {ready.chat && personas.length > 0 ? (
            <label className={`chip persona ${persona ? "active" : ""}`}>
              <UserRound size={14} />
              <span className="sr-only">Persona for next response</span>
              <select
                value={persona}
                onChange={(event) => onPersona(event.target.value)}
                disabled={locked}
              >
                <option value="">No persona</option>
                {personas.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title}
                  </option>
                ))}
              </select>
              <ChevronDown size={13} />
            </label>
          ) : (
            <span />
          )}
          <div className="send-actions">
            {ready.chat && (
              <button
                className="icon note"
                type="button"
                aria-label="Save as note without a reply"
                title="Save as note · no reply"
                onClick={() => onSend(false)}
                disabled={locked || empty}
              >
                <NotebookPen size={16} />
              </button>
            )}
            {sending && onStop ? (
              <button
                className="send stop"
                type="button"
                aria-label="Stop response"
                title="Stop"
                onClick={onStop}
              >
                <Square size={13} fill="currentColor" />
              </button>
            ) : (
              <button
                className="send"
                disabled={locked || empty || !ready.storage || (ready.chat && !plugins.ready)}
                aria-label={ready.chat ? "Send message" : "Save note"}
                title={ready.chat ? "Send · Enter" : "Save note · Enter"}
              >
                {ready.chat ? <ArrowUp size={18} /> : <NotebookPen size={16} />}
              </button>
            )}
          </div>
        </div>
      </form>
    </div>
  );
}
