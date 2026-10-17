import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  ExternalLink,
  FileText,
  MessageSquare,
  Plug,
  Square,
} from "lucide-react";
import Markdown from "./Markdown";
import ToolActivity from "./ToolActivity";
import { conversationActivity } from "./toolActivity";
import { agentPath, api, streamChat } from "./api";
import type { Plugin, Resource, Status } from "./types";
import { randomUuid } from "./uuid";

interface Props {
  agentKey: string;
  thread: Resource;
  personas: Resource[];
  status: Status | null;
  onChange: (value: Resource) => void;
  onBusy: (value: boolean) => void;
  blocked: boolean;
  refreshStatus: () => Promise<void>;
}
interface PendingMessage {
  id: string;
  content: string;
}
function pendingMessage(key: string): PendingMessage | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || "null");
    return value &&
      typeof value.id === "string" &&
      typeof value.content === "string"
      ? value
      : null;
  } catch {
    return null;
  }
}
function disabledPlugins(key: string): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(value) && value.every((id) => typeof id === "string")
      ? value
      : [];
  } catch {
    return [];
  }
}
export default function Chat({
  agentKey,
  thread,
  personas,
  status,
  onChange,
  onBusy,
  blocked,
  refreshStatus,
}: Props) {
  const draftKey = `kadan.draft.${agentKey}.${thread.id}`;
  const pendingKey = `${draftKey}.pending`;
  const pluginsKey = `kadan.plugins.${agentKey}.${thread.id}.disabled`;
  const pending = useRef<PendingMessage | null>(pendingMessage(pendingKey));
  const [draft, setDraft] = useState(
    () => sessionStorage.getItem(draftKey) || "",
  );
  const [persona, setPersona] = useState("");
  const [sending, setSending] = useState(false);
  const [partial, setPartial] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [streamStatus, setStreamStatus] = useState("");
  const [activeTools, setActiveTools] = useState<Set<string>>(new Set());
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [pluginsLoading, setPluginsLoading] = useState(true);
  const [pluginsError, setPluginsError] = useState("");
  const [pluginsOpen, setPluginsOpen] = useState(false);
  const [pluginsRetry, setPluginsRetry] = useState(0);
  const [disabledPluginIds, setDisabledPluginIds] = useState(() =>
    disabledPlugins(pluginsKey),
  );
  const controller = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const ready =
    status?.hikari.state === "ready" && status.tsuchi.state === "ready";
  const storageReady = status?.tsuchi.state === "ready";
  const pluginIds = plugins
    .filter(
      (plugin) => plugin.enabled && !disabledPluginIds.includes(plugin.id),
    )
    .map((plugin) => plugin.id);
  const pluginsReady = !pluginsLoading && !pluginsError;
  const activities = conversationActivity(thread.messages);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [thread.messages.length, partial]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    let cancelled = false;
    setPluginsLoading(true);
    setPluginsError("");
    api<Plugin[]>("/plugins")
      .then((value) => {
        if (!cancelled) setPlugins(value);
      })
      .catch((e: Error) => {
        if (!cancelled) setPluginsError(e.message);
      })
      .finally(() => {
        if (!cancelled) setPluginsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [pluginsRetry]);
  function togglePlugin(id: string) {
    const next = disabledPluginIds.includes(id)
      ? disabledPluginIds.filter((value) => value !== id)
      : [...disabledPluginIds, id];
    setDisabledPluginIds(next);
    try {
      localStorage.setItem(pluginsKey, JSON.stringify(next));
    } catch {
      setNotice(
        "Plugin choice applies now, but this browser could not save it for later.",
      );
    }
  }
  useEffect(() => {
    if (draft) sessionStorage.setItem(draftKey, draft);
    else sessionStorage.removeItem(draftKey);
  }, [draft, draftKey]);
  function reconcileSaved(value: Resource) {
    const confirmed = pending.current;
    if (
      confirmed &&
      value.messages.some((message) => message.id === confirmed.id)
    ) {
      setDraft((current) =>
        current.trim() === confirmed.content ? "" : current,
      );
      pending.current = null;
      sessionStorage.removeItem(pendingKey);
    }
    onChange(value);
  }
  async function send(generate: boolean) {
    if (!draft.trim() || sending || blocked || (generate && !pluginsReady))
      return;
    setSending(true);
    onBusy(true);
    setError("");
    setNotice("");
    setPartial("");
    setStreamStatus(generate ? "Waiting for Hikari…" : "Saving note…");
    setActiveTools(new Set());
    const currentDraft = draft.trim();
    const messageId =
      pending.current?.content === currentDraft
        ? pending.current.id
        : randomUuid();
    pending.current = { id: messageId, content: currentDraft };
    sessionStorage.setItem(pendingKey, JSON.stringify(pending.current));
    const path = `${agentPath(agentKey)}/conversations/${thread.id}`;
    try {
      if (!generate) {
        const saved = await api<Resource>(`${path}/notes`, {
          content: currentDraft,
          expected_version: thread.version,
          message_id: messageId,
        });
        reconcileSaved(saved);
        setDraft("");
        setNotice("Note saved. No response requested.");
      } else {
        controller.current = new AbortController();
        await streamChat(
          `${path}/chat`,
          {
            content: currentDraft,
            expected_version: thread.version,
            persona_id: persona || null,
            message_id: messageId,
            plugin_ids: pluginIds,
          },
          controller.current.signal,
          (event) => {
            if (event.event === "user_saved") {
              reconcileSaved(event.data.conversation);
              setDraft("");
            }
            if (event.event === "delta")
              setPartial((text) => text + event.data.content);
            if (event.event === "status") setStreamStatus(event.data.message);
            if (
              event.event === "tool_started" ||
              event.event === "tool_finished"
            ) {
              onChange(event.data.conversation);
              setPartial("");
              setActiveTools((current) => {
                const next = new Set(current);
                if (event.event === "tool_started")
                  next.add(event.data.execution.id);
                else next.delete(event.data.execution.id);
                return next;
              });
            }
            if (event.event === "done") {
              onChange(event.data.conversation);
              setPartial("");
              setNotice(
                event.data.finish_reason === "length"
                  ? "Response saved. Hikari reached its response limit."
                  : "Response saved.",
              );
            }
          },
        );
      }
    } catch (e) {
      setError(
        (e as Error).name === "AbortError"
          ? "Response stopped. Reloading saved messages…"
          : (e as Error).message,
      );
      try {
        const saved = await api<Resource>(
          `${agentPath(agentKey)}/resources/${thread.id}`,
        );
        reconcileSaved(saved);
        if ((e as Error).name === "AbortError")
          setError("Response stopped. Saved messages reloaded.");
      } catch {
        /* Keep the already confirmed messages visible. */
      }
      void refreshStatus();
    } finally {
      setSending(false);
      setActiveTools(new Set());
      onBusy(false);
      controller.current = null;
    }
  }
  return (
    <section className="chat" aria-label="Conversation">
      <div className="messages">
        {!thread.messages.length && (
          <div className="conversation-empty">
            <MessageSquare size={26} />
            <h2>New thread</h2>
          </div>
        )}
        {activities.map((activity) => {
          const message = activity.message;
          if (activity.kind === "tool")
            return (
              <ToolActivity
                key={activity.execution?.id || message.id}
                execution={activity.execution}
                active={Boolean(
                  activity.execution && activeTools.has(activity.execution.id),
                )}
              />
            );
          return (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="message-meta">
                <strong>
                  {message.role === "user"
                    ? "You"
                    : message.role === "assistant"
                      ? "Assistant"
                      : message.role === "system"
                        ? "System"
                        : "Tool"}
                </strong>
                <time dateTime={message.created_at}>
                  {new Date(message.created_at).toLocaleTimeString(undefined, {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </time>
              </div>
              <Markdown>{message.content}</Markdown>
              {message.resource_ids?.length > 0 && (
                <div className="message-refs">
                  {message.resource_ids.map((id) => (
                    <span key={id}>
                      {personas.find((p) => p.id === id)?.title ||
                        `Resource ${id.slice(0, 8)}`}
                    </span>
                  ))}
                </div>
              )}
              {(message.attachment_ids?.length > 0 ||
                message.sources?.length > 0) && (
                <p className="field-hint">
                  This stored message has attachments or source links. View them
                  in Tsuchi.
                </p>
              )}
            </article>
          );
        })}
        {partial && (
          <article className="message assistant">
            <div className="message-meta">
              <strong>Assistant</strong>
              <span>
                {sending ? "Responding…" : "Partial response · not saved"}
              </span>
            </div>
            <Markdown>{partial}</Markdown>
          </article>
        )}
        {sending && !partial && (
          <p className="stream-status" role="status">
            {streamStatus}
          </p>
        )}
        <div ref={end} />
      </div>
      <div className="composer-wrap">
        {!ready && (
          <div className="inference-state">
            <div>
              <span className="status-dot" />
              <strong>
                {!status
                  ? "Checking services"
                  : !storageReady
                    ? "Storage unavailable"
                    : "Chat unavailable"}
              </strong>
            </div>
            <p>
              {!storageReady ? status?.tsuchi.message : status?.hikari.message}
            </p>
            {(!storageReady
              ? status?.tsuchi.browser_url
              : status?.hikari.settings_url) && (
              <a
                href={
                  (!storageReady
                    ? status?.tsuchi.browser_url
                    : status?.hikari.settings_url) ?? undefined
                }
                target="_blank"
                rel="noreferrer"
              >
                {!storageReady ? "Open Tsuchi" : "Open Hikari Settings"}
                <ExternalLink size={13} />
              </a>
            )}
          </div>
        )}
        {error && (
          <div className="error feedback" role="alert">
            {error}
            <button
              disabled={sending || blocked}
              onClick={() => {
                setSending(true);
                onBusy(true);
                api<Resource>(`${agentPath(agentKey)}/resources/${thread.id}`)
                  .then((value) => {
                    reconcileSaved(value);
                    setPartial("");
                    setError("");
                  })
                  .catch((e) => setError(e.message))
                  .finally(() => {
                    setSending(false);
                    onBusy(false);
                  });
              }}
            >
              Reload thread
            </button>
          </div>
        )}
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void send(Boolean(ready));
          }}
        >
          <label className="sr-only" htmlFor="message">
            Message
          </label>
          <textarea
            id="message"
            placeholder={ready ? "Message your agent…" : "Write a note…"}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            maxLength={16000}
            rows={3}
            disabled={sending || blocked || !storageReady}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                void send(Boolean(ready));
              }
            }}
          />
          <div className="plugin-controls">
            <button
              className="quiet-button"
              type="button"
              aria-expanded={pluginsOpen}
              aria-controls="plugin-options"
              onClick={() => setPluginsOpen((open) => !open)}
            >
              <Plug size={14} />
              {pluginsLoading
                ? "Loading plugins…"
                : pluginsError
                  ? "Plugins unavailable"
                  : `Plugins · ${pluginIds.length}`}
            </button>
            {pluginsError && (
              <div className="feedback error" role="alert">
                <span>{pluginsError} Reload plugins to send a message.</span>
                <button
                  type="button"
                  disabled={sending || blocked}
                  onClick={() => setPluginsRetry((value) => value + 1)}
                >
                  Reload plugins
                </button>
              </div>
            )}
            {pluginsOpen && (
              <div id="plugin-options" className="plugin-options">
                <p className="field-hint">
                  Selected plugins have full access to their tools. Calls run
                  without confirmation.
                </p>
                {pluginsLoading && (
                  <p className="field-hint" role="status">
                    Loading plugins…
                  </p>
                )}
                {!pluginsLoading && !pluginsError && !plugins.length && (
                  <p className="field-hint">No plugins configured.</p>
                )}
                {plugins.map((plugin) => (
                  <div className="plugin-option" key={plugin.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={
                          plugin.enabled &&
                          !disabledPluginIds.includes(plugin.id)
                        }
                        disabled={
                          sending || blocked || !plugin.enabled || !pluginsReady
                        }
                        onChange={() => togglePlugin(plugin.id)}
                      />
                      <span>
                        <strong>{plugin.name}</strong>
                        <span className="field-hint">{plugin.description}</span>
                      </span>
                      {!plugin.enabled && (
                        <span className="field-hint">Disabled on server</span>
                      )}
                    </label>
                    {(plugin.skills.length > 0 ||
                      plugin.setup.instructions) && (
                      <details className="plugin-details">
                        <summary>Plugin details</summary>
                        {plugin.setup.instructions && (
                          <p>{plugin.setup.instructions}</p>
                        )}
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
          <div className="composer-actions">
            {ready ? (
              <label className="persona-select">
                <SettingsIcon />
                <span className="sr-only">Persona for next response</span>
                <select
                  aria-label="Persona for next response"
                  value={persona}
                  onChange={(event) => setPersona(event.target.value)}
                  disabled={sending || blocked}
                >
                  <option value="">No persona</option>
                  {personas.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.title}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <span />
            )}
            <div className="send-actions">
              {ready && (
                <button
                  className="quiet-button"
                  type="button"
                  onClick={() => void send(false)}
                  disabled={sending || blocked || !draft.trim()}
                >
                  Save note
                </button>
              )}
              {sending && controller.current ? (
                <button
                  className="icon stop"
                  type="button"
                  aria-label="Stop response"
                  onClick={() => controller.current?.abort()}
                >
                  <Square size={15} />
                </button>
              ) : (
                <button
                  className="primary send"
                  disabled={
                    sending ||
                    blocked ||
                    !draft.trim() ||
                    !storageReady ||
                    (ready && !pluginsReady)
                  }
                  aria-label={ready ? "Send message" : "Save note"}
                >
                  {ready ? (
                    <ArrowUp size={18} />
                  ) : (
                    <>
                      <FileText size={15} />
                      Save note
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </form>
        <div className="composer-foot" role="status">
          {notice ||
            (draft
              ? "Unsent draft · kept in this browser tab"
              : ready
                ? pluginIds.length
                  ? "Selected plugins run tools without confirmation."
                  : "No plugins selected. Persona and thread messages are sent to Hikari."
                : "")}
        </div>
      </div>
    </section>
  );
}
function SettingsIcon() {
  return <span className="persona-indicator" aria-hidden="true" />;
}
