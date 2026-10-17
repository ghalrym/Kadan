import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, RotateCw } from "lucide-react";
import Markdown from "./Markdown";
import Composer, { readiness } from "./Composer";
import Pet, { agentPet, useMood } from "./Pet";
import { agentPath, api, streamChat } from "./api";
import type { Resource, Status } from "./types";
import { randomUuid } from "./uuid";

/** The first message of a thread started from the agent's home screen. */
export interface Opening {
  content: string;
  persona: string;
  generate: boolean;
}

interface Props {
  agentKey: string;
  agentName: string;
  thread: Resource;
  personas: Resource[];
  status: Status | null;
  opening: Opening | null;
  onOpened: () => void;
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
const ROLE = { user: "You", assistant: "Assistant", system: "System", tool: "Tool" } as const;
const clock = (value: string) =>
  new Date(value).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

export default function Chat({
  agentKey,
  agentName,
  thread,
  personas,
  status,
  opening,
  onOpened,
  onChange,
  onBusy,
  blocked,
  refreshStatus,
}: Props) {
  const draftKey = `kadan.draft.${agentKey}.${thread.id}`;
  const pendingKey = `${draftKey}.pending`;
  const pending = useRef<PendingMessage | null>(pendingMessage(pendingKey));
  const [draft, setDraft] = useState(
    () => sessionStorage.getItem(draftKey) || "",
  );
  const [persona, setPersona] = useState(opening?.persona ?? "");
  const [sending, setSending] = useState(false);
  const [partial, setPartial] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const controller = useRef<AbortController | null>(null);
  const opened = useRef(false);
  const end = useRef<HTMLDivElement>(null);
  const ready = readiness(status);
  const [mood, react] = useMood(
    !status
      ? "idle"
      : sending
        ? "running"
        : !ready.chat
          ? "waiting"
          : draft.trim()
            ? "review"
            : "idle",
  );
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [thread.messages.length, partial]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (draft) sessionStorage.setItem(draftKey, draft);
    else sessionStorage.removeItem(draftKey);
  }, [draft, draftKey]);
  useEffect(() => {
    if (opened.current) return;
    opened.current = true;
    if (!opening) return react("waving");
    sessionStorage.removeItem(`kadan.draft.${agentKey}.home`);
    onOpened();
    setDraft(opening.content);
    void send(opening.generate, opening.content, opening.persona);
    // Runs once per mounted thread; the ref guards StrictMode's re-run.
  }, []);
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
  async function send(generate: boolean, text = draft, personaId = persona) {
    if (!text.trim() || sending || blocked) return;
    setSending(true);
    onBusy(true);
    setError("");
    setNotice("");
    setPartial("");
    const currentDraft = text.trim();
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
        react("waving");
      } else {
        controller.current = new AbortController();
        await streamChat(
          `${path}/chat`,
          {
            content: currentDraft,
            expected_version: thread.version,
            persona_id: personaId || null,
            message_id: messageId,
          },
          controller.current.signal,
          (event) => {
            if (event.event === "user_saved") {
              reconcileSaved(event.data.conversation);
              setDraft("");
            }
            if (event.event === "delta")
              setPartial((text) => text + event.data.content);
            if (event.event === "done") {
              onChange(event.data.conversation);
              setPartial("");
              react("jumping", 2);
              if (event.data.finish_reason === "length")
                setNotice("Stopped at Hikari's response limit.");
            }
          },
        );
      }
    } catch (e) {
      const aborted = (e as Error).name === "AbortError";
      if (!aborted) react("failed", 2);
      setError(aborted ? "Stopped. Reloading saved messages…" : (e as Error).message);
      try {
        const saved = await api<Resource>(
          `${agentPath(agentKey)}/resources/${thread.id}`,
        );
        reconcileSaved(saved);
        if (aborted) setError("Stopped. Saved messages reloaded.");
      } catch {
        /* Keep the already confirmed messages visible. */
      }
      void refreshStatus();
    } finally {
      setSending(false);
      onBusy(false);
      controller.current = null;
    }
  }
  function reload() {
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
  }
  return (
    <section className="chat" aria-label="Conversation">
      <div className="messages">
        <div className="messages-inner">
          {thread.messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <h3 className={message.role === "user" || message.role === "assistant" ? "sr-only" : "role"}>
                {ROLE[message.role]}
              </h3>
              <div className="body">
                <Markdown>{message.content}</Markdown>
              </div>
              <footer>
                <time dateTime={message.created_at}>{clock(message.created_at)}</time>
                {message.resource_ids?.map((id) => (
                  <span className="tag" key={id}>
                    {personas.find((p) => p.id === id)?.title || `Resource ${id.slice(0, 8)}`}
                  </span>
                ))}
                {(message.attachment_ids?.length > 0 || message.sources?.length > 0) &&
                  (status?.tsuchi.browser_url ? (
                    <a className="tag" href={status.tsuchi.browser_url} target="_blank" rel="noreferrer">
                      Attachments in Tsuchi
                      <ArrowUpRight size={12} />
                    </a>
                  ) : (
                    <span className="tag">Attachments in Tsuchi</span>
                  ))}
              </footer>
            </article>
          ))}
          {partial && (
            <article className={`message assistant ${sending ? "streaming" : "unsaved"}`}>
              <h3 className="sr-only">Assistant</h3>
              <div className="body">
                <Markdown>{partial}</Markdown>
              </div>
              {!sending && (
                <footer>
                  <span className="tag warn">Not saved</span>
                </footer>
              )}
            </article>
          )}
          {sending && !partial && (
            <div className="thinking" role="status">
              <span className="sr-only">{ready.chat ? "Waiting for Hikari" : "Saving note"}</span>
              <i />
              <i />
              <i />
            </div>
          )}
          {notice && !sending && (
            <p className="notice" role="status">
              {notice}
            </p>
          )}
          <div ref={end} />
        </div>
      </div>
      {error && (
        <div className="banner inline" role="alert">
          {error}
          <button disabled={sending || blocked} onClick={reload}>
            <RotateCw size={14} />
            Reload
          </button>
        </div>
      )}
      <Composer
        agentName={agentName}
        value={draft}
        onChange={setDraft}
        onSend={(generate) => void send(generate)}
        onStop={controller.current ? () => controller.current?.abort() : undefined}
        sending={sending}
        blocked={blocked}
        status={status}
        personas={personas}
        persona={persona}
        onPersona={setPersona}
        pet={<Pet sheet={agentPet(agentKey)} state={mood} size={96} look className="hero-pet" />}
      />
    </section>
  );
}
