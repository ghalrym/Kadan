import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowUpRight, RotateCw } from "lucide-react";
import Markdown from "./Markdown";
import Composer, { readiness } from "./Composer";
import Pet, { useMood } from "./Pet";
import { useAgentPet } from "./PetContext";
import { agentPath, api, streamChat } from "./api";
import type { PetState } from "./pet";
import type { Message, Resource, Status } from "./types";
import { randomUuid } from "./uuid";

interface Props {
  agentKey: string;
  agentName: string;
  chain: Resource;
  personas: Resource[];
  status: Status | null;
  entrance: PetState;
  onChange: (value: Resource) => void;
  onBusy: (value: boolean) => void;
  onEditPet: () => void;
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

/** Messages further apart than this get a timestamp between them. */
const BREAK_MS = 4 * 60 * 60 * 1000;
/** Older messages render on request so a long chain opens quickly. */
const PAGE = 200;
const ROLE = { user: "You", assistant: "Assistant", system: "System", tool: "Tool" } as const;

function breakLabel(value: string) {
  const date = new Date(value);
  const now = new Date();
  const time = date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const days = Math.round(
    (new Date(now).setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 86400000,
  );
  if (days === 0) return `Today ${time}`;
  if (days === 1) return `Yesterday ${time}`;
  if (days < 7) return `${date.toLocaleDateString(undefined, { weekday: "long" })} ${time}`;
  return `${date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  })} at ${time}`;
}
const stamp = (value: string) =>
  new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

export default function Chat({
  agentKey,
  agentName,
  chain,
  personas,
  status,
  entrance,
  onChange,
  onBusy,
  onEditPet,
  blocked,
  refreshStatus,
}: Props) {
  const petSheet = useAgentPet(agentKey);
  const draftKey = `kadan.draft.${agentKey}`;
  const pendingKey = `${draftKey}.pending`;
  const pending = useRef<PendingMessage | null>(pendingMessage(pendingKey));
  const [draft, setDraft] = useState(() => {
    const saved = sessionStorage.getItem(draftKey) || "";
    const unsent = pending.current;
    // A send interrupted by a reload returns to the composer unless it was saved.
    return !saved && unsent && !chain.messages.some((message) => message.id === unsent.id)
      ? unsent.content
      : saved;
  });
  const [persona, setPersona] = useState("");
  const [sending, setSending] = useState(false);
  const [generating, setGenerating] = useState(false);
  /** The message being sent, shown before Tsuchi confirms it. */
  const [outgoing, setOutgoing] = useState<Message | null>(null);
  const [partial, setPartial] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [shown, setShown] = useState(PAGE);
  // Tool activity saved by the retired plugin system is history the model no longer sees.
  const saved = chain.messages.filter((message) => message.role !== "tool");
  const controller = useRef<AbortController | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const ready = readiness(status);
  // Hikari loads the model on demand; the first reply waits for it without a deadline.
  const modelLoading = generating && status?.hikari.model_state === "loading";
  // The agent has read the message once Hikari is working on it, and types until the reply lands.
  const typing = generating && !outgoing && !modelLoading;
  const [mood, react] = useMood(
    !status
      ? "idle"
      : modelLoading
        ? "waiting"
        : typing
          ? "running"
          : !ready.chat
            ? "waiting"
            : draft.trim()
              ? "review"
              : "idle",
  );
  const items = outgoing ? [...saved, outgoing] : saved;
  const visible = items.slice(-shown);
  const hidden = items.length - visible.length;
  let lastUser = visible.length - 1;
  while (lastUser >= 0 && visible[lastUser].role !== "user") lastUser--;
  const answered = visible.slice(lastUser + 1).some((message) => message.role === "assistant");
  const receipt =
    lastUser < 0
      ? null
      : visible[lastUser] === outgoing
        ? "Sending…"
        : answered || typing
          ? "Read"
          : "Delivered";

  useLayoutEffect(() => {
    // Open at the newest message without animating through the history.
    end.current?.scrollIntoView({ block: "end" });
  }, []);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [items.length, typing]);
  useEffect(() => react(entrance, 2), [entrance, react]);
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      // StrictMode remounts at once in development; abort only when the chat really closes.
      setTimeout(() => {
        if (!mounted.current) controller.current?.abort();
      });
    };
  }, []);
  useEffect(() => {
    // Until the reply lands, check whether Hikari is still loading the model.
    if (!generating) return;
    const timer = setInterval(() => void refreshStatus(), 2000);
    void refreshStatus();
    return () => clearInterval(timer);
  }, [generating]);
  useEffect(() => {
    if (draft) sessionStorage.setItem(draftKey, draft);
    else sessionStorage.removeItem(draftKey);
  }, [draft, draftKey]);
  useEffect(() => {
    // A send from before a reload may have been saved after all.
    const confirmed = pending.current;
    if (confirmed && chain.messages.some((message) => message.id === confirmed.id)) {
      pending.current = null;
      sessionStorage.removeItem(pendingKey);
    }
  }, []);

  function accept(value: Resource) {
    const confirmed = pending.current;
    if (confirmed && value.messages.some((message) => message.id === confirmed.id)) {
      pending.current = null;
      sessionStorage.removeItem(pendingKey);
      setOutgoing(null);
    }
    onChange(value);
  }
  async function send(generate: boolean) {
    const text = draft.trim();
    if (!text || sending || blocked) return;
    setSending(true);
    setGenerating(generate);
    onBusy(true);
    setError("");
    setNotice("");
    setPartial("");
    const messageId = pending.current?.content === text ? pending.current.id : randomUuid();
    pending.current = { id: messageId, content: text };
    sessionStorage.setItem(pendingKey, JSON.stringify(pending.current));
    setOutgoing({
      id: messageId,
      role: "user",
      content: text,
      created_at: new Date().toISOString(),
      resource_ids: generate && persona ? [persona] : [],
      attachment_ids: [],
      sources: [],
    });
    setDraft("");
    const path = `${agentPath(agentKey)}/conversations/${chain.id}`;
    let reply = "";
    try {
      if (!generate) {
        accept(
          await api<Resource>(`${path}/notes`, {
            content: text,
            expected_version: chain.version,
            message_id: messageId,
          }),
        );
        react("waving");
      } else {
        controller.current = new AbortController();
        await streamChat(
          `${path}/chat`,
          {
            content: text,
            expected_version: chain.version,
            persona_id: persona || null,
            message_id: messageId,
          },
          controller.current.signal,
          (event) => {
            if (event.event === "user_saved") accept(event.data.conversation);
            // Like a text, the reply arrives whole; the agent shows as typing until then.
            if (event.event === "delta") reply += event.data.content;
            if (event.event === "done") {
              onChange(event.data.conversation);
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
      setPartial(reply);
      setError(aborted ? "Stopped. Reloading saved messages…" : (e as Error).message);
      try {
        accept(await api<Resource>(`${agentPath(agentKey)}/resources/${chain.id}`));
        if (aborted) setError("Stopped. Saved messages reloaded.");
      } catch {
        /* Keep the already confirmed messages visible. */
      }
      if (pending.current?.id === messageId) {
        // Not delivered: hand the text back so it can be sent again.
        setOutgoing(null);
        setDraft((current) => (current.trim() ? current : text));
      }
      void refreshStatus();
    } finally {
      setSending(false);
      setGenerating(false);
      onBusy(false);
      controller.current = null;
    }
  }
  function reload() {
    setSending(true);
    onBusy(true);
    api<Resource>(`${agentPath(agentKey)}/resources/${chain.id}`)
      .then((value) => {
        accept(value);
        setPartial("");
        setError("");
      })
      .catch((e) => setError(e.message))
      .finally(() => {
        setSending(false);
        onBusy(false);
      });
  }
  const perchStatus = modelLoading
    ? "Loading model…"
    : typing
      ? "Typing…"
      : !ready.chat
        ? "Notes only"
        : agentName;
  const empty = items.length === 0 && !partial;
  return (
    <section className="chat" aria-label={`Messages with ${agentName}`}>
      {!empty && (
        <div className={`lookout ${generating ? "working" : ""}`}>
          <button type="button" className="lookout-pet" aria-label={`Wave to ${agentName}`} title={`Wave to ${agentName}`}
            onClick={() => react("waving")}>
            <Pet sheet={petSheet} state={mood} size={96} look className="hero-pet" />
          </button>
          <span className="lookout-ledge" role="status" aria-live="polite">
            {generating && <i aria-hidden="true" />}
            {perchStatus}
          </span>
        </div>
      )}
      <div className="messages">
        <div className={`messages-inner ${empty ? "is-empty" : ""}`}>
          {empty ? (
            <div className="home-stage">
              <Pet sheet={petSheet} state={mood} size={192} look grounded className="hero-pet" />
              <h2>{agentName}</h2>
              <button className="pet-edit-link" onClick={onEditPet}>Edit pet</button>
            </div>
          ) : (
            hidden > 0 && (
              <button className="earlier" onClick={() => setShown((count) => count + PAGE)}>
                Show earlier messages
              </button>
            )
          )}
          {visible.map((message, index) => {
            const previous = visible[index - 1];
            const next = visible[index + 1];
            const gap = (a?: Message, b?: Message) =>
              !a || !b || new Date(b.created_at).getTime() - new Date(a.created_at).getTime() > BREAK_MS;
            const split = gap(previous, message);
            const first = split || previous?.role !== message.role;
            const last = gap(message, next) || next?.role !== message.role;
            const tags = message.resource_ids?.length > 0;
            const attached = message.attachment_ids?.length > 0 || message.sources?.length > 0;
            return (
              <Fragment key={message.id}>
                {split && (
                  <div className="time-break">
                    <time dateTime={message.created_at}>{breakLabel(message.created_at)}</time>
                  </div>
                )}
                <article
                  className={`message ${message.role} ${first ? "first" : ""} ${last ? "last" : ""} ${message === outgoing ? "sending" : ""}`}
                  title={stamp(message.created_at)}
                >
                  <h3 className={message.role === "user" || message.role === "assistant" ? "sr-only" : "role"}>
                    {ROLE[message.role]}
                  </h3>
                  <div className="body">
                    <Markdown>{message.content}</Markdown>
                  </div>
                  {(tags || attached) && (
                    <footer>
                      {message.resource_ids.map((id) => (
                        <span className="tag" key={id}>
                          {personas.find((p) => p.id === id)?.title || `Resource ${id.slice(0, 8)}`}
                        </span>
                      ))}
                      {attached &&
                        (status?.tsuchi.browser_url ? (
                          <a className="tag" href={status.tsuchi.browser_url} target="_blank" rel="noreferrer">
                            Attachments in Tsuchi
                            <ArrowUpRight size={12} />
                          </a>
                        ) : (
                          <span className="tag">Attachments in Tsuchi</span>
                        ))}
                    </footer>
                  )}
                </article>
                {index === lastUser && receipt && (
                  <p className={`receipt ${receipt === "Read" ? "read" : ""}`} aria-live="polite">
                    {receipt}
                  </p>
                )}
              </Fragment>
            );
          })}
          {partial && !sending && (
            <article className="message assistant first last unsaved">
              <h3 className="sr-only">Assistant</h3>
              <div className="body">
                <Markdown>{partial}</Markdown>
              </div>
              <footer>
                <span className="tag warn">Not saved</span>
              </footer>
            </article>
          )}
          {typing && (
            <div className="typing" role="status">
              <span className="sr-only">{agentName} is typing</span>
              <i />
              <i />
              <i />
            </div>
          )}
          {modelLoading && (
            <p className="notice" role="status">
              {agentName} is waking up. Loading the model can take a few minutes.
            </p>
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
        onStop={generating ? () => controller.current?.abort() : undefined}
        sending={sending}
        blocked={blocked}
        status={status}
        personas={personas}
        persona={persona}
        onPersona={setPersona}
        pet={null}
        autoFocus
      />
    </section>
  );
}
