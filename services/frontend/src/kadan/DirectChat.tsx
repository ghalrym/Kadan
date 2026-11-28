import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  CornerUpLeft,
  Mic,
  RotateCw,
  Search,
  Square,
  X,
} from "lucide-react";
import AgentCard from "./AgentCard";
import { api } from "./api";
import ChatExport from "./ChatExport";
import { readiness } from "./Composer";
import {
  formatLastSeen,
  isAwaitingReply,
  isParticipantActive,
  messageSnippet,
  participantStatusBadge,
  turnCapNotice,
  type GroupChatMessage,
  type GroupChatParticipant,
  type GroupChatSnapshot,
  type GroupChatUpdateParticipantResponse,
  type GroupChatUserMessageResponse,
} from "./groupChat";
import Markdown from "./Markdown";
import Pet, { useMood } from "./Pet";
import { usePetRefJob, usePetRefSheet } from "./PetContext";
import PetEditor from "./PetEditor";
import type { PetRef } from "./profile";
import type { Message, Resource, Status } from "./types";
import { randomUuid } from "./uuid";
import { microphoneUnavailable, VoiceInput, type VoiceState } from "./voiceInput";

interface Props {
  participantId: string;
  agentKey: string;
  status: Status | null;
  /** Every snapshot shown here; the sidebar uses it for members and unread state. */
  onSnapshot?: (snapshot: GroupChatSnapshot) => void;
}

/** Messages further apart than 30 minutes get a full-width timestamp divider bar. */
const BREAK_MS = 30 * 60 * 1000;
const NEAR_LATEST = 80;
const POLL_MS = 2000;
const DRAFT_KEY_PREFIX = "kadan:group-chat:draft:direct:";
const OUTBOX_KEY_PREFIX = "kadan:group-chat:pending:direct:";

interface PendingDirectMessage {
  client_request_id: string;
  content: string;
  handoff_to_participant_ids: string[];
  reply_to_message_id?: string | null;
}

function readPendingOutbox(key: string): PendingDirectMessage | null {
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PendingDirectMessage>;
    if (
      typeof parsed.client_request_id === "string" &&
      typeof parsed.content === "string" &&
      Array.isArray(parsed.handoff_to_participant_ids)
    ) {
      return {
        client_request_id: parsed.client_request_id,
        content: parsed.content,
        handoff_to_participant_ids: parsed.handoff_to_participant_ids.filter(
          (id): id is string => typeof id === "string",
        ),
        reply_to_message_id:
          typeof parsed.reply_to_message_id === "string" ? parsed.reply_to_message_id : null,
      };
    }
  } catch {
    // Ignore invalid stored state.
  }
  return null;
}

function breakLabel(value: string) {
  const date = new Date(value);
  const now = new Date();
  const time = date
    .toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true })
    .toLowerCase();
  const days = Math.round(
    (new Date(now).setHours(0, 0, 0, 0) - new Date(date).setHours(0, 0, 0, 0)) / 86400000,
  );
  if (days === 0) return time;
  if (days === 1) return `Yesterday · ${time}`;
  if (days < 7) return `${date.toLocaleDateString(undefined, { weekday: "long" })} · ${time}`;
  return `${date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  })} · ${time}`;
}

const stamp = (value: string) =>
  new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

function asExportResource(
  participantId: string,
  displayName: string,
  messages: GroupChatMessage[],
): Resource {
  const now = new Date().toISOString();
  const mapped: Message[] = messages.map((msg) => ({
    id: msg.id,
    role: msg.sender_kind === "user" ? "user" : "assistant",
    content: msg.content,
    resource_ids: [],
    attachment_ids: [],
    sources: [],
    created_at: msg.created_at,
  }));
  return {
    id: `direct-${participantId}`,
    agent_key: displayName.toLowerCase().replace(/[^a-z0-9]+/g, "-") || "agent",
    kind: "conversation",
    title: displayName,
    markdown: "",
    parent_id: null,
    tags: [],
    sources: [],
    messages: mapped,
    attachment_ids: [],
    version: 1,
    created_at: mapped[0]?.created_at ?? now,
    updated_at: mapped.at(-1)?.created_at ?? now,
    deleted_at: null,
  };
}

/** Private 1-on-1 chat with a connected external agent, matching Kadan's native Chat view. */
export default function DirectChat({
  participantId,
  agentKey,
  status,
  onSnapshot,
}: Props) {
  const draftKey = DRAFT_KEY_PREFIX + participantId;
  const outboxKey = OUTBOX_KEY_PREFIX + participantId;
  const basePath = `/group-chat/direct/${encodeURIComponent(participantId)}`;

  const [snapshot, setSnapshot] = useState<GroupChatSnapshot | null>(null);
  const [messages, setMessages] = useState<GroupChatMessage[]>([]);
  const [hasOlder, setHasOlder] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [error, setError] = useState("");
  const [draft, setDraft] = useState(() => {
    try {
      return sessionStorage.getItem(draftKey) || "";
    } catch {
      return "";
    }
  });
  const [pendingOutbox, setPendingOutbox] = useState<PendingDirectMessage | null>(() =>
    readPendingOutbox(outboxKey),
  );
  const [outgoing, setOutgoing] = useState<GroupChatMessage | null>(null);
  const [sending, setSending] = useState(false);
  const [replyTo, setReplyTo] = useState<GroupChatMessage | null>(null);
  const [card, setCard] = useState(false);
  const [petFor, setPetFor] = useState<GroupChatParticipant | null>(null);
  const [away, setAway] = useState(false);
  const [find, setFind] = useState({ open: false, query: "", messageId: "" });
  const [voice, setVoice] = useState<VoiceState | null>(null);
  const [voiceError, setVoiceError] = useState("");

  const activeRef = useRef(true);
  const listRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const findButton = useRef<HTMLButtonElement>(null);
  const messageElements = useRef(new Map<string, HTMLElement>());
  const followLatest = useRef(true);
  const jumpRequested = useRef(false);
  const microphone = useRef<VoiceInput | null>(null);
  const draftRef = useRef(draft);
  draftRef.current = draft;

  const ready = readiness(status);
  const storageReady = status?.tsuchi.state !== "unavailable";
  const participants = snapshot?.participants ?? [];
  const participantsById = new Map(participants.map((item) => [item.id, item]));
  const directAgent = participantsById.get(participantId) ?? null;
  const displayName = directAgent?.display_name ?? snapshot?.room.title ?? "Private chat";
  const agentSheet = usePetRefSheet(directAgent?.pet);
  const petForJob = usePetRefJob(petFor?.pet);

  const replier = directAgent && isParticipantActive(directAgent) ? directAgent : undefined;
  const replierReplying = !!replier && isAwaitingReply(messages, replier.id);
  const capBanner = snapshot ? turnCapNotice(snapshot.room) : null;
  const working = replierReplying || sending;

  const [mood, react] = useMood(
    replierReplying
      ? "running"
      : !storageReady
        ? "waiting"
        : draft.trim()
          ? "review"
          : "idle",
  );

  const items = outgoing ? [...messages, outgoing] : messages;
  const messagesById = new Map(items.map((item) => [item.id, item]));
  const query = find.open ? find.query.trim().toLowerCase() : "";
  const matches = query ? messages.filter((m) => m.content.toLowerCase().includes(query)) : [];
  const match = matches.find((m) => m.id === find.messageId) ?? matches[0];
  const matchIndex = match ? matches.indexOf(match) : -1;
  let lastUser = items.length - 1;
  while (lastUser >= 0 && items[lastUser].sender_kind !== "user") lastUser--;
  const receipt =
    lastUser < 0 ? null : items[lastUser] === outgoing ? "Sending…" : "Delivered";

  const exportResource = useMemo(
    () => asExportResource(participantId, displayName, messages),
    [participantId, displayName, messages],
  );

  const closeCard = useCallback(() => setCard(false), []);
  const scrollToLatest = useCallback(() => {
    const node = listRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, []);

  const mergeLatestSnapshot = useCallback(
    (next: GroupChatSnapshot) => {
      setSnapshot(next);
      onSnapshot?.(next);
      setMessages((prev) => {
        if (!prev.length) {
          setHasOlder(next.has_more);
          return next.messages;
        }
        const byId = new Map(prev.map((item) => [item.id, item]));
        for (const msg of next.messages) byId.set(msg.id, msg);
        return [...byId.values()].sort((a, b) => a.seq - b.seq);
      });
      setPendingOutbox((current) => {
        if (!current) return null;
        if (next.messages.some((m) => m.client_request_id === current.client_request_id)) {
          try {
            sessionStorage.removeItem(outboxKey);
          } catch {
            // Ignore storage error.
          }
          setOutgoing(null);
          return null;
        }
        return current;
      });
    },
    [onSnapshot, outboxKey],
  );

  const refresh = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      try {
        const next = await api<GroupChatSnapshot>(`${basePath}?limit=100`);
        if (!activeRef.current) return;
        mergeLatestSnapshot(next);
        setError("");
      } catch (e) {
        if (activeRef.current) {
          setError(`Could not load chat: ${(e as Error).message}`);
        }
      } finally {
        if (activeRef.current && !silent) setLoading(false);
      }
    },
    [mergeLatestSnapshot, basePath],
  );

  useEffect(() => {
    activeRef.current = true;
    void refresh(false);
    const timer = setInterval(() => void refresh(true), POLL_MS);
    return () => {
      activeRef.current = false;
      clearInterval(timer);
    };
  }, [refresh]);

  useEffect(() => () => microphone.current?.cancel(), []);
  useEffect(() => {
    if (sending || !ready.chat) {
      microphone.current?.cancel();
      microphone.current = null;
      setVoice(null);
    }
  }, [sending, ready.chat]);

  useEffect(() => {
    try {
      if (draft) sessionStorage.setItem(draftKey, draft);
      else sessionStorage.removeItem(draftKey);
    } catch {
      // Ignore storage error.
    }
  }, [draft, draftKey]);

  useLayoutEffect(() => {
    if (followLatest.current && !query) scrollToLatest();
  }, [items.length, replierReplying, query, scrollToLatest]);

  useLayoutEffect(() => {
    if (!jumpRequested.current) return;
    jumpRequested.current = false;
    scrollToLatest();
  }, [away, find.open, scrollToLatest]);

  useEffect(() => {
    if (!match) return;
    const node = messageElements.current.get(match.id);
    if (node) node.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [match?.id, find.query]);

  useLayoutEffect(() => {
    const node = composerRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 240)}px`;
  }, [draft]);

  async function loadOlderMessages() {
    if (loadingOlder || !messages.length) return;
    setLoadingOlder(true);
    try {
      const oldestSeq = messages[0].seq;
      const older = await api<GroupChatSnapshot>(
        `${basePath}?before_seq=${oldestSeq}&limit=50`,
      );
      if (!activeRef.current) return;
      setHasOlder(older.has_more);
      setMessages((prev) => {
        const byId = new Map(prev.map((m) => [m.id, m]));
        for (const m of older.messages) byId.set(m.id, m);
        return [...byId.values()].sort((a, b) => a.seq - b.seq);
      });
    } catch (e) {
      if (activeRef.current) setError((e as Error).message);
    } finally {
      if (activeRef.current) setLoadingOlder(false);
    }
  }

  async function submitMessage(payload: PendingDirectMessage) {
    if (sending) return;
    setSending(true);
    setError("");
    setPendingOutbox(payload);
    try {
      sessionStorage.setItem(outboxKey, JSON.stringify(payload));
    } catch {
      // Ignore storage error.
    }
    const optimistic: GroupChatMessage = {
      id: `pending-${payload.client_request_id}`,
      room_id: snapshot?.room.id ?? "",
      seq: (messages.at(-1)?.seq ?? 0) + 1,
      sender_kind: "user",
      participant_id: null,
      sender_name: "You",
      content: payload.content,
      handoff_to_participant_ids: payload.handoff_to_participant_ids,
      handoff_suppressed: false,
      reply_to_message_id: payload.reply_to_message_id ?? null,
      client_request_id: payload.client_request_id,
      created_at: new Date().toISOString(),
    };
    setOutgoing(optimistic);
    setDraft("");
    setReplyTo(null);
    followLatest.current = true;
    setAway(false);
    try {
      const result = await api<GroupChatUserMessageResponse>(
        `${basePath}/messages`,
        {
          content: payload.content,
          handoff_to_participant_ids: payload.handoff_to_participant_ids,
          client_request_id: payload.client_request_id,
          reply_to_message_id: payload.reply_to_message_id ?? null,
        },
        "POST",
      );
      if (!activeRef.current) return;
      try {
        sessionStorage.removeItem(outboxKey);
      } catch {
        // Ignore storage error.
      }
      setPendingOutbox(null);
      setOutgoing(null);
      setMessages((prev) => {
        const byId = new Map(prev.map((m) => [m.id, m]));
        byId.set(result.message.id, result.message);
        return [...byId.values()].sort((a, b) => a.seq - b.seq);
      });
      setSnapshot((prev) =>
        prev
          ? {
              ...prev,
              room: result.room,
              latest_seq: Math.max(prev.latest_seq, result.message.seq),
              turn_cap_reached: result.turn_cap_reached,
            }
          : prev,
      );
      react("waving");
    } catch (e) {
      if (activeRef.current) {
        react("failed", 2);
        setOutgoing(null);
        setError(`Could not send message: ${(e as Error).message}`);
      }
    } finally {
      if (activeRef.current) setSending(false);
    }
  }

  function handleSend(event?: FormEvent) {
    event?.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || sending || !storageReady || voice !== null) return;
    void submitMessage({
      client_request_id: randomUuid(),
      content: trimmed,
      handoff_to_participant_ids: [participantId],
      reply_to_message_id: replyTo?.id ?? null,
    });
  }

  async function selectParticipantPet(participant: GroupChatParticipant, petId: string | null) {
    const current = participantsById.get(participant.id) ?? participant;
    const change: { pet?: PetRef; clear_pet?: boolean } = petId
      ? { pet: { agent_key: agentKey, pet_id: petId } }
      : { clear_pet: true };
    try {
      const updated = await api<GroupChatUpdateParticipantResponse>(
        `/group-chat/participants/${encodeURIComponent(participant.id)}`,
        { ...change, expected_version: current.version },
        "PUT",
      );
      if (activeRef.current) setPetFor(updated.participant);
    } finally {
      await refresh(true);
    }
  }

  function recordVoice() {
    if (microphone.current) {
      microphone.current.finish();
      return;
    }
    setVoiceError("");
    const session = new VoiceInput(
      (text) => {
        const next = draftRef.current ? `${draftRef.current}\n${text}` : text;
        if (next.length > 16000) {
          throw new Error(
            "The transcript would exceed the message limit. Shorten the draft and try again.",
          );
        }
        setDraft(next);
      },
      setVoice,
      (err) => {
        microphone.current = null;
        setVoice(null);
        setVoiceError(err ?? "");
      },
    );
    microphone.current = session;
    void session.start();
  }

  function jumpToMessage(id: string) {
    const node = messageElements.current.get(id);
    if (!node) return;
    node.scrollIntoView({ block: "center", behavior: "smooth" });
    node.classList.add("flash");
    setTimeout(() => node.classList.remove("flash"), 1200);
  }

  function startReply(msg: GroupChatMessage) {
    setReplyTo(msg);
    requestAnimationFrame(() => composerRef.current?.focus());
  }

  function toggleCard() {
    if (!card) react("waving");
    setCard(!card);
  }

  function closeFind() {
    setFind({ open: false, query: "", messageId: "" });
    findButton.current?.focus();
  }

  function moveMatch(direction: number) {
    if (!matches.length) return;
    const index = (matchIndex + direction + matches.length) % matches.length;
    setFind((current) => ({ ...current, messageId: matches[index].id }));
  }

  function jumpToLatest() {
    setFind({ open: false, query: "", messageId: "" });
    followLatest.current = true;
    setAway(false);
    jumpRequested.current = true;
  }

  const canEditPet = !!directAgent && directAgent.revoked_at === null;
  const openPetEditor = () => {
    if (!directAgent || directAgent.revoked_at !== null) return;
    setCard(false);
    setPetFor(directAgent);
  };

  const perchStatus = replierReplying ? `${displayName} is replying…` : displayName;
  const cardPresence = replierReplying
    ? `${displayName} is replying…`
    : directAgent
      ? `${participantStatusBadge(directAgent).label} · ${formatLastSeen(directAgent.last_seen_at)}`
      : "External agent";

  const agentCard = card && (
    <AgentCard
      agentName={displayName}
      sheet={agentSheet}
      mood={mood}
      presence={cardPresence}
      onClose={closeCard}
      onEditPet={canEditPet ? openPetEditor : undefined}
      showCall={false}
      note={
        directAgent?.description ||
        `Private chat with ${displayName}. Only you and ${displayName} can see these messages.`
      }
    />
  );

  const empty = !loading && items.length === 0;
  const locked = sending || voice !== null;

  return (
    <section
      className={`chat ${find.open ? "finding" : ""}`}
      aria-label={`Chat with ${displayName}`}
    >
      {!empty && !loading && !find.open && (
        <div className={`lookout ${working ? "working" : ""}`}>
          <button
            type="button"
            className="lookout-pet"
            aria-label={`Open ${displayName}’s card`}
            title={displayName}
            aria-expanded={card}
            aria-haspopup="dialog"
            onClick={toggleCard}
          >
            <Pet sheet={agentSheet} state={mood} size={96} look className="hero-pet" />
          </button>
          <span className="lookout-ledge" role="status" aria-live="polite">
            {working && <i aria-hidden="true" />}
            {perchStatus}
          </span>
          {agentCard}
        </div>
      )}

      <ChatExport chain={exportResource} agentName={displayName} />

      <div
        className={`conversation-find ${find.open ? "is-open" : ""}`}
        onKeyDown={(event) => {
          if (find.open && event.key === "Escape") {
            event.preventDefault();
            closeFind();
          }
        }}
      >
        <button
          ref={findButton}
          type="button"
          disabled={!messages.length && !find.open}
          aria-expanded={find.open}
          aria-label={find.open ? "Close find" : "Find in conversation"}
          title={find.open ? "Close find" : "Find in conversation"}
          onClick={() => (find.open ? closeFind() : setFind({ open: true, query: "", messageId: "" }))}
        >
          {find.open ? <X size={16} /> : <Search size={16} />}
        </button>
        {find.open && (
          <form
            role="search"
            aria-label="Find in conversation"
            onSubmit={(event) => {
              event.preventDefault();
              moveMatch(1);
            }}
          >
            <label className="search-field">
              <span className="sr-only">Find in saved text</span>
              <input
                type="search"
                autoFocus
                placeholder="Find in saved text"
                value={find.query}
                onChange={(event) =>
                  setFind((current) => ({
                    ...current,
                    query: event.target.value,
                    messageId: "",
                  }))
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter" && event.shiftKey) {
                    event.preventDefault();
                    moveMatch(-1);
                  }
                }}
              />
            </label>
            <span className="find-count" role="status" aria-live="polite">
              {query
                ? matches.length
                  ? `${matchIndex + 1} of ${matches.length} messages`
                  : "No matching messages"
                : "Saved text only"}
            </span>
            <button
              type="button"
              disabled={!matches.length}
              aria-label="Previous matching message"
              title="Previous (Shift+Enter)"
              onClick={() => moveMatch(-1)}
            >
              <ArrowUp size={16} />
            </button>
            <button
              type="submit"
              disabled={!matches.length}
              aria-label="Next matching message"
              title="Next (Enter)"
            >
              <ArrowDown size={16} />
            </button>
          </form>
        )}
      </div>

      <div
        className="messages"
        ref={listRef}
        role="region"
        aria-label="Conversation messages"
        tabIndex={-1}
        onScroll={(event) => {
          const element = event.currentTarget;
          const near = element.scrollHeight - element.clientHeight - element.scrollTop <= NEAR_LATEST;
          followLatest.current = near;
          setAway(!near);
        }}
      >
        <div className={`messages-inner ${empty || (loading && !items.length) ? "is-empty" : ""}`}>
          {loading && !items.length ? (
            <div className="home-stage" role="status">
              <span className="sr-only">Loading chat</span>
              <span className="loader" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
            </div>
          ) : empty ? (
            <div className="home-stage">
              <div className="home-pet">
                <button
                  type="button"
                  className="lookout-pet"
                  aria-label={`Open ${displayName}’s card`}
                  title={displayName}
                  aria-expanded={card}
                  aria-haspopup="dialog"
                  onClick={toggleCard}
                >
                  <Pet sheet={agentSheet} state={mood} size={192} look grounded className="hero-pet" />
                </button>
                {agentCard}
              </div>
              <h2>{displayName}</h2>
              <p className="direct-home-hint">
                Only you and {displayName} can see this chat.
              </p>
              {canEditPet && (
                <button type="button" className="pet-edit-link" onClick={openPetEditor}>
                  Edit pet
                </button>
              )}
            </div>
          ) : (
            hasOlder && (
              <button
                type="button"
                className="earlier"
                disabled={loadingOlder}
                onClick={() => void loadOlderMessages()}
              >
                {loadingOlder ? "Loading earlier messages…" : "Show earlier messages"}
              </button>
            )
          )}

          {items.map((msg, index) => {
            const role = msg.sender_kind === "user" ? "user" : "assistant";
            const senderName = role === "user" ? "You" : displayName;
            const previous = items[index - 1];
            const next = items[index + 1];
            const prevRole = previous ? (previous.sender_kind === "user" ? "user" : "assistant") : null;
            const nextRole = next ? (next.sender_kind === "user" ? "user" : "assistant") : null;
            const gap = (a?: GroupChatMessage, b?: GroupChatMessage) =>
              !a || !b || new Date(b.created_at).getTime() - new Date(a.created_at).getTime() > BREAK_MS;
            const split = gap(previous, msg);
            const first = split || prevRole !== role || !!msg.reply_to_message_id;
            const last = gap(msg, next) || nextRole !== role || !!next?.reply_to_message_id;
            const repliedTo = msg.reply_to_message_id ? messagesById.get(msg.reply_to_message_id) : null;
            const isPending = msg === outgoing;

            return (
              <Fragment key={msg.id}>
                {split && (
                  <div className="time-break" role="separator">
                    <time dateTime={msg.created_at}>{breakLabel(msg.created_at)}</time>
                  </div>
                )}
                <article
                  ref={(element) => {
                    if (element) messageElements.current.set(msg.id, element);
                    else messageElements.current.delete(msg.id);
                  }}
                  className={`message ${role} ${first ? "first" : ""} ${last ? "last" : ""} ${isPending ? "sending" : ""} ${msg.id === match?.id ? "find-match" : ""}`}
                  aria-label={
                    msg.id === match?.id
                      ? `Matching message ${matchIndex + 1} of ${matches.length}`
                      : undefined
                  }
                  title={stamp(msg.created_at)}
                  data-message-id={msg.id}
                >
                  <h3 className="sr-only">{senderName}</h3>
                  {msg.reply_to_message_id && (
                    <button
                      type="button"
                      className="message-reply-ref"
                      disabled={!repliedTo}
                      onClick={() => repliedTo && jumpToMessage(repliedTo.id)}
                    >
                      <CornerUpLeft size={12} aria-hidden="true" />
                      {repliedTo ? (
                        <>
                          <strong>{repliedTo.sender_kind === "user" ? "You" : displayName}</strong>
                          <span>{messageSnippet(repliedTo.content, 80)}</span>
                        </>
                      ) : (
                        <span>Replying to an earlier message</span>
                      )}
                    </button>
                  )}
                  <div className="message-bubble-row">
                    <div className="body">
                      <Markdown>{msg.content}</Markdown>
                    </div>
                    {!isPending && (
                      <button
                        type="button"
                        className="icon message-reply-action"
                        aria-label={`Reply to ${senderName}`}
                        title="Reply"
                        onClick={() => startReply(msg)}
                      >
                        <CornerUpLeft size={14} />
                      </button>
                    )}
                  </div>
                  {msg.handoff_suppressed && (
                    <footer>
                      <span className="tag warn">Reply paused (turn limit reached)</span>
                    </footer>
                  )}
                </article>
                {index === lastUser && receipt && (
                  <p className="receipt" aria-live="polite">
                    {receipt}
                  </p>
                )}
              </Fragment>
            );
          })}

          {replierReplying && (
            <>
              <div className="typing" aria-hidden="true">
                <i />
                <i />
                <i />
              </div>
              <p className="notice" role="status">
                {displayName} is replying…
              </p>
            </>
          )}
        </div>
      </div>

      {(away || query) && (
        <button type="button" className="jump-latest" onClick={jumpToLatest}>
          <ArrowDown size={14} />
          {find.open ? "Close find and jump to latest" : "Jump to latest"}
        </button>
      )}

      {error && (
        <div className="banner inline" role="alert">
          {error}
          <button type="button" disabled={sending} onClick={() => void refresh(false)}>
            <RotateCw size={14} />
            Reload
          </button>
        </div>
      )}

      {capBanner && (
        <div className="banner inline" role="status">
          <AlertTriangle size={15} aria-hidden="true" />
          <span>{capBanner}</span>
        </div>
      )}

      {pendingOutbox && !sending && (
        <div className="banner inline" role="status">
          <div>
            <strong>Unsent message</strong>
            <p>{pendingOutbox.content}</p>
          </div>
          <button type="button" disabled={!storageReady} onClick={() => void submitMessage(pendingOutbox)}>
            Retry
          </button>
          <button
            type="button"
            onClick={() => {
              setDraft(pendingOutbox.content);
              setReplyTo(
                pendingOutbox.reply_to_message_id
                  ? messagesById.get(pendingOutbox.reply_to_message_id) ?? null
                  : null,
              );
              setPendingOutbox(null);
              try {
                sessionStorage.removeItem(outboxKey);
              } catch {
                // Ignore storage error.
              }
            }}
          >
            Restore to draft
          </button>
        </div>
      )}

      <div className="composer-wrap">
        {voiceError && <p role="alert">{voiceError}</p>}
        {voice && (
          <p role="status">
            {voice === "requesting"
              ? "Waiting for microphone permission…"
              : voice === "recording"
                ? "Recording · up to 20 seconds. Stop to transcribe."
                : "Transcribing…"}
          </p>
        )}
        {!storageReady && (
          <p className="attention" role="status" title={status?.tsuchi.message}>
            <span className="dot warn" />
            Storage offline · check Kadan Server
          </p>
        )}
        {replyTo && (
          <div className="gc-replying-to direct-replying-to">
            <CornerUpLeft size={14} aria-hidden="true" />
            <span className="gc-replying-to-text">
              Replying to <strong>{replyTo.sender_kind === "user" ? "You" : displayName}</strong>
              <span className="quiet"> {messageSnippet(replyTo.content, 60)}</span>
            </span>
            <button
              type="button"
              className="icon"
              aria-label="Cancel reply"
              onClick={() => setReplyTo(null)}
            >
              <X size={14} />
            </button>
          </div>
        )}
        <form className={`composer ${sending ? "is-sending" : ""}`} onSubmit={handleSend}>
          <label className="sr-only" htmlFor="message">
            Message {displayName}
          </label>
          <textarea
            id="message"
            ref={composerRef}
            autoFocus
            rows={1}
            maxLength={16000}
            disabled={!storageReady}
            placeholder={`Message ${displayName}`}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape" && replyTo) {
                event.preventDefault();
                setReplyTo(null);
                return;
              }
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                handleSend();
              }
            }}
          />
          <div className="composer-bar">
            <span />
            <div className="send-actions">
              {ready.chat && (
                <button
                  className="icon"
                  type="button"
                  aria-label={
                    voice === "recording" ? "Stop recording and transcribe" : "Record a voice message"
                  }
                  title={
                    microphoneUnavailable() ??
                    "Record a voice message · review the text before sending"
                  }
                  disabled={
                    sending ||
                    voice === "requesting" ||
                    voice === "transcribing" ||
                    !!microphoneUnavailable()
                  }
                  onClick={recordVoice}
                >
                  {voice === "recording" ? <Square size={16} /> : <Mic size={16} />}
                </button>
              )}
              {voice && (
                <button
                  className="icon"
                  type="button"
                  aria-label="Discard recording"
                  onClick={() => {
                    microphone.current?.cancel();
                    microphone.current = null;
                    setVoice(null);
                  }}
                >
                  <X size={16} />
                </button>
              )}
              <button
                type="submit"
                className="send"
                disabled={locked || !draft.trim() || !storageReady}
                aria-label="Send message"
                title="Send · Enter"
              >
                <ArrowUp size={18} />
              </button>
            </div>
          </div>
        </form>
      </div>

      {petFor && (
        <PetEditor
          agentKey={agentKey}
          agentName={petFor.display_name}
          selected={petForJob}
          onSelect={(petId) => selectParticipantPet(petFor, petId)}
          onRefreshSelection={() => refresh(true)}
          onClose={() => setPetFor(null)}
        />
      )}
    </section>
  );
}
