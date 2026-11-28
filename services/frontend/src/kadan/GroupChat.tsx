import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type MouseEvent } from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  AtSign,
  Check,
  CornerUpLeft,
  Copy,
  Download,
  KeyRound,
  Palette,
  Pause,
  Pencil,
  Play,
  Plus,
  ShieldOff,
  Users,
  X,
} from "lucide-react";
import { api } from "./api";
import { isDesktopApp, saveDesktopDownload } from "./desktop";
import {
  applyMention,
  filterMentionCandidates,
  findActiveMention,
  formatLastSeen,
  groupChatRows,
  isAwaitingReply,
  isParticipantActive,
  messageSnippet,
  participantStatusBadge,
  resolveTaggedParticipantIds,
  turnCapNotice,
  type ActiveMention,
  type GroupChatCreateParticipantResponse,
  type GroupChatMessage,
  type GroupChatParticipant,
  type GroupChatSnapshot,
  type GroupChatUpdateParticipantResponse,
  type GroupChatUserMessageResponse,
} from "./groupChat";
import Markdown from "./Markdown";
import Pet from "./Pet";
import { ParticipantAvatar, UserAvatar } from "./PetAvatar";
import { useAgentPet, usePetRefJob } from "./PetContext";
import PetEditor from "./PetEditor";
import { USER_SEED, type PetRef, type UserProfile } from "./profile";
import { agentHue } from "./pet";
import type { Status } from "./types";
import { randomUuid } from "./uuid";

interface Props {
  agentKey: string;
  agentName: string;
  status: Status | null;
  profile: UserProfile;
  /** Every snapshot shown here; the sidebar uses it for members and unread state. */
  onSnapshot?: (snapshot: GroupChatSnapshot) => void;
  onEditProfile: () => void;
  onEditAgentPet: () => void;
}

const DRAFT_KEY = "kadan:group-chat:draft";
const OUTBOX_KEY = "kadan:group-chat:pending";
const POLL_MS = 2000;
const NEAR_LATEST = 80;

interface PendingGroupChatMessage {
  client_request_id: string;
  content: string;
  handoff_to_participant_ids: string[];
  reply_to_message_id?: string | null;
}

interface OneTimeCredential {
  participantName: string;
  handle: string;
  mcpUrl: string;
  bearerToken: string;
  rotated: boolean;
}

function readPendingOutbox(): PendingGroupChatMessage | null {
  try {
    const raw = sessionStorage.getItem(OUTBOX_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PendingGroupChatMessage>;
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

function formatMessageTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function formatFullTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString();
}

export default function GroupChat({
  agentKey,
  agentName,
  status,
  profile,
  onSnapshot,
  onEditProfile,
  onEditAgentPet,
}: Props) {
  const petSheet = useAgentPet(agentKey);
  const [snapshot, setSnapshot] = useState<GroupChatSnapshot | null>(null);
  const [messages, setMessages] = useState<GroupChatMessage[]>([]);
  const [hasOlder, setHasOlder] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [drawerOpen, setDrawerOpen] = useState(false);

  const [draft, setDraft] = useState(() => {
    try {
      return sessionStorage.getItem(DRAFT_KEY) || "";
    } catch {
      return "";
    }
  });
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [pendingOutbox, setPendingOutbox] = useState<PendingGroupChatMessage | null>(() =>
    readPendingOutbox(),
  );
  const [sending, setSending] = useState(false);

  // @mention autocomplete state
  const [mention, setMention] = useState<ActiveMention | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  // Reading position / scroll state
  const listRef = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const [away, setAway] = useState(false);

  // External agent setup / management state
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [managing, setManaging] = useState(false);
  const [credential, setCredential] = useState<OneTimeCredential | null>(null);
  const [copiedField, setCopiedField] = useState<"url" | "token" | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [confirmRevokeId, setConfirmRevokeId] = useState<string | null>(null);
  const [downloadingPlugin, setDownloadingPlugin] = useState(false);
  const [pluginDownloadNotice, setPluginDownloadNotice] = useState("");
  const [petFor, setPetFor] = useState<GroupChatParticipant | null>(null);
  const petForJob = usePetRefJob(petFor?.pet);
  const [replyTo, setReplyTo] = useState<GroupChatMessage | null>(null);

  const activeRef = useRef(true);
  const storageReady = status?.tsuchi.state !== "unavailable";
  const participants = snapshot?.participants ?? [];
  const activeParticipants = participants.filter(isParticipantActive);
  const participantsById = new Map(participants.map((item) => [item.id, item]));
  const messagesById = new Map(messages.map((item) => [item.id, item]));
  const mentionHandles = new Set(participants.map((item) => item.handle.toLowerCase()));
  const kadanParticipant = participants.find((item) => item.kind === "kadan_agent" && isParticipantActive(item));
  const kadanReplying = !!kadanParticipant && isAwaitingReply(messages, kadanParticipant.id);

  const mentionCandidates = mention
    ? filterMentionCandidates(participants, mention.query)
    : [];
  const safeMentionIndex =
    mentionCandidates.length > 0 ? mentionIndex % mentionCandidates.length : 0;

  const effectiveTags = resolveTaggedParticipantIds(draft, selectedTags, participants);
  const capBanner = snapshot ? turnCapNotice(snapshot.room) : null;

  const scrollToLatest = useCallback(() => {
    const node = listRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, []);

  const mergeLatestSnapshot = useCallback((next: GroupChatSnapshot) => {
    setSnapshot(next);
    onSnapshot?.(next);
    setMessages((prev) => {
      if (!prev.length) {
        setHasOlder(next.has_more);
        return next.messages;
      }
      const byId = new Map(prev.map((item) => [item.id, item]));
      for (const msg of next.messages) byId.set(msg.id, msg);
      const merged = [...byId.values()].sort((a, b) => a.seq - b.seq);
      return merged;
    });
    setPendingOutbox((current) => {
      if (!current) return null;
      if (next.messages.some((m) => m.client_request_id === current.client_request_id)) {
        try {
          sessionStorage.removeItem(OUTBOX_KEY);
        } catch {
          // Ignore storage error.
        }
        return null;
      }
      return current;
    });
  }, [onSnapshot]);

  const refresh = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      try {
        const next = await api<GroupChatSnapshot>("/group-chat?limit=100");
        if (!activeRef.current) return;
        mergeLatestSnapshot(next);
        setError("");
      } catch (e) {
        if (activeRef.current) {
          setError(`Could not load Group Chat: ${(e as Error).message}`);
        }
      } finally {
        if (activeRef.current && !silent) setLoading(false);
      }
    },
    [mergeLatestSnapshot],
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

  useEffect(() => {
    try {
      if (draft) sessionStorage.setItem(DRAFT_KEY, draft);
      else sessionStorage.removeItem(DRAFT_KEY);
    } catch {
      // Ignore storage error.
    }
  }, [draft]);

  useLayoutEffect(() => {
    if (followLatest.current) scrollToLatest();
  }, [messages.length, scrollToLatest]);

  useLayoutEffect(() => {
    const node = composerRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 200)}px`;
  }, [draft]);

  async function loadOlderMessages() {
    if (loadingOlder || !messages.length) return;
    setLoadingOlder(true);
    try {
      const oldestSeq = messages[0].seq;
      const older = await api<GroupChatSnapshot>(
        `/group-chat?before_seq=${oldestSeq}&limit=50`,
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

  function updateDraft(nextValue: string, cursorPos?: number) {
    setDraft(nextValue);
    const pos = cursorPos ?? nextValue.length;
    const nextMention = findActiveMention(nextValue, pos);
    setMention(nextMention);
    setMentionIndex(0);
  }

  function selectMention(participant: GroupChatParticipant) {
    if (!mention) return;
    const applied = applyMention(draft, mention, participant.handle);
    setDraft(applied.text);
    setMention(null);
    setMentionIndex(0);
    setSelectedTags((prev) =>
      prev.includes(participant.id) ? prev : [...prev, participant.id],
    );
    requestAnimationFrame(() => {
      const node = composerRef.current;
      if (node) {
        node.focus();
        node.setSelectionRange(applied.cursor, applied.cursor);
      }
    });
  }

  function toggleParticipantChip(participant: GroupChatParticipant) {
    setSelectedTags((prev) =>
      prev.includes(participant.id)
        ? prev.filter((id) => id !== participant.id)
        : [...prev, participant.id],
    );
  }

  async function submitMessage(payload: PendingGroupChatMessage) {
    if (sending) return;
    setSending(true);
    setError("");
    setPendingOutbox(payload);
    try {
      sessionStorage.setItem(OUTBOX_KEY, JSON.stringify(payload));
    } catch {
      // Ignore storage error.
    }
    try {
      const result = await api<GroupChatUserMessageResponse>(
        "/group-chat/messages",
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
        sessionStorage.removeItem(OUTBOX_KEY);
      } catch {
        // Ignore storage error.
      }
      setPendingOutbox(null);
      setDraft("");
      setSelectedTags([]);
      setReplyTo(null);
      setMention(null);
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
      followLatest.current = true;
      setAway(false);
    } catch (e) {
      if (activeRef.current) {
        setError(`Could not send message: ${(e as Error).message}`);
      }
    } finally {
      if (activeRef.current) setSending(false);
    }
  }

  function handleSend(event?: React.FormEvent) {
    event?.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || sending || !storageReady) return;
    void submitMessage({
      client_request_id: randomUuid(),
      content: trimmed,
      handoff_to_participant_ids: effectiveTags,
      reply_to_message_id: replyTo?.id ?? null,
    });
  }

  async function handleAddExternalAgent(event: React.FormEvent) {
    event.preventDefault();
    if (!newName.trim() || managing) return;
    setManaging(true);
    setError("");
    setNotice("");
    try {
      const created = await api<GroupChatCreateParticipantResponse>(
        "/group-chat/participants",
        {
          display_name: newName.trim(),
          description: newDescription.trim(),
        },
        "POST",
      );
      if (!activeRef.current) return;
      setNewName("");
      setNewDescription("");
      setCredential({
        participantName: created.participant.display_name,
        handle: created.participant.handle,
        mcpUrl: created.mcp_url,
        bearerToken: created.bearer_token,
        rotated: false,
      });
      setNotice(`Added ${created.participant.display_name}. Copy its connection key now. It won't be shown again.`);
      await refresh(true);
    } catch (e) {
      if (activeRef.current) setError((e as Error).message);
    } finally {
      if (activeRef.current) setManaging(false);
    }
  }

  async function handleUpdateParticipant(
    participant: GroupChatParticipant,
    patch: {
      display_name?: string;
      description?: string;
      enabled?: boolean;
      rotate_token?: boolean;
      revoke?: boolean;
    },
    successMessage: string,
  ) {
    if (managing) return;
    setManaging(true);
    setError("");
    setNotice("");
    try {
      const updated = await api<GroupChatUpdateParticipantResponse>(
        `/group-chat/participants/${encodeURIComponent(participant.id)}`,
        {
          ...patch,
          expected_version: participant.version,
        },
        "PUT",
      );
      if (!activeRef.current) return;
      if (updated.bearer_token) {
        setCredential({
          participantName: updated.participant.display_name,
          handle: updated.participant.handle,
          mcpUrl: updated.mcp_url || snapshot?.mcp_url || "",
          bearerToken: updated.bearer_token,
          rotated: true,
        });
      }
      setEditingId(null);
      setConfirmRevokeId(null);
      setNotice(successMessage);
      await refresh(true);
    } catch (e) {
      if (activeRef.current) setError((e as Error).message);
    } finally {
      if (activeRef.current) setManaging(false);
    }
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
      // A lost response may still have saved; the next snapshot shows what stuck.
      await refresh(true);
    }
  }

  async function copyText(text: string, field: "url" | "token") {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(field);
      setTimeout(() => {
        if (activeRef.current) setCopiedField((prev) => (prev === field ? null : prev));
      }, 2000);
    } catch {
      setError("Could not copy to clipboard. Select and copy the text manually.");
    }
  }

  async function handleDownloadPlugin(event: MouseEvent<HTMLAnchorElement>) {
    if (!isDesktopApp()) return;
    event.preventDefault();
    if (downloadingPlugin) return;
    setDownloadingPlugin(true);
    setPluginDownloadNotice("");
    setError("");
    try {
      const result = await saveDesktopDownload(
        "/kadan/api/group-chat/chatgpt-plugin.zip",
        "kadan-group-chat-plugin.zip",
      );
      if (!activeRef.current) return;
      if (result.saved) {
        const savedName = result.filename || "kadan-group-chat-plugin.zip";
        const message = result.supported
          ? `Saved ${savedName}. Upload it in ChatGPT Plugins to connect.`
          : `Saved ${savedName} to your Downloads folder. Upload it in ChatGPT Plugins to connect.`;
        setPluginDownloadNotice(message);
        setNotice(message);
      }
    } catch (e) {
      if (activeRef.current) {
        setError(
          (e as Error).message || "Could not download the ChatGPT plugin. Try again.",
        );
      }
    } finally {
      if (activeRef.current) setDownloadingPlugin(false);
    }
  }

  function senderOf(msg: GroupChatMessage) {
    const isUser = msg.sender_kind === "user";
    const isKadan = msg.sender_kind === "kadan_agent";
    const participant = msg.participant_id ? participantsById.get(msg.participant_id) : null;
    const avatar = participant ?? {
      kind: isKadan ? ("kadan_agent" as const) : ("external_agent" as const),
      agent_key: isKadan ? agentKey : null,
      handle: msg.sender_name,
      pet: null,
    };
    const name = isUser ? profile.display_name : participant?.display_name ?? msg.sender_name;
    const seed = isUser ? USER_SEED : isKadan ? agentKey : avatar.handle;
    return { isUser, isKadan, participant, avatar, name, seed };
  }

  function senderAvatar(msg: GroupChatMessage, size: number) {
    const sender = senderOf(msg);
    return sender.isUser
      ? <UserAvatar profile={profile} size={size} />
      : <ParticipantAvatar participant={sender.avatar} size={size} />;
  }

  function jumpToMessage(id: string) {
    const node = listRef.current?.querySelector<HTMLElement>(`[data-message-id="${id}"]`);
    if (!node) return;
    node.scrollIntoView({ block: "center", behavior: "smooth" });
    node.classList.add("flash");
    setTimeout(() => node.classList.remove("flash"), 1200);
  }

  function startReply(msg: GroupChatMessage) {
    setReplyTo(msg);
    const sender = senderOf(msg);
    if (!sender.isUser && sender.participant && isParticipantActive(sender.participant)) {
      setSelectedTags((prev) => (prev.includes(sender.participant!.id) ? prev : [...prev, sender.participant!.id]));
    }
    requestAnimationFrame(() => composerRef.current?.focus());
  }

  return (
    <section className="group-chat" aria-label="Group Chat">
      <header className="group-chat-head">
        <span className="member-stack gc-head-members" aria-hidden="true">
          <UserAvatar profile={profile} size={28} />
          {activeParticipants.slice(0, 4).map((participant) => (
            <ParticipantAvatar key={participant.id} participant={participant} size={28} />
          ))}
        </span>
        <div className="group-chat-title">
          <h1>{snapshot?.room.title || "Group Chat"}</h1>
          <span className="quiet">
            {activeParticipants.length}{" "}
            {activeParticipants.length === 1 ? "agent" : "agents"}
          </span>
        </div>
        <button
          type="button"
          className={drawerOpen ? "selected" : ""}
          aria-expanded={drawerOpen}
          onClick={() => setDrawerOpen((open) => !open)}
        >
          <Users size={16} />
          Participants & Connections
        </button>
      </header>

      {error && (
        <div className="banner" role="alert">
          {error}
          <button type="button" onClick={() => setError("")}>
            Dismiss
          </button>
        </div>
      )}

      {notice && (
        <div className="banner group-chat-notice" role="status">
          {notice}
          <button type="button" onClick={() => setNotice("")}>
            Dismiss
          </button>
        </div>
      )}

      {capBanner && (
        <div className="group-chat-cap-banner" role="status">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>{capBanner}</span>
        </div>
      )}

      {pendingOutbox && !sending && (
        <div className="banner" role="alert">
          <span>Your last message may not have been sent.</span>
          <button type="button" onClick={() => void submitMessage(pendingOutbox)}>
            Retry sending
          </button>
          <button
            type="button"
            onClick={() => {
              setDraft(pendingOutbox.content);
              setSelectedTags(pendingOutbox.handoff_to_participant_ids);
              setReplyTo(
                pendingOutbox.reply_to_message_id
                  ? messagesById.get(pendingOutbox.reply_to_message_id) ?? null
                  : null,
              );
              setPendingOutbox(null);
              try {
                sessionStorage.removeItem(OUTBOX_KEY);
              } catch {
                // Ignore storage error.
              }
            }}
          >
            Restore to draft
          </button>
        </div>
      )}

      <div className={`group-chat-body ${drawerOpen ? "with-drawer" : ""}`}>
        <div className="group-chat-main">
          <div
            className="group-chat-stream"
            ref={listRef}
            tabIndex={-1}
            onScroll={() => {
              const node = listRef.current;
              if (!node) return;
              const nearBottom =
                node.scrollHeight - node.scrollTop - node.clientHeight <= NEAR_LATEST;
              followLatest.current = nearBottom;
              setAway(!nearBottom);
            }}
          >
            {hasOlder && (
              <div className="group-chat-load-older">
                <button
                  type="button"
                  disabled={loadingOlder}
                  onClick={() => void loadOlderMessages()}
                >
                  {loadingOlder ? "Loading earlier messages…" : "Load earlier messages"}
                </button>
              </div>
            )}

            {loading && !messages.length ? (
              <div className="group-chat-empty" role="status">
                Loading Group Chat…
              </div>
            ) : !messages.length ? (
              <div className="group-chat-empty">
                <Pet sheet={petSheet} state="waving" size={72} />
                <h2>Start a shared conversation</h2>
                <p className="quiet">
                  Tag {agentName} or an external agent below to ask them to reply. Messages without
                  a tag stay in the room without waking any agent.
                </p>
              </div>
            ) : (
              groupChatRows(messages).map(({ message: msg, startsBlock, dayLabel }) => {
                const { isUser, isKadan, avatar, name, seed } = senderOf(msg);
                const repliedTo = msg.reply_to_message_id ? messagesById.get(msg.reply_to_message_id) : null;
                const handoffTargets = msg.handoff_to_participant_ids.map((id) => {
                  const target = participantsById.get(id);
                  return target ? `@${target.handle}` : "@agent";
                });
                return (
                  <div key={msg.id} className="gc-row">
                    {dayLabel && (
                      <div className="gc-day" role="separator">
                        <span>{dayLabel}</span>
                      </div>
                    )}
                    <article
                      className={`gc-message ${startsBlock ? "starts-block" : "continues"} ${isUser ? "is-user" : isKadan ? "is-kadan" : "is-external"}`}
                      style={{ "--avatar-h": agentHue(seed) } as CSSProperties}
                      aria-label={`${name} at ${formatMessageTime(msg.created_at)}`}
                      data-message-id={msg.id}
                    >
                      {msg.reply_to_message_id && (
                        <button
                          type="button"
                          className="gc-reply-ref"
                          disabled={!repliedTo}
                          onClick={() => repliedTo && jumpToMessage(repliedTo.id)}
                        >
                          {repliedTo ? (
                            <>
                              {senderAvatar(repliedTo, 16)}
                              <strong>{senderOf(repliedTo).name}</strong>
                              <span>{messageSnippet(repliedTo.content)}</span>
                            </>
                          ) : (
                            <span>Replying to an earlier message</span>
                          )}
                        </button>
                      )}
                      <div className="gc-actions">
                        <button type="button" className="icon" aria-label={`Reply to ${name}`} title="Reply"
                          onClick={() => startReply(msg)}>
                          <CornerUpLeft size={15} />
                        </button>
                      </div>
                      <div className="gc-gutter">
                        {startsBlock ? (
                          isUser ? (
                            <button type="button" className="gc-avatar-button" aria-label="Edit your profile"
                              onClick={onEditProfile}>
                              <UserAvatar profile={profile} size={40} />
                            </button>
                          ) : (
                            <ParticipantAvatar participant={avatar} size={40} />
                          )
                        ) : (
                          <time className="gc-hover-time" dateTime={msg.created_at} title={formatFullTime(msg.created_at)}>
                            {formatMessageTime(msg.created_at)}
                          </time>
                        )}
                      </div>
                      <div className="gc-body">
                        {startsBlock && (
                          <header className="gc-message-head">
                            <strong className="gc-name">{name}</strong>
                            {!isUser && (
                              <span className={`gc-role-badge ${isKadan ? "kadan" : "external"}`}>
                                {isKadan ? "Kadan" : "External"}
                              </span>
                            )}
                            <time className="gc-time" dateTime={msg.created_at} title={formatFullTime(msg.created_at)}>
                              {formatMessageTime(msg.created_at)}
                            </time>
                          </header>
                        )}
                        <div className="gc-message-content">
                          <Markdown mentions={mentionHandles}>{msg.content}</Markdown>
                        </div>
                        {(handoffTargets.length > 0 || msg.handoff_suppressed) && (
                          <footer className="gc-message-foot">
                            {handoffTargets.length > 0 && (
                              <span className="gc-handoff-chip">
                                {isUser ? "Tagged " : "Handed off to "}
                                {handoffTargets.join(", ")}
                              </span>
                            )}
                            {msg.handoff_suppressed && (
                              <span className="gc-handoff-chip suppressed">
                                Handoff paused (agent turn limit reached)
                              </span>
                            )}
                          </footer>
                        )}
                      </div>
                    </article>
                  </div>
                );
              })
            )}
          </div>

          {away && (
            <button
              type="button"
              className="gc-jump-latest"
              onClick={() => {
                followLatest.current = true;
                setAway(false);
                scrollToLatest();
              }}
            >
              <ArrowDown size={14} />
              Latest
            </button>
          )}

          <div className="gc-composer-wrap">
            {kadanParticipant && kadanReplying && (
              <div className="gc-replying" role="status">
                <ParticipantAvatar participant={kadanParticipant} size={20} state="running" />
                <span>
                  <strong>{kadanParticipant.display_name}</strong> is replying
                </span>
                <span className="gc-replying-dots" aria-hidden="true"><i /><i /><i /></span>
              </div>
            )}
            {replyTo && (
              <div className="gc-replying-to">
                <CornerUpLeft size={14} aria-hidden="true" />
                <span className="gc-replying-to-text">
                  Replying to <strong>{senderOf(replyTo).name}</strong>
                  <span className="quiet"> {messageSnippet(replyTo.content, 60)}</span>
                </span>
                <button type="button" className="icon" aria-label="Cancel reply" onClick={() => setReplyTo(null)}>
                  <X size={14} />
                </button>
              </div>
            )}
            {activeParticipants.length > 0 && (
              <div className="gc-tag-bar" aria-label="Ask an agent to reply">
                <span className="quiet gc-tag-label">
                  <AtSign size={13} aria-hidden="true" />
                  Ask to reply:
                </span>
                {activeParticipants.map((participant) => {
                  const selected = effectiveTags.includes(participant.id);
                  return (
                    <button
                      key={participant.id}
                      type="button"
                      className={`gc-chip ${selected ? "selected" : ""}`}
                      aria-pressed={selected}
                      onClick={() => toggleParticipantChip(participant)}
                    >
                      <ParticipantAvatar participant={participant} size={18} />
                      <span>{participant.display_name}</span>
                      <small>@{participant.handle}</small>
                    </button>
                  );
                })}
              </div>
            )}

            {mention && mentionCandidates.length > 0 && (
              <ul className="gc-mention-menu" role="listbox" aria-label="Mention a participant">
                {mentionCandidates.map((candidate, index) => (
                  <li key={candidate.id} role="option" aria-selected={index === safeMentionIndex}>
                    <button
                      type="button"
                      className={index === safeMentionIndex ? "active" : ""}
                      onMouseDown={(event) => {
                        event.preventDefault();
                        selectMention(candidate);
                      }}
                    >
                      <ParticipantAvatar participant={candidate} size={22} />
                      <strong>{candidate.display_name}</strong>
                      <span className="quiet">@{candidate.handle}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <form className="composer gc-composer" onSubmit={handleSend}>
              <label className="sr-only" htmlFor="group-chat-input">
                Message Group Chat
              </label>
              <textarea
                id="group-chat-input"
                ref={composerRef}
                rows={1}
                maxLength={16000}
                disabled={!storageReady || sending}
                placeholder="Write a message… use @ to tag an agent or click a chip above"
                value={draft}
                onChange={(event) =>
                  updateDraft(event.target.value, event.target.selectionStart ?? undefined)
                }
                onKeyDown={(event) => {
                  if (mention && mentionCandidates.length > 0) {
                    if (event.key === "ArrowDown") {
                      event.preventDefault();
                      setMentionIndex((idx) => (idx + 1) % mentionCandidates.length);
                      return;
                    }
                    if (event.key === "ArrowUp") {
                      event.preventDefault();
                      setMentionIndex(
                        (idx) =>
                          (idx - 1 + mentionCandidates.length) % mentionCandidates.length,
                      );
                      return;
                    }
                    if (event.key === "Enter" || event.key === "Tab") {
                      event.preventDefault();
                      selectMention(mentionCandidates[safeMentionIndex]);
                      return;
                    }
                    if (event.key === "Escape") {
                      event.preventDefault();
                      setMention(null);
                      return;
                    }
                  }
                  if (event.key === "Escape" && replyTo) {
                    event.preventDefault();
                    setReplyTo(null);
                    return;
                  }
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    handleSend();
                  }
                }}
              />
              <div className="composer-bar">
                <span className="quiet gc-composer-hint">
                  {effectiveTags.length > 0 ? (
                    <>
                      <span className="member-stack" aria-hidden="true">
                        {effectiveTags.map((id) => {
                          const target = participantsById.get(id);
                          return target ? <ParticipantAvatar key={id} participant={target} size={18} /> : null;
                        })}
                      </span>
                      {`${effectiveTags
                        .map((id) => participantsById.get(id)?.display_name || "agent")
                        .join(", ")} will be asked to reply`}
                    </>
                  ) : (
                    "No one tagged. Your message is posted without asking anyone to reply."
                  )}
                </span>
                <button
                  type="submit"
                  className="send"
                  disabled={!draft.trim() || sending || !storageReady}
                  aria-label="Send message"
                >
                  <ArrowUp size={18} />
                </button>
              </div>
            </form>
          </div>
        </div>

        {drawerOpen && (
          <aside className="group-chat-drawer" aria-label="Participants and connections">
            <header className="gc-drawer-head">
              <h2>Participants & Connections</h2>
              <button
                type="button"
                className="icon"
                aria-label="Close participants panel"
                onClick={() => setDrawerOpen(false)}
              >
                <X size={16} />
              </button>
            </header>

            <div className="gc-participants-list">
              <h3>In this room</h3>
              {participants.map((participant) => {
                const badge = participantStatusBadge(participant);
                const isExternal = participant.kind === "external_agent";
                const isRevoked = participant.revoked_at !== null;
                const isEditing = editingId === participant.id;
                return (
                  <div
                    key={participant.id}
                    className={`gc-participant-card ${isRevoked ? "is-revoked" : ""}`}
                  >
                    {isEditing ? (
                      <form
                        className="gc-edit-form"
                        onSubmit={(e) => {
                          e.preventDefault();
                          void handleUpdateParticipant(
                            participant,
                            {
                              display_name: editName.trim(),
                              description: editDescription.trim(),
                            },
                            `Updated ${editName.trim()}.`,
                          );
                        }}
                      >
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          maxLength={200}
                          required
                        />
                        <input
                          type="text"
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          placeholder="Short role description"
                          maxLength={500}
                        />
                        <div className="gc-participant-actions">
                          <button type="submit" className="primary" disabled={!editName.trim() || managing}>
                            Save
                          </button>
                          <button type="button" onClick={() => setEditingId(null)}>
                            Cancel
                          </button>
                        </div>
                      </form>
                    ) : (
                      <>
                        <div className="gc-participant-top">
                          <ParticipantAvatar participant={participant} size={36} />
                          <div className="gc-participant-name">
                            <strong>{participant.display_name}</strong>{" "}
                            <span className="quiet">@{participant.handle}</span>
                          </div>
                          <span className={`gc-status-pill ${badge.tone}`}>{badge.label}</span>
                        </div>
                        {participant.description && (
                          <p className="quiet">{participant.description}</p>
                        )}
                        <p className="field-hint">{participant.push_status_detail}</p>
                        {isExternal && (
                          <p className="field-hint">
                            {formatLastSeen(participant.last_seen_at)}
                          </p>
                        )}
                        {!isExternal && (
                          <div className="gc-participant-actions">
                            <button type="button" onClick={onEditAgentPet}>
                              <Palette size={13} />
                              Edit pet
                            </button>
                          </div>
                        )}
                        {isExternal && !isRevoked && (
                          <div className="gc-participant-actions">
                            <button type="button" disabled={managing} onClick={() => setPetFor(participant)}>
                              <Palette size={13} />
                              Edit pet
                            </button>
                            <button
                              type="button"
                              disabled={managing}
                              onClick={() => {
                                setEditingId(participant.id);
                                setEditName(participant.display_name);
                                setEditDescription(participant.description);
                              }}
                            >
                              <Pencil size={13} />
                              Rename
                            </button>
                            <button
                              type="button"
                              disabled={managing}
                              onClick={() =>
                                void handleUpdateParticipant(
                                  participant,
                                  { enabled: !participant.enabled },
                                  participant.enabled
                                    ? `Paused ${participant.display_name}.`
                                    : `Resumed ${participant.display_name}.`,
                                )
                              }
                            >
                              {participant.enabled ? <Pause size={13} /> : <Play size={13} />}
                              {participant.enabled ? "Pause" : "Resume"}
                            </button>
                            <button
                              type="button"
                              disabled={managing}
                              onClick={() =>
                                void handleUpdateParticipant(
                                  participant,
                                  { rotate_token: true },
                                  `Made a new connection key for ${participant.display_name}. Copy it now.`,
                                )
                              }
                            >
                              <KeyRound size={13} />
                              New connection key
                            </button>
                            {confirmRevokeId === participant.id ? (
                              <button
                                type="button"
                                className="danger"
                                disabled={managing}
                                onClick={() =>
                                  void handleUpdateParticipant(
                                    participant,
                                    { revoke: true },
                                    `Revoked access for ${participant.display_name}.`,
                                  )
                                }
                              >
                                <ShieldOff size={13} />
                                Confirm revoke
                              </button>
                            ) : (
                              <button
                                type="button"
                                className="danger"
                                disabled={managing}
                                onClick={() => setConfirmRevokeId(participant.id)}
                              >
                                <ShieldOff size={13} />
                                Revoke
                              </button>
                            )}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {snapshot?.mcp_url && (
              <div className="gc-connection-box">
                <h3>Connect ChatGPT</h3>
                <p className="quiet">
                  Download the plugin package and upload it in ChatGPT Plugins, or paste this
                  address when adding a custom plugin. ChatGPT will open Kadan in your browser so
                  you can pick or name the agent.
                </p>
                <a
                  className={`gc-plugin-download${downloadingPlugin ? " is-busy" : ""}`}
                  href="/kadan/api/group-chat/chatgpt-plugin.zip"
                  download="kadan-group-chat-plugin.zip"
                  aria-disabled={downloadingPlugin || undefined}
                  onClick={(event) => void handleDownloadPlugin(event)}
                >
                  <Download size={14} aria-hidden="true" />
                  {downloadingPlugin ? "Saving plugin…" : "Download ChatGPT plugin"}
                </a>
                {pluginDownloadNotice && (
                  <p className="quiet gc-download-status" role="status">
                    {pluginDownloadNotice}
                  </p>
                )}
                <span className="field-hint">Group Chat address</span>
                <div className="gc-copy-row">
                  <code>{snapshot.mcp_url}</code>
                  <button
                    type="button"
                    className="icon"
                    aria-label="Copy Group Chat address"
                    onClick={() => void copyText(snapshot.mcp_url, "url")}
                  >
                    {copiedField === "url" ? <Check size={14} /> : <Copy size={14} />}
                  </button>
                </div>
              </div>
            )}

            {credential && (
              <div className="gc-credential-card" role="status">
                <div className="gc-credential-head">
                  <strong>
                    {credential.rotated
                      ? `New token for ${credential.participantName}`
                      : `Connect ${credential.participantName} (@${credential.handle})`}
                  </strong>
                  <button
                    type="button"
                    className="icon"
                    aria-label="Dismiss connection key"
                    onClick={() => setCredential(null)}
                  >
                    <X size={14} />
                  </button>
                </div>
                <p className="quiet">
                  Copy this connection key into your external agent now. For safety, it will not be
                  shown again.
                </p>
                <div className="gc-copy-row">
                  <code>{credential.bearerToken}</code>
                  <button
                    type="button"
                    className="icon"
                    aria-label="Copy connection key"
                    onClick={() => void copyText(credential.bearerToken, "token")}
                  >
                    {copiedField === "token" ? <Check size={14} /> : <Copy size={14} />}
                  </button>
                </div>
              </div>
            )}

            <form className="gc-add-form" onSubmit={(e) => void handleAddExternalAgent(e)}>
              <h3>Add external agent</h3>
              <label>
                Display name
                <input
                  type="text"
                  placeholder="e.g. Dot"
                  maxLength={200}
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  required
                />
              </label>
              <label>
                Short role description (optional)
                <input
                  type="text"
                  placeholder="e.g. Research & coding teammate"
                  maxLength={500}
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                />
              </label>
              <button type="submit" className="primary" disabled={!newName.trim() || managing}>
                <Plus size={15} />
                Create connection key
              </button>
            </form>

          </aside>
        )}
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
