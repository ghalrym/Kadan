import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  AtSign,
  Bot,
  Check,
  Copy,
  Download,
  KeyRound,
  Pause,
  Pencil,
  Play,
  Plus,
  ShieldOff,
  Users,
  X,
} from "lucide-react";
import { api } from "./api";
import {
  applyMention,
  filterMentionCandidates,
  findActiveMention,
  formatLastSeen,
  isParticipantActive,
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
import { useAgentPet } from "./PetContext";
import type { Status } from "./types";
import { randomUuid } from "./uuid";

interface Props {
  agentKey: string;
  agentName: string;
  status: Status | null;
}

const DRAFT_KEY = "kadan:group-chat:draft";
const OUTBOX_KEY = "kadan:group-chat:pending";
const POLL_MS = 2000;
const NEAR_LATEST = 80;

interface PendingGroupChatMessage {
  client_request_id: string;
  content: string;
  handoff_to_participant_ids: string[];
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

export default function GroupChat({ agentKey, agentName, status }: Props) {
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

  const activeRef = useRef(true);
  const storageReady = status?.tsuchi.state !== "unavailable";
  const participants = snapshot?.participants ?? [];
  const activeParticipants = participants.filter(isParticipantActive);
  const participantsById = new Map(participants.map((item) => [item.id, item]));

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
  }, []);

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
      setNotice(`Added ${created.participant.display_name}. Copy its token now — it won't be shown again.`);
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

  return (
    <section className="group-chat" aria-label="Group Chat">
      <header className="group-chat-head">
        <div className="group-chat-title">
          <h1>{snapshot?.room.title || "Group Chat"}</h1>
          <span className="quiet">
            {activeParticipants.length}{" "}
            {activeParticipants.length === 1 ? "active agent" : "active agents"}
            {snapshot &&
              ` · ${snapshot.room.consecutive_agent_turns}/${snapshot.room.max_consecutive_agent_turns} agent turns`}
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
              messages.map((msg) => {
                const isUser = msg.sender_kind === "user";
                const isKadan = msg.sender_kind === "kadan_agent";
                const senderParticipant = msg.participant_id
                  ? participantsById.get(msg.participant_id)
                  : null;
                const handoffTargets = msg.handoff_to_participant_ids.map((id) => {
                  const target = participantsById.get(id);
                  return target ? `@${target.handle}` : "@agent";
                });
                return (
                  <article
                    key={msg.id}
                    className={`gc-message ${isUser ? "is-user" : isKadan ? "is-kadan" : "is-external"}`}
                  >
                    <header className="gc-message-head">
                      <span className="gc-sender">
                        {isKadan ? (
                          <span className="gc-avatar" aria-hidden="true">
                            <Pet sheet={petSheet} state="idle" size={22} />
                          </span>
                        ) : !isUser ? (
                          <span className="gc-avatar gc-avatar-ext" aria-hidden="true">
                            <Bot size={14} />
                          </span>
                        ) : null}
                        <strong>{isUser ? "You" : msg.sender_name}</strong>
                        {senderParticipant && (
                          <span className="gc-handle">@{senderParticipant.handle}</span>
                        )}
                        {!isUser && (
                          <span className={`gc-role-badge ${isKadan ? "kadan" : "external"}`}>
                            {isKadan ? "Kadan" : "External"}
                          </span>
                        )}
                      </span>
                      <time className="gc-time" dateTime={msg.created_at}>
                        {formatMessageTime(msg.created_at)}
                      </time>
                    </header>
                    <div className="gc-message-content">
                      <Markdown>{msg.content}</Markdown>
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
                  </article>
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
            {activeParticipants.length > 0 && (
              <div className="gc-tag-bar" aria-label="Tag an agent to reply">
                <span className="quiet gc-tag-label">
                  <AtSign size={13} aria-hidden="true" />
                  Wake:
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
                  {effectiveTags.length > 0
                    ? `Will wake ${effectiveTags
                        .map((id) => participantsById.get(id)?.display_name || "agent")
                        .join(", ")}`
                    : "No agent tagged — posts to the room without waking anyone."}
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

            {snapshot?.mcp_url && (
              <div className="gc-connection-box">
                <h3>Connect ChatGPT</h3>
                <p className="quiet">
                  Download the plugin package and upload it in ChatGPT Plugins, or paste this
                  address when adding a custom plugin. ChatGPT will open Kadan in your browser so
                  you can pick or name the agent.
                </p>
                <a
                  className="gc-plugin-download"
                  href="/kadan/api/group-chat/chatgpt-plugin.zip"
                  download="kadan-group-chat-plugin.zip"
                >
                  <Download size={14} aria-hidden="true" />
                  Download ChatGPT plugin
                </a>
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
                    aria-label="Dismiss token card"
                    onClick={() => setCredential(null)}
                  >
                    <X size={14} />
                  </button>
                </div>
                <p className="quiet">
                  Copy this bearer token into your external agent now. For safety, it will not be
                  shown again.
                </p>
                <div className="gc-copy-row">
                  <code>{credential.bearerToken}</code>
                  <button
                    type="button"
                    className="icon"
                    aria-label="Copy bearer token"
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
                Create agent token
              </button>
            </form>

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
                          <div>
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
                        {isExternal && !isRevoked && (
                          <div className="gc-participant-actions">
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
                                  `Rotated token for ${participant.display_name}. Copy the new token now.`,
                                )
                              }
                            >
                              <KeyRound size={13} />
                              Rotate token
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
          </aside>
        )}
      </div>
    </section>
  );
}
