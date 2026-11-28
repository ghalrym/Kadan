import { useCallback, useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import { sortDirectChats, type GroupChatDirectChat, type GroupChatParticipant, type GroupChatSnapshot } from "./groupChat";
import { DEFAULT_PROFILE, type UserProfile } from "./profile";
import type { McpServer } from "./integrations";
import { recentItems } from "./spaceItems";
import type { ConversationSummary, SpaceItem } from "./types";

const READ_KEY = "kadan:group-chat:read-seq";
const SUMMARY_POLL_MS = 15000;

export interface GroupChatSummary {
  participants: GroupChatParticipant[];
  latestSeq: number;
}

function readSeq(): number {
  try {
    return Number(localStorage.getItem(READ_KEY)) || 0;
  } catch {
    return 0;
  }
}

/** Who is in Group Chat and whether it has messages the person has not seen, for the sidebar. */
export function useGroupChatSummary(enabled: boolean, viewing: boolean) {
  const [summary, setSummary] = useState<GroupChatSummary | null>(null);
  const [seen, setSeen] = useState(readSeq);
  const viewingRef = useRef(viewing);
  viewingRef.current = viewing;

  const record = useCallback((snapshot: Pick<GroupChatSnapshot, "participants" | "latest_seq">, read: boolean) => {
    setSummary({ participants: snapshot.participants, latestSeq: snapshot.latest_seq });
    if (!read) return;
    setSeen((current) => {
      if (snapshot.latest_seq <= current) return current;
      try {
        localStorage.setItem(READ_KEY, String(snapshot.latest_seq));
      } catch {
        // The dot may reappear after a reload; nothing else depends on it.
      }
      return snapshot.latest_seq;
    });
  }, []);

  useEffect(() => {
    if (!enabled || viewing) return;
    let alive = true;
    const poll = async () => {
      try {
        const snapshot = await api<GroupChatSnapshot>("/group-chat?limit=1");
        if (alive && !viewingRef.current) record(snapshot, false);
      } catch {
        // The sidebar keeps the last summary until Group Chat answers again.
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), SUMMARY_POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [enabled, viewing, record]);

  /** Group Chat reports each snapshot it shows; showing it means the person has seen it. */
  const onSnapshot = useCallback((snapshot: GroupChatSnapshot) => record(snapshot, true), [record]);
  return { summary, unread: !viewing && !!summary && summary.latestSeq > seen, onSnapshot };
}

/** The person's name and pet, loaded once Kadan knows its agent. */
export function useProfile(enabled: boolean) {
  const [profile, setProfile] = useState<UserProfile>(DEFAULT_PROFILE);
  const reload = useCallback(async () => {
    setProfile(await api<UserProfile>("/profile"));
  }, []);
  useEffect(() => {
    if (enabled) void reload().catch(() => {});
  }, [enabled, reload]);
  return { profile, setProfile, reload };
}

const CHAT_SEEN_KEY = "kadan:chat:seen:";

function readChatSeen(chatId: string): number | null {
  try {
    const raw = localStorage.getItem(CHAT_SEEN_KEY + chatId);
    return raw === null ? null : Number(raw) || 0;
  } catch {
    return null;
  }
}

function writeChatSeen(chatId: string, count: number) {
  try {
    localStorage.setItem(CHAT_SEEN_KEY + chatId, String(count));
  } catch {
    // The dot may reappear after a reload; nothing else depends on it.
  }
}

/** Whether the agent's chat gained messages since the person last had it open. */
export function useChatUnread(agentKey: string, chatId: string | null, shownCount: number | null, viewing: boolean) {
  const [seen, setSeen] = useState<number | null>(null);
  const [latest, setLatest] = useState<number | null>(null);

  useEffect(() => {
    setSeen(chatId ? readChatSeen(chatId) : null);
    setLatest(null);
  }, [chatId]);

  useEffect(() => {
    if (!chatId || !viewing || shownCount === null) return;
    writeChatSeen(chatId, shownCount);
    setSeen(shownCount);
  }, [chatId, viewing, shownCount]);

  useEffect(() => {
    if (!agentKey || !chatId || viewing) return;
    let alive = true;
    const poll = async () => {
      try {
        const rows = await api<ConversationSummary[]>(`${agentPath(agentKey)}/conversations`);
        const count = rows.find((row) => row.id === chatId)?.message_count;
        if (!alive || count === undefined) return;
        setLatest(count);
        // Nothing to compare against yet: what exists now counts as seen.
        setSeen((current) => {
          if (current !== null) return current;
          writeChatSeen(chatId, count);
          return count;
        });
      } catch {
        // Keep the last answer until the server responds again.
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), SUMMARY_POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [agentKey, chatId, viewing]);

  return !viewing && seen !== null && latest !== null && latest > seen;
}

export interface NavPreviews {
  recentSpace: SpaceItem[];
  plugins: McpServer[];
}

/** Recent Space items and installed plugins for the sidebar, refreshed after leaving those pages. */
export function useNavPreviews(agentKey: string, pageName: string): NavPreviews {
  const [recentSpace, setRecentSpace] = useState<SpaceItem[]>([]);
  const [plugins, setPlugins] = useState<McpServer[]>([]);
  const onSpace = pageName === "space";
  const onPlugins = pageName === "plugins";
  useEffect(() => {
    if (!agentKey || onSpace) return;
    let alive = true;
    api<SpaceItem[]>(`${agentPath(agentKey)}/space/items`)
      .then((items) => { if (alive) setRecentSpace(recentItems(items, 3)); })
      .catch(() => {});
    return () => { alive = false; };
  }, [agentKey, onSpace]);
  useEffect(() => {
    if (!agentKey || onPlugins) return;
    let alive = true;
    api<McpServer[]>("/mcp/servers")
      .then((servers) => { if (alive) setPlugins(servers); })
      .catch(() => {});
    return () => { alive = false; };
  }, [agentKey, onPlugins]);
  return { recentSpace, plugins };
}

const DIRECT_SEEN_KEY = "kadan:direct:seen:";

function readDirectSeen(): Record<string, number> {
  const seen: Record<string, number> = {};
  try {
    for (let index = 0; index < localStorage.length; index++) {
      const key = localStorage.key(index);
      if (key?.startsWith(DIRECT_SEEN_KEY)) seen[key.slice(DIRECT_SEEN_KEY.length)] = Number(localStorage.getItem(key)) || 0;
    }
  } catch {
    // Unread dots may reappear after a reload; nothing else depends on them.
  }
  return seen;
}

/** Private chats with each active external agent, newest first, with unread state. */
export function useDirectChats(enabled: boolean, viewingId: string | null) {
  const [chats, setChats] = useState<GroupChatDirectChat[]>([]);
  const [seen, setSeen] = useState<Record<string, number>>(readDirectSeen);
  const [loaded, setLoaded] = useState(false);

  const markSeen = useCallback((participantId: string, seq: number) => {
    setSeen((current) => {
      if ((current[participantId] ?? 0) >= seq) return current;
      try {
        localStorage.setItem(DIRECT_SEEN_KEY + participantId, String(seq));
      } catch {
        // See readDirectSeen.
      }
      return { ...current, [participantId]: seq };
    });
  }, []);

  const refresh = useCallback(async () => {
    try {
      setChats(sortDirectChats(await api<GroupChatDirectChat[]>("/group-chat/direct")));
      setLoaded(true);
    } catch {
      // Keep the last list until the server answers again.
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
    const timer = setInterval(() => void refresh(), SUMMARY_POLL_MS);
    return () => clearInterval(timer);
  }, [enabled, refresh]);

  /** The open private chat reports each snapshot it shows, which marks it read. */
  const onSnapshot = useCallback((snapshot: GroupChatSnapshot) => {
    const participantId = snapshot.room.direct_participant_id;
    if (!participantId) return;
    markSeen(participantId, snapshot.latest_seq);
    setChats((current) => {
      const index = current.findIndex((chat) => chat.participant.id === participantId);
      if (index < 0 || current[index].latest_seq === snapshot.latest_seq) return current;
      const last = snapshot.messages[snapshot.messages.length - 1];
      const next = [...current];
      next[index] = { ...next[index], latest_seq: snapshot.latest_seq, last_message_at: last?.created_at ?? null };
      return sortDirectChats(next);
    });
  }, [markSeen]);

  const unread = new Set(
    chats
      .filter((chat) => chat.participant.id !== viewingId && chat.latest_seq > (seen[chat.participant.id] ?? 0))
      .map((chat) => chat.participant.id),
  );
  return { chats, loaded, unread, onSnapshot };
}
