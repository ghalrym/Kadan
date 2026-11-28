import { useCallback, useEffect, useRef, useState } from "react";
import { agentPath, api } from "./api";
import type { GroupChatParticipant, GroupChatSnapshot } from "./groupChat";
import { DEFAULT_PROFILE, type UserProfile } from "./profile";
import type { ConversationSummary } from "./types";

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
