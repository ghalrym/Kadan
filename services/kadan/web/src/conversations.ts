import type { ConversationSummary, Resource } from "./types.ts";

/** Stored titles that mean the chat has not been named yet. */
const DEFAULT_TITLES = new Set(["Chain", "New chat"]);
const TITLE_LENGTH = 60;
const PREVIEW_LENGTH = 160;

const collapse = (text: string) => text.replace(/\s+/g, " ").trim();
const clip = (text: string, length: number) =>
  text.length > length ? `${text.slice(0, length - 1).trimEnd()}…` : text;

/** Newest activity first, matching the service's ordering. */
export function sortConversations(rows: ConversationSummary[]): ConversationSummary[] {
  return [...rows].sort((a, b) =>
    b.updated_at.localeCompare(a.updated_at) || a.id.localeCompare(b.id));
}

/** Summarize a loaded conversation so the sidebar can follow it without another read. */
export function summarize(conversation: Resource, spaceId: string | null): ConversationSummary {
  const first = conversation.messages.find((message) => message.role === "user" && message.content.trim());
  const stored = conversation.title.trim();
  const preview = first ? clip(collapse(first.content), PREVIEW_LENGTH) : "";
  return {
    id: conversation.id,
    agent_key: conversation.agent_key,
    title: DEFAULT_TITLES.has(stored) && first ? clip(collapse(first.content), TITLE_LENGTH) : stored,
    space_id: spaceId,
    message_count: conversation.messages.length,
    preview,
    version: conversation.version,
    created_at: conversation.created_at,
    updated_at: conversation.updated_at,
  };
}

/** Merge a fresher summary into the list, ignoring stale versions. */
export function upsertConversation(rows: ConversationSummary[], value: ConversationSummary): ConversationSummary[] {
  const current = rows.find((row) => row.id === value.id);
  if (current && current.version > value.version) return rows;
  return sortConversations([...rows.filter((row) => row.id !== value.id), value]);
}

/** The chat to open: the preferred one while it exists, else the latest outside Spaces, else the latest. */
export function pickConversation(rows: ConversationSummary[], preferred: string | null): string | null {
  if (preferred && rows.some((row) => row.id === preferred)) return preferred;
  const sorted = sortConversations(rows);
  return (sorted.find((row) => !row.space_id) ?? sorted[0])?.id ?? null;
}

/** Chats shown in a list: started chats plus the open one, even while it is still empty. */
export function listedConversations(
  rows: ConversationSummary[],
  spaceId: string | null,
  selectedId: string | null,
): ConversationSummary[] {
  return sortConversations(rows).filter((row) =>
    row.space_id === spaceId && (row.message_count > 0 || row.id === selectedId));
}

/** An unused chat in the same place, reused instead of piling up empty chats. */
export function emptyConversation(rows: ConversationSummary[], spaceId: string | null): ConversationSummary | null {
  return sortConversations(rows).find((row) => row.space_id === spaceId && row.message_count === 0) ?? null;
}

/** Compact age for list rows: "now", "5m", "3h", "2d", then a short date. */
export function relativeTime(value: string, now = Date.now()): string {
  const seconds = Math.max(0, (now - new Date(value).getTime()) / 1000);
  if (seconds < 60) return "now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)}d`;
  const date = new Date(value);
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() === new Date(now).getFullYear() ? {} : { year: "numeric" }),
  });
}
