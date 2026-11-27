export type GroupChatParticipantKind = "kadan_agent" | "external_agent";
export type GroupChatSenderKind = "user" | "kadan_agent" | "external_agent";
export type GroupChatPushStatus =
  | "built_in"
  | "active"
  | "idle"
  | "paused"
  | "revoked"
  | "expired"
  | "error";

export interface GroupChatRoom {
  id: string;
  title: string;
  consecutive_agent_turns: number;
  max_consecutive_agent_turns: number;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface GroupChatParticipant {
  id: string;
  room_id: string;
  handle: string;
  display_name: string;
  kind: GroupChatParticipantKind;
  agent_key: string | null;
  description: string;
  enabled: boolean;
  revoked_at: string | null;
  last_seen_at: string | null;
  cursor_seq: number;
  version: number;
  created_at: string;
  updated_at: string;
  push_status: GroupChatPushStatus;
  push_status_detail: string;
  active_subscriptions: number;
}

export interface GroupChatMessage {
  id: string;
  room_id: string;
  seq: number;
  sender_kind: GroupChatSenderKind;
  participant_id: string | null;
  sender_name: string;
  content: string;
  handoff_to_participant_ids: string[];
  handoff_suppressed: boolean;
  reply_to_message_id: string | null;
  client_request_id: string | null;
  created_at: string;
}

export interface GroupChatSnapshot {
  room: GroupChatRoom;
  participants: GroupChatParticipant[];
  messages: GroupChatMessage[];
  latest_seq: number;
  has_more: boolean;
  turn_cap_reached: boolean;
  mcp_url: string;
}

export interface GroupChatCreateParticipantResponse {
  participant: GroupChatParticipant;
  bearer_token: string;
  mcp_url: string;
}

export interface GroupChatUpdateParticipantResponse {
  participant: GroupChatParticipant;
  bearer_token: string | null;
  mcp_url: string | null;
}

export interface GroupChatUserMessageResponse {
  message: GroupChatMessage;
  room: GroupChatRoom;
  effective_handoff_to_participant_ids: string[];
  turn_cap_reached: boolean;
  handoff_suppressed: boolean;
  idempotent_replay: boolean;
}

export interface ActiveMention {
  start: number;
  end: number;
  query: string;
}

export function isParticipantActive(participant: Pick<GroupChatParticipant, "enabled" | "revoked_at">): boolean {
  return participant.enabled && participant.revoked_at === null;
}

/** Find an `@query` token immediately preceding the cursor for autocomplete. */
export function findActiveMention(text: string, cursor: number): ActiveMention | null {
  const safeCursor = Math.max(0, Math.min(cursor, text.length));
  const before = text.slice(0, safeCursor);
  const match = /(?:^|[\s(])@([a-zA-Z0-9_-]{0,48})$/.exec(before);
  if (!match) return null;
  const query = match[1];
  const start = safeCursor - query.length - 1;
  return { start, end: safeCursor, query: query.toLowerCase() };
}

export function filterMentionCandidates(
  participants: GroupChatParticipant[],
  query: string,
): GroupChatParticipant[] {
  const needle = query.trim().toLowerCase();
  return participants.filter((participant) => {
    if (!isParticipantActive(participant)) return false;
    if (!needle) return true;
    return (
      participant.handle.toLowerCase().includes(needle) ||
      participant.display_name.toLowerCase().includes(needle)
    );
  });
}

export function applyMention(
  text: string,
  mention: ActiveMention,
  handle: string,
): { text: string; cursor: number } {
  const inserted = `@${handle} `;
  const next = text.slice(0, mention.start) + inserted + text.slice(mention.end);
  return { text: next, cursor: mention.start + inserted.length };
}

/** Resolve all active participant IDs tagged via chips or explicit `@handle` text mentions. */
export function resolveTaggedParticipantIds(
  text: string,
  selectedIds: string[],
  participants: GroupChatParticipant[],
): string[] {
  const activeById = new Map<string, GroupChatParticipant>();
  const activeByHandle = new Map<string, GroupChatParticipant>();
  for (const participant of participants) {
    if (!isParticipantActive(participant)) continue;
    activeById.set(participant.id, participant);
    activeByHandle.set(participant.handle.toLowerCase(), participant);
  }
  const ordered: string[] = [];
  const seen = new Set<string>();
  const push = (id: string) => {
    if (!seen.has(id) && activeById.has(id)) {
      seen.add(id);
      ordered.push(id);
    }
  };
  for (const id of selectedIds) push(id);
  const pattern = /(?:^|[\s(])@([a-zA-Z0-9][a-zA-Z0-9_-]{0,47})\b/g;
  for (const match of text.matchAll(pattern)) {
    const found = activeByHandle.get(match[1].toLowerCase());
    if (found) push(found.id);
  }
  return ordered;
}

export function participantStatusBadge(
  participant: Pick<GroupChatParticipant, "push_status" | "last_seen_at">,
): { label: string; tone: "ok" | "warn" | "error" | "muted" } {
  switch (participant.push_status) {
    case "built_in":
      return { label: "Built-in agent", tone: "ok" };
    case "active":
      return { label: "Push wake-ups active", tone: "ok" };
    case "paused":
      return { label: "Paused", tone: "warn" };
    case "expired":
      return { label: "Subscription expired", tone: "warn" };
    case "error":
      return { label: "Delivery error", tone: "error" };
    case "revoked":
      return { label: "Revoked", tone: "muted" };
    case "idle":
    default:
      return participant.last_seen_at
        ? { label: "Connected (no push)", tone: "muted" }
        : { label: "Not connected yet", tone: "muted" };
  }
}

export function formatLastSeen(value: string | null, now = new Date()): string {
  if (!value) return "Never connected";
  const then = new Date(value);
  if (Number.isNaN(then.getTime())) return "Recently";
  const seconds = Math.max(0, Math.round((now.getTime() - then.getTime()) / 1000));
  if (seconds < 45) return "Active just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `Last seen ${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `Last seen ${hours}h ago`;
  const days = Math.round(hours / 24);
  return `Last seen ${days}d ago`;
}

export function turnCapNotice(room: Pick<GroupChatRoom, "consecutive_agent_turns" | "max_consecutive_agent_turns">): string | null {
  if (room.consecutive_agent_turns < room.max_consecutive_agent_turns) return null;
  return `Agent replies are paused after reaching the ${room.max_consecutive_agent_turns}-turn limit. Send a message to continue.`;
}
