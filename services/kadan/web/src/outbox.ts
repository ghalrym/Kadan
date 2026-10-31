import type { SelectedReference } from "./RequestReferences";

export interface PendingMessage {
  id: string;
  content: string;
  generate: boolean;
  persona: string;
  attachment_ids?: string[];
  references?: SelectedReference[];
}

export function chatStorageKeys(agentKey: string, conversationId: string) {
  const draft = `kadan.draft.${agentKey}`;
  return { draft, pending: `${draft}.${conversationId}.pending` };
}

export function readPending(raw: string | null): PendingMessage | null {
  try {
    const value = JSON.parse(raw || "null");
    return value && typeof value.id === "string" &&
      typeof value.content === "string" && typeof value.generate === "boolean" &&
      typeof value.persona === "string" &&
      (value.references === undefined ||
        (Array.isArray(value.references) && value.references.length <= 3 &&
          value.references.every((item: SelectedReference) => item && typeof item.resource_id === "string" &&
            Number.isInteger(item.expected_version) && item.expected_version >= 1 && typeof item.title === "string") &&
          new Set(value.references.map((item: SelectedReference) => item.resource_id)).size === value.references.length)) &&
      (value.attachment_ids === undefined ||
        (Array.isArray(value.attachment_ids) && value.attachment_ids.length <= 100 &&
          value.attachment_ids.every((id: unknown) => typeof id === "string") &&
          new Set(value.attachment_ids).size === value.attachment_ids.length)) ? value : null;
  } catch {
    return null;
  }
}

export function copyPendingToDraft(draft: string, pending: PendingMessage): string {
  const text = draft ? `${draft}\n${pending.content}` : pending.content;
  if (text.length > 16000) {
    throw new Error("The copied text would exceed the message limit. Shorten the draft before copying.");
  }
  return text;
}

export function discardUnassignedPending(storage: Pick<Storage, "removeItem">, agentKey: string) {
  storage.removeItem(`kadan.draft.${agentKey}.pending`);
}

export function queueSend(pending: PendingMessage | null, next: PendingMessage): PendingMessage {
  if (pending && pending.id !== next.id) {
    throw new Error("Retry or discard the unsent message before sending another.");
  }
  return pending || next;
}

export function confirmSend(
  pending: PendingMessage | null,
  messages: readonly { id: string }[],
): PendingMessage | null {
  return pending && !messages.some((message) => message.id === pending.id) ? pending : null;
}
