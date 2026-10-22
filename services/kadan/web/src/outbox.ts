export interface PendingMessage {
  id: string;
  content: string;
  generate: boolean;
  persona: string;
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
      typeof value.persona === "string" ? value : null;
  } catch {
    return null;
  }
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
