export interface PendingMessage {
  id: string;
  content: string;
  generate: boolean;
  persona: string;
  resume_coding_job_id?: string;
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
      (value.resume_coding_job_id === undefined ||
        (value.generate && typeof value.resume_coding_job_id === "string" && value.resume_coding_job_id.length > 0)) ? value : null;
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
