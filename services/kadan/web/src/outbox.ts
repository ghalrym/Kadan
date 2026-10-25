import type { SelectedReference } from "./RequestReferences";
import type { WorkflowSelection } from "./workflows";

export interface PendingMessage {
  id: string;
  content: string;
  generate: boolean;
  persona: string;
  references: SelectedReference[];
  workflow?: WorkflowSelection;
}

export function readPending(raw: string | null): PendingMessage | null {
  try {
    const value = JSON.parse(raw || "null");
    if (value && value.references === undefined) value.references = [];
    return value && typeof value.id === "string" &&
      typeof value.content === "string" && typeof value.generate === "boolean" &&
      typeof value.persona === "string" &&
      Array.isArray(value.references) && value.references.length <= 3 &&
      value.references.every((item: SelectedReference) => typeof item.resource_id === "string" &&
        Number.isInteger(item.expected_version) && item.expected_version >= 1 && typeof item.title === "string") &&
      (value.workflow === undefined || (value.generate === true &&
        value.workflow && typeof value.workflow.id === "string" &&
        Number.isInteger(value.workflow.version) && value.workflow.version >= 1)) ? value : null;
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
