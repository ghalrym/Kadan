import type { Message, ToolExecution } from "./types.ts";

export type ConversationActivity =
  | { kind: "message"; message: Message }
  | { kind: "tool"; message: Message; execution: ToolExecution | null };

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readExecution(content: string): ToolExecution | null {
  try {
    const value: unknown = JSON.parse(content);
    if (
      !object(value) ||
      typeof value.id !== "string" ||
      !value.id ||
      typeof value.name !== "string" ||
      !value.name ||
      !object(value.arguments) ||
      !["started", "completed", "error"].includes(String(value.status)) ||
      (value.result != null && !object(value.result))
    )
      return null;
    return value as unknown as ToolExecution;
  } catch {
    return null;
  }
}

export function conversationActivity(
  messages: Message[],
): ConversationActivity[] {
  const activities: ConversationActivity[] = [];
  const tools = new Map<string, number>();
  for (const message of messages) {
    if (message.role !== "tool") {
      activities.push({ kind: "message", message });
      continue;
    }
    const execution = readExecution(message.content);
    const previousIndex = execution ? tools.get(execution.id) : undefined;
    if (previousIndex !== undefined) {
      const previous = activities[previousIndex];
      // A durable outcome must not become a running action on replay.
      if (
        execution?.status !== "started" ||
        (previous.kind === "tool" && previous.execution?.status === "started")
      ) {
        activities[previousIndex] = { kind: "tool", message, execution };
      }
    } else {
      if (execution) tools.set(execution.id, activities.length);
      activities.push({ kind: "tool", message, execution });
    }
  }
  return activities;
}

export function toolStatus(execution: ToolExecution, active: boolean): string {
  if (execution.status === "completed") return "Completed";
  if (execution.status === "error") return "Failed";
  return active ? "Running…" : "Outcome unknown";
}
