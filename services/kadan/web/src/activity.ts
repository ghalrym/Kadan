import type { AgentRun, ImageJob, Resource } from "./types.ts";
import { currentResponseOutcome } from "./agentRuns.ts";

export type ActivityTone = "failed" | "stopped" | "active" | "done";

export interface ActivityItem {
  id: string;
  kind: "response" | "image";
  tone: ActivityTone;
  title: string;
  state: string;
  detail: string;
  updatedAt: string;
}

export interface ActivityGroup {
  title: string;
  items: ActivityItem[];
  historical?: boolean;
}

const excerpt = (text: string) => Array.from(text.trim()).slice(0, 140).join("");

/** Derive the inbox from the same persisted snapshot shown by chat. */
export function conversationActivity(conversation: Resource, runs: AgentRun[], images: ImageJob[]): ActivityGroup[] {
  const groups: ActivityGroup[] = [
    { title: "Needs review", items: [] },
    { title: "In progress", items: [] },
    { title: "Finished", items: [] },
    { title: "Earlier", items: [], historical: true },
  ];
  const currentOutcome = currentResponseOutcome(conversation, runs);
  const messages = new Map(conversation.messages.map(message => [message.id, message]));
  for (const run of runs) {
    if (run.conversation_id !== conversation.id) continue;
    const request = run.trigger_message_id ? messages.get(run.trigger_message_id) : undefined;
    const item: ActivityItem = {
      id: run.id,
      kind: "response",
      title: request ? excerpt(request.content) || "Response" : run.trigger_job_id ? "Image follow-up response" : "Response",
      tone: "done",
      state: "",
      detail: "",
      updatedAt: run.updated_at,
    };
    switch (run.status) {
      case "failed":
        item.tone = "failed";
        item.state = "Response failed";
        item.detail = run.error || "";
        groups[run.id === currentOutcome?.id ? 0 : 3].items.push(item);
        break;
      case "cancelled":
        item.tone = "stopped";
        item.state = "Response stopped";
        groups[run.id === currentOutcome?.id ? 0 : 3].items.push(item);
        break;
      case "queued":
      case "running":
        item.tone = "active";
        item.state = run.status === "queued" ? "Response queued" : "Responding";
        groups[1].items.push(item);
        break;
      case "completed":
        item.state = "Response finished";
        if (!run.delivered_message_ids.length) item.detail = "No message sent";
        groups[2].items.push(item);
        break;
    }
  }
  for (const image of images) {
    if (image.conversation_id !== conversation.id) continue;
    const item: ActivityItem = {
      id: image.id,
      kind: "image",
      title: excerpt(image.prompt) || "Image",
      tone: "done",
      state: "",
      detail: "",
      updatedAt: image.updated_at,
    };
    switch (image.status) {
      case "failed":
        item.tone = "failed";
        item.state = "Image failed";
        item.detail = image.error || "";
        groups[0].items.push(item);
        break;
      case "cancelled":
        item.tone = "stopped";
        item.state = "Image cancelled";
        groups[0].items.push(item);
        break;
      case "queued":
      case "running":
      case "cancelling":
        item.tone = "active";
        item.state = image.status === "queued" ? "Image queued" : image.status === "running" ? "Generating image" : "Stopping image";
        groups[1].items.push(item);
        break;
      case "completed":
        if (!image.delivered_message_id) item.tone = "active";
        item.state = image.delivered_message_id ? "Image sent" : "Image ready, not sent yet";
        groups[image.delivered_message_id ? 2 : 1].items.push(item);
        break;
    }
  }
  for (const group of groups) {
    group.items.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt) || a.kind.localeCompare(b.kind) || a.id.localeCompare(b.id));
  }
  return groups;
}
