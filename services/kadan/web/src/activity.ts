import type { AgentRun, ImageJob, Resource } from "./types.ts";

export interface ActivityItem {
  id: string;
  kind: "response" | "image";
  title: string;
  state: string;
  detail: string;
  updatedAt: string;
}

export interface ActivityGroup {
  title: string;
  items: ActivityItem[];
}

const excerpt = (text: string) => Array.from(text.trim()).slice(0, 140).join("");

/** Derive the inbox from the same persisted snapshot shown by chat. */
export function conversationActivity(conversation: Resource, runs: AgentRun[], images: ImageJob[]): ActivityGroup[] {
  const groups: ActivityGroup[] = [
    { title: "Needs review", items: [] },
    { title: "In progress", items: [] },
    { title: "Finished", items: [] },
  ];
  const messages = new Map(conversation.messages.map(message => [message.id, message]));
  for (const run of runs) {
    if (run.conversation_id !== conversation.id) continue;
    const request = run.trigger_message_id ? messages.get(run.trigger_message_id) : undefined;
    const item: ActivityItem = {
      id: run.id,
      kind: "response",
      title: request ? excerpt(request.content) || "Response" : run.trigger_job_id ? "Image follow-up response" : "Response",
      state: "",
      detail: "",
      updatedAt: run.updated_at,
    };
    switch (run.status) {
      case "failed":
        item.state = "Response failed";
        item.detail = run.error || "No failure details were recorded.";
        groups[0].items.push(item);
        break;
      case "cancelled":
        item.state = "Response stopped";
        item.detail = "Already delivered messages remain. Stopping does not undo actions.";
        groups[0].items.push(item);
        break;
      case "queued":
      case "running":
        item.state = run.status === "queued" ? "Response queued" : "Waiting for response";
        groups[1].items.push(item);
        break;
      case "completed":
        item.state = "Response finished";
        item.detail = run.delivered_message_ids.length
          ? `${run.delivered_message_ids.length} message ${run.delivered_message_ids.length === 1 ? "delivery" : "deliveries"} recorded.`
          : "No message delivery was recorded for this response.";
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
      state: "",
      detail: "",
      updatedAt: image.updated_at,
    };
    switch (image.status) {
      case "failed":
        item.state = "Image failed";
        item.detail = image.error || "No failure details were recorded.";
        groups[0].items.push(item);
        break;
      case "cancelled":
        item.state = "Image cancelled";
        item.detail = "Cancelled separately from chat responses.";
        groups[0].items.push(item);
        break;
      case "queued":
      case "running":
      case "cancelling":
        item.state = image.status === "queued" ? "Image queued" : image.status === "running" ? "Generating image" : "Stopping image";
        groups[1].items.push(item);
        break;
      case "completed":
        item.state = image.delivered_message_id ? "Image delivered" : "Image ready · awaiting delivery";
        item.detail = image.delivered_message_id ? "Image delivery to this conversation was recorded." : "Generation finished; delivery is still pending.";
        groups[image.delivered_message_id ? 2 : 1].items.push(item);
        break;
    }
  }
  for (const group of groups) {
    group.items.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt) || a.kind.localeCompare(b.kind) || a.id.localeCompare(b.id));
  }
  return groups;
}
