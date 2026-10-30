import type { AgentRun, Asset, ImageJob, Resource, Status } from "./types.ts";

export const activeRun = (run: AgentRun) => run.status === "queued" || run.status === "running";
export const cancellableImage = (job: ImageJob) => job.status === "queued" || job.status === "running";

export function currentResponseOutcome(conversation: Resource, runs: AgentRun[]): AgentRun | undefined {
  const latest = runs.filter(run => run.conversation_id === conversation.id).at(-1);
  if (!latest || (latest.status !== "failed" && latest.status !== "cancelled")) return;
  const lastUser = conversation.messages.filter(message => message.role === "user").at(-1);
  if (lastUser && latest.trigger_message_id !== lastUser.id && lastUser.created_at > latest.created_at) return;
  return latest;
}

export function responseProgress(runs: AgentRun[], hikari: Status["hikari"] | undefined): string | null {
  if (!runs.some(activeRun)) return null;
  if (hikari?.state === "model_required" && hikari.download_state === "downloading")
    return "Hikari is downloading the language model…";
  if (!runs.some((run) => run.status === "running")) return "Response queued…";
  if (hikari?.state === "ready" && hikari.model_state === "loading")
    return "Hikari is loading the language model…";
  return "Waiting for response…";
}

export function imageStatus(job: ImageJob): string | null {
  switch (job.status) {
    case "queued": return "Image queued";
    case "running": return "Generating image…";
    case "cancelling": return "Stopping image…";
    case "cancelled": return "Image cancelled";
    case "failed": return `Image failed: ${job.error || "Generation failed."}`;
    case "completed": return job.delivered_message_id ? null : "Image ready · awaiting delivery";
  }
}

export function isImageAsset(asset: Asset): boolean {
  return asset.is_image && asset.media_type.startsWith("image/");
}

/** A poll can finish after a newer send; never replace confirmed messages with an older snapshot. */
export function latestConversation(current: Resource, incoming: Resource): Resource {
  return current.id === incoming.id && incoming.version > current.version ? incoming : current;
}
