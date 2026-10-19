import type { AgentRun, Asset, ImageJob, Resource } from "./types.ts";

export const activeRun = (run: AgentRun) => run.status === "queued" || run.status === "running";
export const cancellableImage = (job: ImageJob) => job.status === "queued" || job.status === "running";

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
