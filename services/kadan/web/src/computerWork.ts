export interface ComputerTask {
  id: string;
  instructions: string;
  origin: { conversation_id: string; run_id: string } | null;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  summary: string | null;
  updated_at: string;
}

/** What the computer agent is doing; its transcript is never sent to the browser. */
export interface ComputerWork {
  task: ComputerTask | null;
  queued: number;
  pending_steering: number;
  last?: ComputerTask | null;
}

/** How long a finished task's result stays in the status before it reads Idle. */
export const RESULT_MS = 2 * 60 * 1000;
const RESULT_LABELS: Partial<Record<ComputerTask["status"], string>> = {
  completed: "Done",
  failed: "Failed",
  cancelled: "Stopped",
};

export function workLabel(work: ComputerWork | null, now = Date.now()): string {
  if (!work?.task) {
    const last = work?.last;
    const label = last && RESULT_LABELS[last.status];
    if (!last || !label || now - Date.parse(last.updated_at) > RESULT_MS) return "Idle";
    return last.summary ? `${label}: ${last.summary}` : label;
  }
  const parts = [work.task.status === "queued" ? "Starting" : "Working"];
  if (work.pending_steering) parts.push("reading your request");
  if (work.queued) parts.push(`${work.queued} more queued`);
  return parts.join(" · ");
}
