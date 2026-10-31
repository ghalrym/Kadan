export interface CodingJob {
  id: string;
  status: "starting" | "running" | "stopping" | "completed" | "failed" | "cancelled" | "timed_out" | "interrupted" | "cleanup_required";
  cwd: string;
  session_id: string | null;
  created_at: string;
  message: string;
  events: { sequence: number; kind: string; message: string }[];
  events_truncated?: boolean;
}

export interface CodingStatus {
  setup: { state: string; message: string; version?: string | null };
  jobs: CodingJob[];
}

export const codingLabels: Record<CodingJob["status"], string> = {
  starting: "Starting", running: "Working", stopping: "Stopping and cleaning up…",
  completed: "Completed", failed: "Failed", cancelled: "Cancelled", timed_out: "Timed out",
  interrupted: "Interrupted", cleanup_required: "Cleanup needs attention",
};

export const codingBusy = (job: CodingJob) =>
  ["starting", "running", "stopping", "cleanup_required"].includes(job.status);

export function currentCodingJob(jobs: CodingJob[]): CodingJob | undefined {
  return jobs.find(codingBusy) ?? jobs[0];
}
